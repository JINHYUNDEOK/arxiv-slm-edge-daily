import os
import re
import json
import html
import time
import difflib
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from fpdf import FPDF

# =========================================================
# 기본 설정
# =========================================================

KST = timezone(timedelta(hours=9))

OUTPUT_DIR = "outputs"
PROCESSED_PATH = "processed_ids.json"

MAX_RESULTS = 50
SELECT_LIMIT = 3
RECENT_DAYS = 60

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini 3.1 Pro Preview
# GitHub Actions YAML에서 GEMINI_MODEL을 지정하면 그 값을 우선 사용함.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

KEYWORD_GROUPS = [
    "small language model",
    "small language models",
    "SLM",
    "edge device",
    "edge devices",
    "on-device",
    "on device",
    "mobile LLM",
    "mobile language model",
    "TinyML",
    "LoRA",
    "MoE",
    "PEFT",
    "adapter",
    "adapters",
    "quantization",
    "pruning",
    "compression",
    "memory optimization",
    "inference latency",
    "power efficiency",
    "knowledge distillation",
    "prompt distillation",
    "continual learning",
    "online learning",
    "federated learning",
]


# =========================================================
# 파일 입출력
# =========================================================

def load_processed():
    if not os.path.exists(PROCESSED_PATH):
        return []

    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_processed(processed):
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


# =========================================================
# 유틸
# =========================================================

def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9가-힣]", "", title)
    return title


def extract_arxiv_id(url: str) -> str | None:
    if not url:
        return None

    match = re.search(
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?",
        url
    )

    if match:
        return match.group(1)

    return None


def is_duplicate(paper, processed):
    paper_id = paper["arxiv_id"]
    paper_title_norm = normalize_title(paper["title"])

    for old in processed:
        old_id = old.get("arxiv_id", "")
        old_title_norm = normalize_title(old.get("title", ""))

        # 1차 기준: arXiv ID
        if old_id == paper_id:
            return True

        # 2차 기준: 제목 유사도
        if old_title_norm:
            sim = difflib.SequenceMatcher(
                None,
                paper_title_norm,
                old_title_norm
            ).ratio()

            if sim >= 0.95:
                return True

    return False
def get_with_retry(url, timeout=90, retries=3, sleep_sec=10):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"요청 시도 {attempt}/{retries}: {url}")
            res = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "arxiv-slm-edge-daily/1.0 (personal research automation)"
                },
            )
            res.raise_for_status()
            return res

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"요청 실패 {attempt}/{retries}: {e}")

            if attempt < retries:
                time.sleep(sleep_sec)

    raise last_error

def verify_pdf_exists(pdf_url):
    try:
        res = requests.head(pdf_url, timeout=15, allow_redirects=True)
        return res.status_code == 200
    except Exception:
        return False


# =========================================================
# arXiv 검색
# =========================================================

def build_arxiv_query():
    keyword_query = " OR ".join([f'all:"{kw}"' for kw in KEYWORD_GROUPS])

    category_query = (
        "cat:cs.AI OR "
        "cat:cs.LG OR "
        "cat:cs.CL OR "
        "cat:cs.CV OR "
        "cat:cs.RO"
    )

    return f"({keyword_query}) AND ({category_query})"


def search_arxiv():
    query = build_arxiv_query()
    encoded_query = requests.utils.quote(query)

    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={encoded_query}"
        f"&start=0"
        f"&max_results={MAX_RESULTS}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )

    print(f"arXiv API 호출: {url}")

    res = get_with_retry(url, timeout=90, retries=3, sleep_sec=10)

    feed = feedparser.parse(res.text)
    now = datetime.now(timezone.utc)

    papers = []

    for entry in feed.entries:
        abs_url = entry.link
        arxiv_id = extract_arxiv_id(abs_url)

        if not arxiv_id:
            continue

        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        updated = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

        if (now - updated).days > RECENT_DAYS and (now - published).days > RECENT_DAYS:
            continue

        title = html.unescape(entry.title).replace("\n", " ").strip()
        abstract = html.unescape(entry.summary).replace("\n", " ").strip()
        authors = ", ".join(author.name for author in entry.authors)

        paper = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "abs_url": abs_url,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "published": published.isoformat(),
            "updated": updated.isoformat(),
        }

        papers.append(paper)

    return papers


def local_score(paper):
    text = f"{paper['title']} {paper['abstract']}".lower()

    score = 0

    high_terms = [
        "small language model",
        "small language models",
        "slm",
        "edge",
        "on-device",
        "on device",
        "mobile llm",
        "mobile language model",
        "tinyml",
        "lora",
        "moe",
        "peft",
        "adapter",
        "adapters",
    ]

    bonus_terms = [
        "quantization",
        "pruning",
        "compression",
        "latency",
        "memory",
        "power",
        "distillation",
        "continual",
        "online",
        "federated",
    ]

    for term in high_terms:
        if term in text:
            score += 3

    for term in bonus_terms:
        if term in text:
            score += 1

    return score


# =========================================================
# Gemini 3.1 Pro 호출
# =========================================================

def call_gemini(prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            "GitHub Secrets에 GEMINI_API_KEY를 등록하세요."
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 4096,
        },
    }

    last_error_text = ""

    for attempt in range(1, 4):
        print(f"Gemini 호출 시도 {attempt}/3 - model: {GEMINI_MODEL}")

        res = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=180
        )

        if res.status_code == 200:
            data = res.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                raise RuntimeError(f"Gemini 응답 파싱 실패: {data}") from e

        last_error_text = res.text
        print(f"Gemini API error {res.status_code}: {last_error_text}")

        if res.status_code in [429, 503]:
            wait_sec = 60 * attempt
            print(f"{wait_sec}초 후 재시도합니다.")
            time.sleep(wait_sec)
            continue

        raise RuntimeError(
            f"Gemini API error: {res.status_code}\n{res.text}"
        )

    raise RuntimeError(
        f"Gemini API error after retries:\n{last_error_text}"
    )


def gemini_judge_and_summarize(papers):
    paper_blocks = []

    for i, p in enumerate(papers, start=1):
        paper_blocks.append(
            f"""
[후보 {i}]
arXiv ID: {p['arxiv_id']}
제목: {p['title']}
저자: {p['authors']}
PDF: {p['pdf_url']}
제출일: {p['published']}
수정일: {p['updated']}
초록:
{p['abstract']}
"""
        )

    prompt = f"""
너는 AI/ML 연구자를 위한 arXiv 리서치 어시스턴트다.

핵심 목표:
SLM, small language model, edge device, on-device, TinyML, LoRA, MoE, PEFT, adapter, quantization, compression, latency, memory efficiency와 직접 관련 있는 최신 논문만 선별한다.

절대 규칙:
- 아래 제공된 후보 논문만 사용하라.
- 새로운 논문을 검색하거나 추가하지 마라.
- 없는 논문 제목, 없는 저자, 없는 arXiv ID를 만들지 마라.
- PDF 링크는 제공된 arXiv PDF 링크만 사용하라.
- 관련성이 낮으면 억지로 3편을 채우지 마라.
- 적합한 논문이 없으면 정확히 "선정 논문 없음"이라고만 답하라.
- 최대 3편만 선정하라.
- 한국어로 작성하라.
- 초록 요약은 원문 번역/복사가 아니라 핵심 재구성으로 작성하라.

선정 기준:
1. SLM 또는 small language model과 직접 관련
2. edge device, on-device, mobile, TinyML 환경과 직접 관련
3. LoRA, MoE, PEFT, adapter, quantization, compression, latency, memory optimization 중 하나 이상과 관련
4. 단순 대형 LLM 일반 논문, 순수 benchmark 논문, 주제와 무관한 CV/NLP 논문은 제외

출력 형식:

**📄 논문 1**
1. **제목**:
2. **저자**:
3. **관련성 판단**:
4. **초록 요약** (3-4문장):
5. **핵심 기여사항**:
6. **실험 결과**:
7. **arXiv PDF**:

**📄 논문 2**
동일 형식 반복

**📄 논문 3**
동일 형식 반복

후보 논문 목록:
{chr(10).join(paper_blocks)}
"""

    return call_gemini(prompt)


# =========================================================
# PDF 생성
# =========================================================

class PDF(FPDF):
    pass


def safe_text(text):
    replacements = {
        "📄": "[논문]",
        "→": "->",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def create_pdf(content, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # GitHub Actions Ubuntu 환경에서 설치되는 Nanum font 경로
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    bold_font_path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

    pdf.add_font("Nanum", "", font_path)
    pdf.add_font("Nanum", "B", bold_font_path)

    pdf.add_page()
    pdf.set_font("Nanum", "B", 16)
    pdf.multi_cell(0, 10, "SLM / Edge Device arXiv Daily Summary")

    pdf.set_font("Nanum", "", 10)
    today_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    pdf.multi_cell(0, 8, f"생성일: {today_kst}")
    pdf.multi_cell(0, 8, f"Gemini 모델: {GEMINI_MODEL}")
    pdf.ln(5)

    pdf.set_font("Nanum", "", 10)

    content = safe_text(content)
    content = content.replace("**", "")

    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("[논문]") or stripped.startswith("논문"):
            pdf.set_font("Nanum", "B", 12)
            pdf.multi_cell(0, 8, stripped)
            pdf.set_font("Nanum", "", 10)
        else:
            pdf.multi_cell(0, 7, line)

    output_path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(output_path)

    return output_path


# =========================================================
# 메인 실행
# =========================================================

def main():
    today = datetime.now(KST)
    date_token = today.strftime("%Y%m%d")
    filename = f"{date_token}.pdf"

    processed = load_processed()
    papers = search_arxiv()

    print(f"arXiv 전체 후보 수: {len(papers)}")

    filtered = []

    for paper in papers:
        score = local_score(paper)

        if is_duplicate(paper, processed):
            print(f"중복 제외: {paper['arxiv_id']} | {paper['title']}")
            continue

        if score <= 0:
            continue

        if not verify_pdf_exists(paper["pdf_url"]):
            print(f"PDF 확인 실패: {paper['pdf_url']}")
            continue

        paper["local_score"] = score
        filtered.append(paper)

    filtered = sorted(filtered, key=lambda x: x["local_score"], reverse=True)

    # Gemini 2.5 Flash에는 상위 6개 후보만 전달
    candidates = filtered[:6]

    print(f"Gemini 검사용 후보 수: {len(candidates)}")

    if not candidates:
        print("새 후보 논문이 없습니다.")
        return

    summary = gemini_judge_and_summarize(candidates)

    if "선정 논문 없음" in summary:
        print("Gemini가 선정한 논문이 없습니다.")
        return

    output_path = create_pdf(summary, filename)

    # 요약 결과에 실제 포함된 arXiv ID만 processed에 기록
    already_ids = {x.get("arxiv_id") for x in processed}

    for paper in candidates:
        if paper["arxiv_id"] in summary and paper["arxiv_id"] not in already_ids:
            processed.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "processed_at": today.isoformat(),
                "model": GEMINI_MODEL,
            })

    save_processed(processed)

    print(f"PDF 생성 완료: {output_path}")
    time.sleep(1)


if __name__ == "__main__":
    main()
