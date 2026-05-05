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

KST = timezone(timedelta(hours=9))

OUTPUT_DIR = "outputs"
PROCESSED_PATH = "processed_ids.json"

MAX_RESULTS = 40
SELECT_LIMIT = 3
RECENT_DAYS = 60

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

KEYWORD_GROUPS = [
    "small language model",
    "small language models",
    "SLM",
    "edge device",
    "edge devices",
    "on-device",
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


def load_processed():
    if not os.path.exists(PROCESSED_PATH):
        return []
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed(processed):
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9가-힣]", "", title)
    return title


def extract_arxiv_id(url: str) -> str | None:
    if not url:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", url)
    if match:
        return match.group(1)
    return None


def is_duplicate(paper, processed):
    paper_id = paper["arxiv_id"]
    paper_title_norm = normalize_title(paper["title"])

    for old in processed:
        old_id = old.get("arxiv_id", "")
        old_title_norm = normalize_title(old.get("title", ""))

        if old_id == paper_id:
            return True

        if old_title_norm:
            sim = difflib.SequenceMatcher(None, paper_title_norm, old_title_norm).ratio()
            if sim >= 0.95:
                return True

    return False


def build_arxiv_query():
    # cs.AI, cs.LG, cs.CL, cs.CV 중심으로 검색
    # 키워드는 title/abstract 전체에서 검색
    keyword_query = " OR ".join([f'all:"{kw}"' for kw in KEYWORD_GROUPS])
    category_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
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

    res = requests.get(url, timeout=30)
    res.raise_for_status()

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
            "categories": ", ".join(getattr(entry, "tags", [])) if False else "",
        }

        papers.append(paper)

    return papers


def local_score(paper):
    text = f"{paper['title']} {paper['abstract']}".lower()

    score = 0

    high_terms = [
        "small language model",
        "slm",
        "edge",
        "on-device",
        "tinyml",
        "lora",
        "moe",
        "peft",
        "adapter",
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


def verify_pdf_exists(pdf_url):
    try:
        res = requests.head(pdf_url, timeout=15, allow_redirects=True)
        return res.status_code == 200
    except Exception:
        return False


def call_gemini(prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

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

    res = requests.post(url, json=payload, timeout=120)

    if res.status_code != 200:
        raise RuntimeError(f"Gemini API error: {res.status_code}\n{res.text}")

    data = res.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Gemini 응답 파싱 실패: {data}") from e


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
초록:
{p['abstract']}
"""
        )

    prompt = f"""
너는 AI/ML 연구자를 위한 arXiv 리서치 어시스턴트다.

중요 규칙:
- 아래 제공된 후보 논문만 사용하라.
- 새로운 논문을 검색하거나 만들어내지 마라.
- PDF 링크는 제공된 arXiv PDF 링크만 사용하라.
- SLM, small language model, edge device, on-device, TinyML, LoRA, MoE, PEFT, adapter, quantization, compression, latency, memory efficiency와 직접 관련 있는 논문만 고르라.
- 관련성이 낮으면 1편만 골라도 되고, 없으면 "선정 논문 없음"이라고 답하라.
- 최대 3편만 선정하라.

출력 형식은 반드시 아래 형식을 따른다.

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


class PDF(FPDF):
    pass


def safe_text(text):
    # 이모지 일부가 PDF 폰트에서 깨질 수 있어 제거
    text = text.replace("📄", "[논문]")
    text = text.replace("→", "->")
    return text


def create_pdf(content, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

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
    pdf.ln(5)

    pdf.set_font("Nanum", "", 10)
    content = safe_text(content)
    content = content.replace("**", "")

    for line in content.splitlines():
        if line.strip().startswith("[논문]") or line.strip().startswith("논문"):
            pdf.set_font("Nanum", "B", 12)
            pdf.multi_cell(0, 8, line)
            pdf.set_font("Nanum", "", 10)
        else:
            pdf.multi_cell(0, 7, line)

    output_path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(output_path)
    return output_path


def main():
    today = datetime.now(KST)
    date_token = today.strftime("%y%m%d")
    filename = f"SLM_Edge_arXiv_({date_token}).pdf"

    processed = load_processed()
    papers = search_arxiv()

    print(f"arXiv 후보 수: {len(papers)}")

    filtered = []
    for paper in papers:
        if is_duplicate(paper, processed):
            continue

        if local_score(paper) <= 0:
            continue

        if not verify_pdf_exists(paper["pdf_url"]):
            continue

        filtered.append(paper)

    filtered = sorted(filtered, key=local_score, reverse=True)

    # Gemini에는 너무 많이 넣지 말고 상위 10개만 전달
    candidates = filtered[:10]

    if not candidates:
        print("새 후보 논문이 없습니다.")
        return

    print(f"Gemini 검사용 후보 수: {len(candidates)}")

    summary = gemini_judge_and_summarize(candidates)

    if "선정 논문 없음" in summary:
        print("Gemini가 선정한 논문이 없습니다.")
        return

    output_path = create_pdf(summary, filename)

    # 이번 Gemini 후보로 들어간 논문 중 요약에 포함된 arXiv ID만 processed에 기록
    # 단순하게는 후보 전체를 처리 기록으로 넣어도 되지만,
    # 여기서는 summary 안에 ID가 들어간 논문만 기록함.
    already_ids = {x.get("arxiv_id") for x in processed}

    for paper in candidates:
        if paper["arxiv_id"] in summary and paper["arxiv_id"] not in already_ids:
            processed.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "processed_at": today.isoformat(),
            })

    save_processed(processed)

    print(f"PDF 생성 완료: {output_path}")
    time.sleep(1)


if __name__ == "__main__":
    main()
