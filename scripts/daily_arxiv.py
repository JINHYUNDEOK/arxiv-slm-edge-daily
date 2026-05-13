import os
import re
import json
import html
import time
import random
import difflib
import requests
import feedparser
from io import BytesIO
from pypdf import PdfReader
from datetime import datetime, timezone, timedelta
from fpdf import FPDF


# =========================================================
# 기본 설정
# =========================================================

KST = timezone(timedelta(hours=9))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
PROCESSED_PATH = "processed_ids.json"

# arXiv에서 가져올 최대 후보 수
# 3년까지 넓게 보려면 너무 작으면 과거 후보가 안 잡힐 수 있음
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "100"))

# GitHub Actions 공유 IP에서 arXiv 429/503이 날 수 있으므로 넉넉하게 재시도
ARXIV_RETRIES = int(os.getenv("ARXIV_RETRIES", "5"))
ARXIV_BASE_SLEEP_SEC = int(os.getenv("ARXIV_BASE_SLEEP_SEC", "90"))
ARXIV_MAX_SLEEP_SEC = int(os.getenv("ARXIV_MAX_SLEEP_SEC", "600"))
ARXIV_INITIAL_JITTER_SEC = int(os.getenv("ARXIV_INITIAL_JITTER_SEC", "180"))

# 최근 7일 -> 15일 -> 30일 -> 60일 -> 120일 -> 1년 -> 2년 -> 3년
RECENT_WINDOWS = [7, 15, 30, 60, 120, 365, 730, 1095]

# 최소 목표 논문 수
MIN_TARGET_PAPERS = 3

# Gemini에게 넘길 후보 수
GEMINI_CANDIDATE_LIMIT = 7

# 최종 PDF에 들어갈 최대 논문 수
FINAL_PAPER_LIMIT = 3

# PDF 앞부분 파싱 설정
# 후보 10개 중 상위 5개만 PDF 앞부분/서론 일부를 읽고,
# 최종 결과는 FINAL_PAPER_LIMIT에 따라 최대 3편만 PDF에 정리됨
PDF_PARSE_CANDIDATE_LIMIT = 5
PDF_INTRO_PAGES = 2
INTRO_TEXT_CHAR_LIMIT = 6000

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    "adapter routing",
    "adapter selection",
    "dynamic adapter",
    "conditional adapter",
    "mixture of adapters",
    "LoRA router",
    "LoRA routing",
    "LoRA gating",
    "dynamic LoRA",
    "conditional LoRA",
    "mixture of LoRA experts",
    "PEFT routing",
    "gating network",
    "expert routing",
    "conditional computation",
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
    "efficient LLM",
    "efficient language model",
    "model compression",
    "edge AI",
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

class ArxivTemporarilyUnavailable(RuntimeError):
    pass

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


def get_retry_after_seconds(res, fallback):
    retry_after = res.headers.get("Retry-After")

    if retry_after:
        try:
            return max(1, min(int(retry_after), ARXIV_MAX_SLEEP_SEC))
        except ValueError:
            pass

    return fallback


def sleep_before_retry(wait_sec):
    wait_sec = min(wait_sec, ARXIV_MAX_SLEEP_SEC)
    jitter = random.randint(0, 15)
    total_wait = wait_sec + jitter
    print(f"{total_wait}초 후 재시도합니다.")
    time.sleep(total_wait)


def get_with_retry(url, timeout=90, retries=ARXIV_RETRIES, sleep_sec=ARXIV_BASE_SLEEP_SEC):
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

            if res.status_code == 429:
                wait_sec = get_retry_after_seconds(res, sleep_sec * attempt)
                last_error = RuntimeError(f"arXiv 429 Too Many Requests: {url}")
                print("arXiv 429 Too Many Requests.")
                if attempt < retries:
                    sleep_before_retry(wait_sec)
                continue

            if res.status_code in [500, 502, 503, 504]:
                wait_sec = get_retry_after_seconds(res, sleep_sec * attempt)
                last_error = RuntimeError(f"arXiv 서버 오류 {res.status_code}: {url}")
                print(f"arXiv 서버 오류 {res.status_code}.")
                if attempt < retries:
                    sleep_before_retry(wait_sec)
                continue

            res.raise_for_status()
            return res

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"요청 실패 {attempt}/{retries}: {e}")

            if attempt < retries:
                wait_sec = sleep_sec * attempt
                sleep_before_retry(wait_sec)

    if last_error:
        raise ArxivTemporarilyUnavailable(str(last_error)) from last_error

    raise ArxivTemporarilyUnavailable("arXiv 요청 실패")


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

    res = get_with_retry(url, timeout=90, retries=4, sleep_sec=30)

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

        # 제출일 또는 수정일 중 더 최근 것을 기준으로 age 계산
        age_days = min(
            (now - published).days,
            (now - updated).days
        )

        # 최대 3년 범위를 벗어나면 제외
        if age_days > max(RECENT_WINDOWS):
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
            "age_days": age_days,
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
        "adapter routing",
        "adapter selection",
        "dynamic adapter",
        "conditional adapter",
        "mixture of adapters",
        "lora router",
        "lora routing",
        "lora gating",
        "dynamic lora",
        "conditional lora",
        "mixture of lora experts",
        "peft routing",
        "gating network",
        "expert routing",
        "conditional computation",
        "efficient llm",
        "efficient language model",
        "edge ai",
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
        "model compression",
    ]

    for term in high_terms:
        if term in text:
            score += 3

    for term in bonus_terms:
        if term in text:
            score += 1

    return score


def select_candidates_by_windows(filtered):
    """
    최근 7일 -> 15일 -> 30일 -> 60일 -> 120일 -> 1년 -> 2년 -> 3년 순서로 확장.
    최소 3개 후보가 확보되면 그 기간의 후보를 Gemini에 전달.
    최종 출력은 Gemini 프롬프트에서 최대 3편으로 제한.
    """
    if not filtered:
        return [], None

    for window in RECENT_WINDOWS:
        window_candidates = [
            paper for paper in filtered
            if paper.get("age_days", 999999) <= window
        ]

        window_candidates = sorted(
            window_candidates,
            key=lambda x: x["local_score"],
            reverse=True
        )

        print(f"최근 {window}일 후보 수: {len(window_candidates)}")

        if len(window_candidates) >= MIN_TARGET_PAPERS:
            return window_candidates[:GEMINI_CANDIDATE_LIMIT], window

    final_window = RECENT_WINDOWS[-1]

    final_candidates = [
        paper for paper in filtered
        if paper.get("age_days", 999999) <= final_window
    ]

    final_candidates = sorted(
        final_candidates,
        key=lambda x: x["local_score"],
        reverse=True
    )

    return final_candidates[:GEMINI_CANDIDATE_LIMIT], final_window


# =========================================================
# PDF 앞부분/서론 일부 추출
# =========================================================

def clean_extracted_pdf_text(text):
    """
    PDF에서 추출한 텍스트는 줄바꿈, 하이픈, 공백이 깨지는 경우가 많으므로
    Gemini에 넘기기 전에 가볍게 정리한다.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_intro_text_from_pdf(pdf_url, max_pages=PDF_INTRO_PAGES):
    """
    arXiv PDF를 다운로드한 뒤 앞 max_pages 페이지만 텍스트로 추출한다.
    논문 전체 요약용이 아니라, 초록만으로 부족한 선별 품질을 보완하기 위한 용도.
    """
    try:
        print(f"PDF 앞부분 추출 시도: {pdf_url}")

        res = requests.get(
            pdf_url,
            timeout=90,
            headers={
                "User-Agent": "arxiv-slm-edge-daily/1.0 (personal research automation)"
            },
        )
        res.raise_for_status()

        reader = PdfReader(BytesIO(res.content))
        texts = []

        page_count = min(len(reader.pages), max_pages)

        for i in range(page_count):
            try:
                page_text = reader.pages[i].extract_text() or ""
                texts.append(page_text)
            except Exception as e:
                print(f"PDF {i + 1}페이지 텍스트 추출 실패: {e}")

        intro_text = clean_extracted_pdf_text("\n".join(texts))

        if len(intro_text) > INTRO_TEXT_CHAR_LIMIT:
            intro_text = intro_text[:INTRO_TEXT_CHAR_LIMIT] + " ... [앞부분 텍스트 일부 생략]"

        return intro_text

    except Exception as e:
        print(f"PDF 앞부분 추출 실패: {pdf_url} | {e}")
        return ""


def enrich_candidates_with_intro_text(candidates):
    """
    Gemini에 넘기기 전에 상위 후보 일부에 대해서만 PDF 앞부분을 추출한다.
    모든 후보를 파싱하면 arXiv 요청/시간/비용이 늘어나므로 제한한다.
    """
    enriched = []

    for idx, paper in enumerate(candidates):
        paper = dict(paper)

        if idx < PDF_PARSE_CANDIDATE_LIMIT:
            intro_text = extract_intro_text_from_pdf(
                paper["pdf_url"],
                max_pages=PDF_INTRO_PAGES
            )
        else:
            intro_text = ""

        paper["intro_text"] = intro_text
        enriched.append(paper)

        # arXiv에 너무 연속 요청하지 않도록 짧게 쉼
        time.sleep(2)

    return enriched


# =========================================================
# Gemini 호출
# =========================================================

def call_gemini(prompt, max_output_tokens=20000):
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
            "maxOutputTokens": max_output_tokens,
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
                candidate = data["candidates"][0]
                finish_reason = candidate.get("finishReason", "")

                text = candidate["content"]["parts"][0]["text"].strip()

                if finish_reason == "MAX_TOKENS":
                    raise RuntimeError(
                        "Gemini 응답이 maxOutputTokens 제한으로 중간에 잘렸습니다. "
                        "요약 PDF 대신 후보 목록 PDF로 처리합니다."
                    )

                if not text:
                    raise RuntimeError(f"Gemini 응답 텍스트가 비어 있습니다: {data}")

                return text

            except Exception as e:
                raise RuntimeError(f"Gemini 응답 파싱 실패 또는 응답 불완전: {data}") from e

        last_error_text = res.text
        print(f"Gemini API error {res.status_code}: {last_error_text}")

        if res.status_code in [429, 503]:
            if attempt < 3:
                wait_sec = 60 * attempt
                print(f"{wait_sec}초 후 재시도합니다.")
                time.sleep(wait_sec)
            continue

        raise RuntimeError(
            f"Gemini API error: {res.status_code}\n{res.text}"
        )

    raise RuntimeError(
        f"모든 Gemini 모델 호출 실패:\n{last_error_text}"
    )


def gemini_judge_and_summarize(papers, selected_window):
    paper_blocks = []

    for i, p in enumerate(papers, start=1):
        intro_text = p.get("intro_text", "")

        if intro_text:
            intro_block = f"""
PDF 앞부분/서론 일부:
{intro_text}
"""
        else:
            intro_block = """
PDF 앞부분/서론 일부:
추출 실패 또는 미추출. 제목과 초록을 중심으로 판단할 것.
"""

        paper_blocks.append(
            f"""
[후보 {i}]
arXiv ID: {p['arxiv_id']}
제목: {p['title']}
저자: {p['authors']}
PDF: {p['pdf_url']}
제출일: {p['published']}
수정일: {p['updated']}
최근성: 최근 {p.get('age_days', 'N/A')}일 이내

초록:
{p['abstract']}

{intro_block}
"""
        )

    prompt = f"""
너는 AI/ML 연구자를 위한 arXiv 리서치 어시스턴트다.

핵심 목표:
SLM, small language model, edge device, on-device, TinyML, LoRA, MoE, PEFT, adapter, adapter routing, adapter selection, dynamic LoRA, LoRA gating, gating network, quantization, compression, latency, memory efficiency와 직접 관련 있는 논문만 선별한다.

이번 검색 범위:
최근 {selected_window}일 이내 후보 중에서 선별한다.

너에게 제공되는 정보:
- 제목
- 저자
- arXiv ID
- PDF 링크
- 초록
- PDF 앞부분/서론 일부 텍스트

중요:
PDF 전체를 읽은 것이 아니라, 초록과 PDF 앞부분/서론 일부를 기반으로 판단하는 것이다.
따라서 실험 결과가 초록이나 서론 일부에 명확히 없으면 추측하지 말고
"제공된 정보 기준으로는 구체적 실험 결과가 명시되지 않음"이라고 써라.

절대 규칙:
- 아래 제공된 후보 논문만 사용하라.
- 새로운 논문을 검색하거나 추가하지 마라.
- 없는 논문 제목, 없는 저자, 없는 arXiv ID를 만들지 마라.
- PDF 링크는 제공된 arXiv PDF 링크만 사용하라.
- 관련성이 낮으면 억지로 3편을 채우지 마라.
- 적합한 논문이 없으면 정확히 "선정 논문 없음"이라고만 답하라.
- 후보가 여러 개 제공되더라도 최종 출력은 반드시 최대 {FINAL_PAPER_LIMIT}편까지만 작성하라.
- 한국어로 작성하라.
- 초록 요약은 원문 번역/복사가 아니라 핵심 재구성으로 작성하라.
- PDF 앞부분/서론에서 확인되는 연구 동기와 문제의식을 반영하라.
- 논문 전체를 읽은 것처럼 단정하지 마라.

선정 기준:
1. SLM 또는 small language model과 직접 관련
2. edge device, on-device, mobile, TinyML 환경과 직접 관련
3. LoRA, MoE, PEFT, adapter, quantization, compression, latency, memory optimization 중 하나 이상과 관련
4. 입력별 LoRA/adapter/expert 선택, adapter routing, gating network, conditional computation과 관련 있으면 우선
5. 실제 엣지 배포, 메모리 절감, 추론 지연시간, 전력 효율, 경량화와 관련 있으면 우선
6. 단순 대형 LLM 일반 논문, 순수 benchmark 논문, 주제와 무관한 CV/NLP 논문은 제외
7. 최신 논문을 우선하되, 오래된 논문이라도 주제 관련성이 높으면 선정 가능

출력 형식:

**📄 논문 1**
1. **제목**:
2. **저자**:
3. **관련성 판단**:
4. **초록 요약** (3-4문장):
5. **서론/앞부분 기반 판단**:
6. **핵심 기여사항**:
7. **실험 결과**:
8. **내 연구와의 관련성**:
9. **확인해야 할 부분**:
10. **arXiv PDF**:

**📄 논문 2**
동일 형식 반복

**📄 논문 3**
동일 형식 반복

후보 논문 목록:
{chr(10).join(paper_blocks)}
"""

    return call_gemini(prompt, max_output_tokens=20000)


def translate_abstract_with_gemini(abstract):
    prompt = f"""
아래 arXiv 논문 초록을 한국어로 번역하라.

규칙:
- 원문 내용을 빠뜨리지 마라.
- 과도하게 요약하지 마라.
- 자연스러운 한국어 논문체로 번역하라.
- 없는 내용을 추가하지 마라.

초록:
{abstract}
"""

    return call_gemini(prompt, max_output_tokens=2048)


def create_candidate_fallback_summary(candidates, error_message, selected_window):
    lines = []

    lines.append("Gemini 전체 요약 실패")
    lines.append("")
    lines.append("오늘은 Gemini API 오류 또는 서버 혼잡으로 인해 전체 한국어 요약을 생성하지 못했습니다.")
    lines.append("대신 arXiv API로 수집하고 중복 제거한 후보 논문 목록을 저장합니다.")
    lines.append("각 논문에 대해 초록 원문, PDF 앞부분/서론 일부, 초록 한국어 번역을 함께 정리합니다.")
    lines.append("이 후보들은 pending 상태로 기록되므로 다음 실행 때 중복 후보로 반복 생성되지 않습니다.")
    lines.append("")
    lines.append(f"선택된 검색 범위: 최근 {selected_window}일")
    lines.append(f"오류 메시지: {error_message}")
    lines.append("")

    for idx, paper in enumerate(candidates[:FINAL_PAPER_LIMIT], start=1):
        lines.append(f"[논문 후보 {idx}]")
        lines.append(f"제목: {paper['title']}")
        lines.append(f"저자: {paper['authors']}")
        lines.append(f"arXiv ID: {paper['arxiv_id']}")
        lines.append(f"제출일: {paper['published']}")
        lines.append(f"수정일: {paper['updated']}")
        lines.append(f"최근성: 최근 {paper.get('age_days', 'N/A')}일 이내")
        lines.append(f"arXiv PDF: {paper['pdf_url']}")
        lines.append("")

        lines.append("[초록 원문]")
        lines.append(paper["abstract"])
        lines.append("")

        lines.append("[PDF 앞부분/서론 일부]")
        intro_text = paper.get("intro_text", "")
        if intro_text:
            lines.append(intro_text)
        else:
            lines.append("PDF 앞부분/서론 일부 텍스트를 추출하지 못했습니다.")
        lines.append("")

        lines.append("[초록 번역]")
        try:
            translated = translate_abstract_with_gemini(paper["abstract"])
            lines.append(translated)
        except Exception as e:
            lines.append(f"초록 번역 실패: {e}")
        lines.append("")

    return "\n".join(lines)


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
        "\u00a0": " ",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Markdown 링크를 일반 URL로 변환
    text = re.sub(r"\[(https?://[^\]]+)\]\((https?://[^)]+)\)", r"\1", text)
    text = text.replace("](", " ")

    return text


def break_long_words(text, max_len=50):
    """
    fpdf2는 공백 없는 긴 URL/토큰을 줄바꿈하지 못해서 오류가 날 수 있음.
    긴 단어를 일정 길이마다 공백으로 끊어 PDF 렌더링 오류를 방지함.
    """
    new_lines = []

    for line in text.splitlines():
        words = line.split(" ")
        fixed_words = []

        for word in words:
            if len(word) > max_len:
                chunks = [word[i:i + max_len] for i in range(0, len(word), max_len)]
                fixed_words.append(" ".join(chunks))
            else:
                fixed_words.append(word)

        new_lines.append(" ".join(fixed_words))

    return "\n".join(new_lines)


def pdf_write(pdf, text, h=7, bold=False):
    """
    fpdf2에서 multi_cell 이후 x 위치가 오른쪽으로 남아
    'Not enough horizontal space' 오류가 나는 것을 방지하는 안전 출력 함수
    """
    pdf.set_x(pdf.l_margin)

    if bold:
        pdf.set_font("Nanum", "B", 10)
    else:
        pdf.set_font("Nanum", "", 10)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable_width, h, str(text))


def create_pdf(content, filename, selected_window):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    bold_font_path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

    pdf.add_font("Nanum", "", font_path)
    pdf.add_font("Nanum", "B", bold_font_path)

    pdf.add_page()

    # 제목
    pdf.set_font("Nanum", "B", 16)
    pdf.set_x(pdf.l_margin)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable_width, 10, "SLM / Edge Device arXiv Daily Summary")

    pdf.ln(2)

    # 생성 정보
    today_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    pdf_write(pdf, f"생성일: {today_kst}", h=8)
    pdf_write(pdf, f"Gemini 모델: {GEMINI_MODEL}", h=8)
    pdf_write(pdf, f"검색 범위: 최근 {selected_window}일", h=8)

    pdf.ln(5)

    content = safe_text(content)
    content = content.replace("**", "")
    content = break_long_words(content, max_len=50)

    for line in content.splitlines():
        stripped = line.strip()

        if not stripped:
            pdf.ln(3)
            continue

        if (
            stripped.startswith("[논문]")
            or stripped.startswith("논문")
            or stripped.startswith("[논문 후보")
            or stripped.startswith("Gemini 전체 요약 실패")
            or stripped.startswith("[초록 원문]")
            or stripped.startswith("[초록 번역]")
            or stripped.startswith("[PDF 앞부분/서론 일부]")
        ):
            pdf_write(pdf, stripped, h=8, bold=True)
        else:
            pdf_write(pdf, stripped, h=7, bold=False)

    output_path = os.path.join(OUTPUT_DIR, filename)
    pdf.output(output_path)

    return output_path


# =========================================================
# processed_ids 업데이트
# =========================================================

def mark_summarized(processed, paper, today):
    for old in processed:
        if old.get("arxiv_id") == paper["arxiv_id"]:
            old["title"] = paper["title"]
            old["pdf_url"] = paper["pdf_url"]
            old["status"] = "summarized"
            old["processed_at"] = today.isoformat()
            old["model"] = GEMINI_MODEL
            return

    processed.append({
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "pdf_url": paper["pdf_url"],
        "status": "summarized",
        "processed_at": today.isoformat(),
        "model": GEMINI_MODEL,
    })


def mark_pending(processed, paper, today):
    for old in processed:
        if old.get("arxiv_id") == paper["arxiv_id"]:
            old["last_attempt_at"] = today.isoformat()
            old["attempts"] = old.get("attempts", 0) + 1
            if "status" not in old:
                old["status"] = "pending"
            return

    processed.append({
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "pdf_url": paper["pdf_url"],
        "status": "pending",
        "first_seen_at": today.isoformat(),
        "last_attempt_at": today.isoformat(),
        "attempts": 1,
        "model": GEMINI_MODEL,
    })


# =========================================================
# 메인 실행
# =========================================================

def main():
    today = datetime.now(KST)
    date_token = today.strftime("%Y%m%d")
    filename = f"{date_token}.pdf"

    processed = load_processed()

    if ARXIV_INITIAL_JITTER_SEC > 0:
        initial_wait = random.randint(0, ARXIV_INITIAL_JITTER_SEC)
        print(f"arXiv 혼잡 완화를 위해 시작 전 {initial_wait}초 대기합니다.")
        time.sleep(initial_wait)

    try:
        papers = search_arxiv()
    except ArxivTemporarilyUnavailable as e:
        print(f"arXiv 임시 오류로 오늘 실행을 건너뜁니다: {e}")
        print("GitHub Actions 실패로 처리하지 않고, 다음 스케줄에서 다시 시도합니다.")
        return

    print(f"arXiv 전체 후보 수: {len(papers)}")

    filtered = []

    for paper in papers:
        score = local_score(paper)

        if is_duplicate(paper, processed):
            print(f"중복 제외: {paper['arxiv_id']} | {paper['title']}")
            continue

        if score <= 0:
            continue

        paper["local_score"] = score
        filtered.append(paper)

    filtered = sorted(filtered, key=lambda x: x["local_score"], reverse=True)

    candidates, selected_window = select_candidates_by_windows(filtered)

    print(f"선택된 검색 기간: 최근 {selected_window}일")
    print(f"Gemini 검사용 후보 수: {len(candidates)}")

    if not candidates:
        print("새 후보 논문이 없습니다.")
        return

    print(f"PDF 앞부분 추출 대상 후보 수: {min(len(candidates), PDF_PARSE_CANDIDATE_LIMIT)}")
    candidates = enrich_candidates_with_intro_text(candidates)

    gemini_success = True

    try:
        summary = gemini_judge_and_summarize(candidates, selected_window)
    except Exception as e:
        gemini_success = False
        print(f"Gemini 요약 실패. 후보 목록 PDF를 생성합니다: {e}")
        summary = create_candidate_fallback_summary(candidates, str(e), selected_window)

    if "선정 논문 없음" in summary:
        gemini_success = False
        print("Gemini가 적합한 논문을 선정하지 않았습니다. 후보 목록 PDF를 생성합니다.")
        summary = create_candidate_fallback_summary(
            candidates,
            "Gemini가 적합한 논문을 선정하지 않았습니다.",
            selected_window
        )

    output_path = create_pdf(summary, filename, selected_window)

    if gemini_success:
        for paper in candidates:
            if paper["arxiv_id"] in summary:
                mark_summarized(processed, paper, today)

        save_processed(processed)
    else:
        print("Gemini 요약 실패/미선정 상태이므로 후보 논문을 pending 상태로 기록합니다.")

        for paper in candidates[:FINAL_PAPER_LIMIT]:
            mark_pending(processed, paper, today)

        save_processed(processed)

    print(f"PDF 생성 완료: {output_path}")
    time.sleep(1)


if __name__ == "__main__":
    main()
