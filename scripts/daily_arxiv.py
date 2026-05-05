"""
daily_arxiv.py

Fetches the latest arXiv papers on Small Language Models (SLM) and
edge-AI / on-device inference topics, deduplicates against previously
processed IDs, and writes a Markdown digest to outputs/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import arxiv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "small language model edge",
    "small language model on-device",
    "SLM edge inference",
    "on-device LLM",
    "edge AI language model",
    "tiny language model",
    "efficient language model edge computing",
]

MAX_RESULTS_PER_QUERY = 20
MAX_AUTHORS_DISPLAY = 5

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_IDS_PATH = REPO_ROOT / "processed_ids.json"
OUTPUTS_DIR = REPO_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_processed_ids() -> set:
    if PROCESSED_IDS_PATH.exists():
        with PROCESSED_IDS_PATH.open("r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_ids(ids: set) -> None:
    with PROCESSED_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, indent=2, ensure_ascii=False)


_CLIENT = arxiv.Client(num_retries=3, delay_seconds=3)


def fetch_papers(queries: list[str], max_results: int) -> list[arxiv.Result]:
    """Fetch papers from arXiv for each query and return a deduplicated list."""
    seen_ids: set[str] = set()
    results: list[arxiv.Result] = []

    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        for paper in _CLIENT.results(search):
            paper_id = paper.get_short_id()
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                results.append(paper)

    return results


def build_markdown(papers: list[arxiv.Result], date_str: str) -> str:
    lines = [
        f"# arXiv SLM & Edge Daily — {date_str}",
        "",
        f"**{len(papers)} new paper(s) found**",
        "",
    ]
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(a.name for a in paper.authors[:MAX_AUTHORS_DISPLAY])
        if len(paper.authors) > MAX_AUTHORS_DISPLAY:
            authors += " et al."
        lines += [
            f"## {i}. {paper.title}",
            "",
            f"- **Authors:** {authors}",
            f"- **Published:** {paper.published.strftime('%Y-%m-%d')}",
            f"- **arXiv ID:** [{paper.get_short_id()}]({paper.entry_id})",
            f"- **PDF:** {paper.pdf_url}",
            "",
            paper.summary.replace("\n", " ").strip(),
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids()
    all_papers = fetch_papers(SEARCH_QUERIES, MAX_RESULTS_PER_QUERY)

    new_papers = [p for p in all_papers if p.get_short_id() not in processed_ids]

    print(f"Total fetched : {len(all_papers)}")
    print(f"Already seen  : {len(all_papers) - len(new_papers)}")
    print(f"New papers    : {len(new_papers)}")

    if not new_papers:
        print("Nothing new today — skipping output file.")
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = OUTPUTS_DIR / f"{date_str}.md"

    markdown = build_markdown(new_papers, date_str)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Written: {output_path}")

    # Persist updated ID set
    new_ids = {p.get_short_id() for p in new_papers}
    processed_ids.update(new_ids)
    save_processed_ids(processed_ids)
    print(f"Saved {len(processed_ids)} processed IDs to {PROCESSED_IDS_PATH}")


if __name__ == "__main__":
    main()
