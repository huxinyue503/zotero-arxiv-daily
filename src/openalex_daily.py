import os
import requests
from datetime import datetime, timedelta

# ===== 复用 zotero-arxiv-daily 的模块（按你的仓库实际路径微调）=====
from src.zotero import load_zotero_library
from src.embedding import build_interest_embedding, calc_similarity
from src.mailer import send_email
from src.llm import generate_tldr_if_needed


OPENALEX_API = "https://api.openalex.org/works"


def fetch_openalex_papers(journals, since_days=1):
    since = (datetime.utcnow() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    results = []

    for journal in journals:
        params = {
            "filter": f"from_publication_date:{since},source.display_name:{journal}",
            "per-page": 50
        }
        print(f"[OpenAlex] Fetching {journal} since {since}")
        r = requests.get(OPENALEX_API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        results.extend(data.get("results", []))

    return results


def openalex_to_paper_dict(item):
    abstract = item.get("abstract_inverted_index")
    if abstract:
        words = []
        for k, idxs in abstract.items():
            for i in idxs:
                if i >= len(words):
                    words.extend([""] * (i - len(words) + 1))
                words[i] = k
        abstract = " ".join(words)
    else:
        abstract = ""

    return {
        "title": item.get("title", ""),
        "abstract": abstract,
        "authors": [a["author"]["display_name"] for a in item.get("authorships", [])],
        "url": item.get("doi") or item.get("id"),
        "pdf_url": None,
        "source": item.get("primary_location", {}).get("source", {}).get("display_name", ""),
        "published": item.get("publication_date", "")
    }


def main():
    journals = os.getenv("OPENALEX_JOURNALS", "").splitlines()
    since_days = int(os.getenv("OPENALEX_SINCE_DAYS", "1"))
    max_paper_num = int(os.getenv("MAX_PAPER_NUM", "20"))

    print("[1/5] Loading Zotero library...")
    zotero_papers = load_zotero_library()
    interest_emb = build_interest_embedding(zotero_papers)

    print("[2/5] Fetching OpenAlex papers...")
    raw_papers = fetch_openalex_papers(journals, since_days)
    print(f"[OpenAlex] Total fetched: {len(raw_papers)}")

    print("[3/5] Normalizing papers...")
    papers = []
    for item in raw_papers:
        paper = openalex_to_paper_dict(item)
        if paper["abstract"].strip():   # 无摘要直接丢弃
            papers.append(paper)

    print(f"[Filter] With abstract: {len(papers)}")

    print("[4/5] Calculating similarity...")
    for p in papers:
        p["score"] = calc_similarity(interest_emb, p["title"], p["abstract"])

    papers = sorted(papers, key=lambda x: x["score"], reverse=True)
    papers = papers[:max_paper_num]

    print("[5/5] Generating TLDR and sending email...")
    for p in papers:
        p["tldr"] = generate_tldr_if_needed(p["title"], p["abstract"])

    send_email(papers, source="OpenAlex Journals")


if __name__ == "__main__":
    main()
