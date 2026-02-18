import os
import requests
from datetime import datetime, timedelta
from loguru import logger
from paper_openalex import OpenAlexPaper

OPENALEX_API = "https://api.openalex.org/works"


def fetch_openalex_papers():
    since_days = int(os.getenv("OPENALEX_SINCE_DAYS", "1"))
    journals = os.getenv("OPENALEX_JOURNALS", "").split(";")

    since = (datetime.utcnow() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    results = []

    for journal in journals:
        journal = journal.strip()
        if not journal:
            continue

        logger.info(f"Fetching OpenAlex: {journal} since {since}")
        params = {
            "filter": f"from_publication_date:{since},source.display_name:{journal}",
            "per-page": 50,
        }
        r = requests.get(OPENALEX_API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        results.extend(data.get("results", []))

    papers = []
    for raw in results:
        if raw.get("abstract_inverted_index"):
            papers.append(OpenAlexPaper(raw))

    logger.info(f"OpenAlex fetched papers with abstract: {len(papers)}")
    return papers
