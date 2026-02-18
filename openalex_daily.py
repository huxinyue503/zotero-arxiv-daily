from loguru import logger
from openalex_fetcher import fetch_openalex_papers
from recommender import recommend
from construct_email import send_email


def main():
    logger.info("Running OpenAlex Daily...")
    papers = fetch_openalex_papers()

    if not papers:
        logger.info("No OpenAlex papers found today.")
        return

    recommended = recommend(papers)
    send_email(recommended, source="OpenAlex Journals")


if __name__ == "__main__":
    main()
