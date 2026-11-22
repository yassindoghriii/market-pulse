"""Nettoyage des articles : retire le bruit, normalise titres et dates, dédoublonne."""
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from pymongo import UpdateOne

from src.db import mongo_db

# Titres qui correspondent à des pages de cotation, pas à des articles
NOISE = re.compile(
    r"stock price|quote & history|stock quote|price & news|share price|stock chart",
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def clean_title(title: str, source: str | None) -> str:
    """Retire le nom du journal ajouté par Google News à la fin du titre."""
    if source and title.endswith(f" - {source}"):
        title = title[: -len(source) - 3]
    return title.strip()


def main() -> None:
    db = mongo_db()
    raw = list(db["articles_raw"].find({}, {"_id": 0}))
    clean_collection = db["articles_clean"]
    clean_collection.create_index([("ticker", 1), ("title", 1)], unique=True)

    operations, noise_examples = [], []
    cleaned_at = datetime.now(timezone.utc)
    for article in raw:
        title = clean_title(article.get("title") or "", article.get("source"))
        if not title or NOISE.search(title):
            noise_examples.append(title)
            continue
        published_at = parsedate_to_datetime(article["published"])
        document = {
            "ticker": article["ticker"],
            "company": article["company"],
            "title": title,
            "source": article.get("source"),
            "link": article["link"],
            "published_at": published_at,
            "published_date": published_at.date().isoformat(),
            "cleaned_at": cleaned_at,
        }
        operations.append(UpdateOne(
            {"ticker": document["ticker"], "title": document["title"]},
            {"$set": document}, upsert=True,
        ))

    if operations:
        clean_collection.bulk_write(operations, ordered=False)
    total_clean = clean_collection.count_documents({})
    log.info("Articles bruts : %d | bruit retiré : %d | gardés : %d | "
             "doublons de titre fusionnés : %d | articles propres en base : %d",
             len(raw), len(noise_examples), len(operations),
             len(operations) - total_clean, total_clean)
    for example in noise_examples[:5]:
        log.info("Exemple de bruit retiré : %s", example)


if __name__ == "__main__":
    main()
