"""Collecte des articles de presse via les flux RSS de Google News."""
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser

from src.collect.prices import TICKERS

RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
RAW_DIR = Path("data/raw/news")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_news(ticker: str, company: str) -> list[dict]:
    """Récupère les derniers articles parlant d'une entreprise."""
    query = quote_plus(f'"{company}" stock')
    feed = feedparser.parse(RSS_URL.format(query=query))
    if feed.bozo and not feed.entries:
        raise ValueError(f"Flux illisible : {feed.bozo_exception}")

    collected_at = datetime.now(timezone.utc).isoformat()
    return [
        {
            "ticker": ticker,
            "company": company,
            "title": entry.get("title"),
            "link": entry.get("link"),
            "published": entry.get("published"),
            "source": entry.get("source", {}).get("title"),
            "collected_at": collected_at,
        }
        for entry in feed.entries
    ]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = RAW_DIR / f"news_{date.today().isoformat()}.jsonl"
    total = 0
    with output.open("w", encoding="utf-8") as file:
        for ticker, company in TICKERS.items():
            try:
                articles = fetch_news(ticker, company)
                for article in articles:
                    file.write(json.dumps(article, ensure_ascii=False) + "\n")
                total += len(articles)
                log.info("%s : %d articles récupérés", ticker, len(articles))
            except Exception as error:
                log.error("%s : échec de la collecte (%s)", ticker, error)
    log.info("Fichier écrit : %s (%d articles)", output, total)


if __name__ == "__main__":
    main()
