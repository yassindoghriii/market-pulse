"""Chargement des fichiers bruts : cours -> PostgreSQL, articles -> MongoDB."""
import json
import logging
import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()  # lit les identifiants dans le fichier .env

PRICES_DIR = Path("data/raw/prices")
NEWS_DIR = Path("data/raw/news")
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UPSERT_PRICES = """
INSERT INTO prices (date, ticker, company, open, high, low, close, adj_close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (ticker, date) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
    volume = EXCLUDED.volume, loaded_at = now()
"""


def latest_file(folder: Path, pattern: str) -> Path:
    """Renvoie le fichier le plus récent d'un dossier (les noms contiennent la date)."""
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier {pattern} dans {folder}")
    return files[-1]


def load_prices() -> None:
    path = latest_file(PRICES_DIR, "prices_*.csv")
    df = pd.read_csv(path)
    df[PRICE_COLUMNS] = df[PRICE_COLUMNS].round(4)
    columns = ["date", "ticker", "company", *PRICE_COLUMNS, "volume"]
    rows = df[columns].to_dict("split")["data"]  # convertit en types Python simples

    conninfo = (
        f"host=localhost port={os.environ['POSTGRES_PORT']} dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} password={os.environ['POSTGRES_PASSWORD']}"
    )
    with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_PRICES, rows)
        cur.execute("SELECT count(*) FROM prices")
        total = cur.fetchone()[0]
    log.info("PostgreSQL : %d lignes traitées depuis %s, %d lignes dans la table",
             len(rows), path.name, total)


def load_news() -> None:
    path = latest_file(NEWS_DIR, "news_*.jsonl")
    with path.open(encoding="utf-8") as file:
        articles = [json.loads(line) for line in file if line.strip()]

    client = MongoClient(
        host="localhost", port=int(os.environ['MONGO_PORT']),
        username=os.environ["MONGO_USER"], password=os.environ["MONGO_PASSWORD"],
        authSource="admin",
    )
    collection = client["marketpulse"]["articles_raw"]
    # un même article peut concerner deux entreprises : unicité sur (lien, action)
    collection.create_index([("link", 1), ("ticker", 1)], unique=True)

    operations = [
        UpdateOne({"link": a["link"], "ticker": a["ticker"]}, {"$set": a}, upsert=True)
        for a in articles
    ]
    result = collection.bulk_write(operations, ordered=False)
    log.info("MongoDB : %d nouveaux articles, %d déjà présents, %d au total",
             result.upserted_count, result.matched_count,
             collection.count_documents({}))
    client.close()


def main() -> None:
    load_prices()
    load_news()


if __name__ == "__main__":
    main()
