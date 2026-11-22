"""Calcule (ou rafraîchit) les indicateurs de cours dans PostgreSQL."""
import logging
from pathlib import Path

import psycopg

from src.db import pg_conninfo

SQL_FILE = Path(__file__).with_name("price_features.sql")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    with psycopg.connect(pg_conninfo()) as conn:
        conn.execute(SQL_FILE.read_text(encoding="utf-8"))
        total = conn.execute("SELECT count(*) FROM price_features").fetchone()[0]
    log.info("price_features : %d lignes calculées", total)


if __name__ == "__main__":
    main()
