"""Analyse de sentiment des titres d'articles avec FinBERT (traitement incrémental)."""
import logging
from datetime import datetime, timezone

from lingua import Language, LanguageDetectorBuilder
from pymongo import UpdateOne
from transformers import pipeline

from src.db import mongo_db

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # masque les requêtes HTTP

# Détecteur limité aux langues probables : plus fiable sur des textes courts
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.ITALIAN,
).build()


def detect_language(text: str) -> str:
    language = detector.detect_language_of(text)
    return language.iso_code_639_1.name.lower() if language else "unknown"


def main() -> None:
    collection = mongo_db()["articles_clean"]
    todo = list(collection.find({"sentiment_model": {"$exists": False}}, {"_id": 1, "title": 1}))
    if not todo:
        log.info("Aucun nouvel article à analyser.")
        return

    for article in todo:
        article["lang"] = detect_language(article["title"])
    english = [a for a in todo if a["lang"] == "en"]
    others = [a for a in todo if a["lang"] != "en"]
    log.info("%d articles à traiter : %d en anglais, %d dans une autre langue (ignorés)",
             len(todo), len(english), len(others))

    now = datetime.now(timezone.utc)
    operations = []

    if english:
        classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
        results = classifier([a["title"] for a in english],
                             batch_size=BATCH_SIZE, truncation=True)
        for article, scores in zip(english, results):
            p = {s["label"]: s["score"] for s in scores}
            operations.append(UpdateOne({"_id": article["_id"]}, {"$set": {
                "lang": "en",
                "sentiment_label": max(p, key=p.get),
                "p_positive": round(p["positive"], 4),
                "p_negative": round(p["negative"], 4),
                "p_neutral": round(p["neutral"], 4),
                "sentiment_score": round(p["positive"] - p["negative"], 4),
                "sentiment_model": MODEL_NAME,
                "scored_at": now,
            }}))

    for article in others:
        operations.append(UpdateOne({"_id": article["_id"]}, {"$set": {
            "lang": article["lang"],
            "sentiment_label": None,
            "sentiment_score": None,
            "sentiment_model": MODEL_NAME,
            "scored_at": now,
        }}))

    collection.bulk_write(operations, ordered=False)

    summary = collection.aggregate([
        {"$group": {"_id": "$sentiment_label", "articles": {"$sum": 1}}},
        {"$sort": {"articles": -1}},
    ])
    for row in summary:
        log.info("Sentiment %-8s : %d articles", row["_id"], row["articles"])


if __name__ == "__main__":
    main()
