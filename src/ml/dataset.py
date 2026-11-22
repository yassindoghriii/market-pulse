"""Construction du jeu de données : indicateurs de cours + sentiment quotidien."""
import pandas as pd

from src.db import mongo_db, pg_engine

FEATURES = [
    "daily_return_pct", "return_lag1_pct", "dist_ma7_pct", "dist_ma30_pct",
    "volatility_30_pct", "sentiment_mean", "news_count",
]
TARGET = "target_up"


def load_prices() -> pd.DataFrame:
    df = pd.read_sql(
        """SELECT date, ticker, adj_close, daily_return_pct, ma_7, ma_30,
                  volatility_30_pct, target_up
           FROM price_features ORDER BY ticker, date""",
        pg_engine(), parse_dates=["date"],
    )
    numeric = ["adj_close", "daily_return_pct", "ma_7", "ma_30", "volatility_30_pct"]
    df[numeric] = df[numeric].astype(float)  # NUMERIC SQL -> nombres décimaux Python
    return df


def load_daily_sentiment() -> pd.DataFrame:
    """Sentiment moyen et nombre d'articles par action et par jour de publication."""
    rows = mongo_db()["articles_clean"].aggregate([
        {"$match": {"sentiment_score": {"$ne": None}}},
        {"$group": {
            "_id": {"ticker": "$ticker", "date": "$published_date"},
            "sentiment_mean": {"$avg": "$sentiment_score"},
            "news_count": {"$sum": 1},
        }},
    ])
    df = pd.DataFrame([
        {"ticker": r["_id"]["ticker"], "date": r["_id"]["date"],
         "sentiment_mean": r["sentiment_mean"], "news_count": r["news_count"]}
        for r in rows
    ])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "sentiment_mean", "news_count"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_dataset() -> pd.DataFrame:
    df = load_prices()
    df["return_lag1_pct"] = df.groupby("ticker")["daily_return_pct"].shift(1)
    df["dist_ma7_pct"] = (df["adj_close"] / df["ma_7"] - 1) * 100
    df["dist_ma30_pct"] = (df["adj_close"] / df["ma_30"] - 1) * 100

    df = df.merge(load_daily_sentiment(), on=["ticker", "date"], how="left")
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)   # pas d'article = ton neutre
    df["news_count"] = df["news_count"].fillna(0).astype(int)
    return df
