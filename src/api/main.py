"""API MarketPulse : cours, sentiment de la presse et prédiction de tendance."""
import json
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from xgboost import XGBClassifier

from src.config import TICKERS
from src.db import mongo_db, pg_engine
from src.ml.dataset import build_dataset

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/production"))
DISCLAIMER = "Outil d'aide à la décision, pas un conseil en investissement."
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : charge une seule fois le modèle de production exporté."""
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(str(MODEL_DIR / "model.json"))
    state.update(model=model, features=metadata["features"],
                 version=metadata["version"], model_name=metadata["model_name"],
                 engine=pg_engine())
    yield
    state.clear()


app = FastAPI(
    title="MarketPulse API",
    version="1.0.0",
    description="Cours de bourse, sentiment de la presse et prédiction de tendance "
                "pour 10 actions. " + DISCLAIMER,
    lifespan=lifespan,
)


class PricePoint(BaseModel):
    date: str
    adj_close: float
    daily_return_pct: float | None
    ma_7: float
    ma_30: float
    volatility_30_pct: float | None


class SentimentPoint(BaseModel):
    date: str
    sentiment_mean: float
    news_count: int


class Prediction(BaseModel):
    ticker: str
    company: str
    based_on_date: str
    proba_up: float
    direction: str
    model_version: str
    disclaimer: str


def check_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"Action inconnue : {ticker}. "
                                 f"Actions suivies : {', '.join(TICKERS)}")
    return ticker


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": state.get("model_name"),
            "model_version": state.get("version")}


@app.get("/tickers")
def tickers() -> list[dict]:
    return [{"ticker": t, "company": c} for t, c in TICKERS.items()]


@app.get("/prices/{ticker}", response_model=list[PricePoint])
def prices(ticker: str, days: int = Query(30, ge=1, le=800)):
    ticker = check_ticker(ticker)
    df = pd.read_sql(
        text("""SELECT date, adj_close, daily_return_pct, ma_7, ma_30, volatility_30_pct
                FROM price_features WHERE ticker = :ticker
                ORDER BY date DESC LIMIT :days"""),
        state["engine"], params={"ticker": ticker, "days": days},
    )
    df["date"] = df["date"].astype(str)
    df = df.astype(object).where(df.notna(), None)   # NaN -> null en JSON
    return df.iloc[::-1].to_dict("records")          # ordre chronologique


@app.get("/sentiment/{ticker}", response_model=list[SentimentPoint])
def sentiment(ticker: str, days: int = Query(30, ge=1, le=365)):
    ticker = check_ticker(ticker)
    since = (date.today() - timedelta(days=days)).isoformat()
    rows = mongo_db()["articles_clean"].aggregate([
        {"$match": {"ticker": ticker, "sentiment_score": {"$ne": None},
                    "published_date": {"$gte": since}}},
        {"$group": {"_id": "$published_date",
                    "sentiment_mean": {"$avg": "$sentiment_score"},
                    "news_count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ])
    return [{"date": r["_id"], "sentiment_mean": round(r["sentiment_mean"], 4),
             "news_count": r["news_count"]} for r in rows]


@app.get("/predict/{ticker}", response_model=Prediction)
def predict(ticker: str):
    ticker = check_ticker(ticker)
    df = build_dataset()
    latest = df[df["ticker"] == ticker].sort_values("date").iloc[-1]
    X = latest[state["features"]].to_frame().T.astype(float)
    proba_up = float(state["model"].predict_proba(X)[0, 1])
    return Prediction(
        ticker=ticker,
        company=TICKERS[ticker],
        based_on_date=str(latest["date"].date()),
        proba_up=round(proba_up, 4),
        direction="hausse" if proba_up >= 0.5 else "baisse",
        model_version=state["version"],
        disclaimer=DISCLAIMER,
    )
