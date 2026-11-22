"""Tableau de bord MarketPulse (Streamlit) : client de l'API."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
SENTIMENT_DAYS = 60

st.set_page_config(page_title="MarketPulse", layout="wide")


@st.cache_data(ttl=300)  # garde les réponses 5 minutes pour ne pas surcharger l'API
def get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


st.title("MarketPulse")
st.caption("Cours, sentiment de la presse et tendance prédite. "
           "Outil d'aide à la décision, pas un conseil en investissement.")

try:
    tickers = get("/tickers")
except requests.RequestException:
    st.error(f"API injoignable ({API_URL}). Lancez-la avec : "
             "uvicorn src.api.main:app --port 8000")
    st.stop()

labels = {f"{t['company']} ({t['ticker']})": t["ticker"] for t in tickers}
choice = st.sidebar.selectbox("Action", list(labels))
days = st.sidebar.slider("Période affichée (séances)", 30, 750, 120)
ticker = labels[choice]

prices = pd.DataFrame(get(f"/prices/{ticker}", days=days))
prices["date"] = pd.to_datetime(prices["date"])
prediction = get(f"/predict/{ticker}")
sentiment = pd.DataFrame(get(f"/sentiment/{ticker}", days=SENTIMENT_DAYS))

# --- Indicateurs clés ---
last = prices.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dernier cours ajusté", f"{last['adj_close']:.2f}",
            f"{last['daily_return_pct']:+.2f} %")
col2.metric("Volatilité 30 séances", f"{last['volatility_30_pct']:.2f} %")
col3.metric("Tendance prédite", prediction["direction"].capitalize())
col3.caption(f"Probabilité de hausse : {prediction['proba_up']:.0%}")
if sentiment.empty:
    col4.metric(f"Sentiment de la presse ({SENTIMENT_DAYS} j)", "—")
    col4.caption("Aucun article noté sur la période")
else:
    col4.metric(f"Sentiment de la presse ({SENTIMENT_DAYS} j)",
                f"{sentiment['sentiment_mean'].mean():+.2f}")
    col4.caption(f"{int(sentiment['news_count'].sum())} articles notés")

# --- Cours et moyennes mobiles ---
st.subheader("Cours ajusté et moyennes mobiles")
series_names = {"adj_close": "Cours ajusté", "ma_7": "Moyenne 7 séances",
                "ma_30": "Moyenne 30 séances"}
long = prices.melt(id_vars="date", value_vars=list(series_names),
                   var_name="série", value_name="valeur")
long["série"] = long["série"].map(series_names)
price_chart = alt.Chart(long).mark_line().encode(
    x=alt.X("date:T", title=None),
    y=alt.Y("valeur:Q", title="Prix", scale=alt.Scale(zero=False)),  # l'axe suit les données
    color=alt.Color("série:N", title=None, legend=alt.Legend(orient="bottom")),
    tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("série:N", title="Série"),
             alt.Tooltip("valeur:Q", title="Valeur", format=".2f")],
)
st.altair_chart(price_chart)

# --- Sentiment de la presse ---
st.subheader(f"Sentiment moyen de la presse par jour, {SENTIMENT_DAYS} derniers jours")
if sentiment.empty:
    st.info("Pas encore d'articles notés pour cette action sur la période.")
else:
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment["tonalité"] = sentiment["sentiment_mean"].ge(0).map(
        {True: "positive", False: "négative"})
    sentiment_chart = alt.Chart(sentiment).mark_bar(size=14).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d/%m", tickCount="day")),
        y=alt.Y("sentiment_mean:Q", title="Sentiment (−1 à +1)",
                scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("tonalité:N", legend=None,
                        scale=alt.Scale(domain=["positive", "négative"],
                                        range=["#2e9e5b", "#d64545"])),
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("sentiment_mean:Q", title="Sentiment", format="+.2f"),
                 alt.Tooltip("news_count:Q", title="Articles")],
    )
    st.altair_chart(sentiment_chart)

st.caption(f"Prédiction calculée à partir de la séance du {prediction['based_on_date']} "
           f"· modèle version {prediction['model_version']}")
