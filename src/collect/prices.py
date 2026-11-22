"""Collecte des cours de bourse journaliers via yfinance."""
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# Les 10 actions suivies : code boursier (ticker) -> nom de l'entreprise
TICKERS = {
    "MC.PA": "LVMH",
    "TTE.PA": "TotalEnergies",
    "AIR.PA": "Airbus",
    "BNP.PA": "BNP Paribas",
    "SAN.PA": "Sanofi",
    "OR.PA": "L'Oréal",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
}
RAW_DIR = Path("data/raw/prices")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_prices(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Télécharge l'historique d'une action et renvoie un tableau propre."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df.empty:
        raise ValueError(f"Aucune donnée reçue pour {ticker}")

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    df["ticker"] = ticker
    df["company"] = TICKERS[ticker]
    return df[["date", "ticker", "company", "open", "high", "low",
               "close", "adj_close", "volume"]]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for ticker in TICKERS:
        try:
            df = fetch_prices(ticker)
            frames.append(df)
            log.info("%s : %d jours récupérés", ticker, len(df))
        except Exception as error:  # une action en échec ne bloque pas les autres
            log.error("%s : échec de la collecte (%s)", ticker, error)

    if not frames:
        raise SystemExit("Aucune donnée collectée : arrêt.")

    all_prices = pd.concat(frames, ignore_index=True)
    output = RAW_DIR / f"prices_{date.today().isoformat()}.csv"
    all_prices.to_csv(output, index=False)
    log.info("Fichier écrit : %s (%d lignes)", output, len(all_prices))


if __name__ == "__main__":
    main()
