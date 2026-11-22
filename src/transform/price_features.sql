-- Indicateurs techniques et variable cible, calculés par action et par date
CREATE MATERIALIZED VIEW IF NOT EXISTS price_features AS
WITH base AS (
    SELECT
        date, ticker, company, close, adj_close, volume,
        adj_close / LAG(adj_close) OVER w - 1 AS daily_return,   -- variation vs la veille
        LEAD(adj_close) OVER w                AS next_adj_close  -- prix du lendemain
    FROM prices
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
)
SELECT
    date, ticker, company, close, adj_close, volume,
    ROUND(daily_return * 100, 4) AS daily_return_pct,
    ROUND(AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date
          ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 4)  AS ma_7,
    ROUND(AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date
          ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 4) AS ma_30,
    ROUND(STDDEV(daily_return) OVER (PARTITION BY ticker ORDER BY date
          ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) * 100, 4) AS volatility_30_pct,
    CASE
        WHEN next_adj_close IS NULL THEN NULL          -- dernier jour : lendemain inconnu
        WHEN next_adj_close > adj_close THEN 1
        ELSE 0
    END AS target_up
FROM base;

CREATE UNIQUE INDEX IF NOT EXISTS price_features_pk ON price_features (ticker, date);
REFRESH MATERIALIZED VIEW price_features;
