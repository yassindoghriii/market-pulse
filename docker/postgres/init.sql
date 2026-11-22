-- Table des cours de bourse journaliers
CREATE TABLE IF NOT EXISTS prices (
    date       DATE          NOT NULL,
    ticker     VARCHAR(10)   NOT NULL,
    company    VARCHAR(50)   NOT NULL,
    open       NUMERIC(12,4),
    high       NUMERIC(12,4),
    low        NUMERIC(12,4),
    close      NUMERIC(12,4),
    adj_close  NUMERIC(12,4),
    volume     BIGINT,
    loaded_at  TIMESTAMPTZ   DEFAULT now(),
    PRIMARY KEY (ticker, date)
);
