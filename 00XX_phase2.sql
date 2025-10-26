-- (Unchanged from previous)

-- Create tables for Propulsion Bot Phase 2
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    run_ts TEXT NOT NULL,
    features_json TEXT,
    rules_passed_json TEXT,
    base_score REAL,
    ai_adj_score REAL,
    final_score REAL,
    rank INTEGER,
    reasons_text TEXT,
    cycle_id TEXT
);

CREATE TABLE IF NOT EXISTS ai_provenance (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    run_ts TEXT NOT NULL,
    source TEXT,
    sentiment_score REAL,
    sentiment_label TEXT,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS bars_cache (
    symbol TEXT,
    tf TEXT,
    ts TEXT,
    o REAL,
    h REAL,
    l REAL,
    c REAL,
    v INTEGER,
    PRIMARY KEY (symbol, tf, ts)
);

CREATE TABLE IF NOT EXISTS risk_events (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    session TEXT NOT NULL,
    type TEXT NOT NULL,
    symbol TEXT,
    value REAL,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS metrics_equity (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    session TEXT NOT NULL,
    starting_equity REAL,
    realized_pnl REAL,
    unrealized_pnl REAL
);

-- Views
CREATE VIEW IF NOT EXISTS v_latest_signals AS
SELECT symbol, MAX(run_ts) as run_ts, features_json, rules_passed_json, base_score, ai_adj_score, final_score, rank, reasons_text, cycle_id
FROM signals
GROUP BY symbol;

CREATE VIEW IF NOT EXISTS v_risk_events_today AS
SELECT * FROM risk_events
WHERE ts >= date('now', 'start of day');

CREATE VIEW IF NOT EXISTS v_intraday_exposure AS
SELECT session, SUM(value) as total_exposure
FROM risk_events
WHERE type = 'exposure_cap'
GROUP BY session;

CREATE VIEW IF NOT EXISTS v_daily_equity AS
SELECT session, starting_equity, realized_pnl, unrealized_pnl,
       (realized_pnl / starting_equity * 100) as drawdown_pct
FROM metrics_equity;