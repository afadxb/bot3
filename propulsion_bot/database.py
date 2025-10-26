"""SQLite backed persistence helpers."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import Settings
from .dtos import BarData, Signal

logger = logging.getLogger("propulsion_bot.database")


class Database:
    def __init__(self, db_path: str = "propulsion_bot.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    run_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    features_json TEXT,
                    rules_passed_json TEXT,
                    base_score REAL,
                    ai_adj_score REAL,
                    final_score REAL,
                    rank INTEGER,
                    reasons_text TEXT,
                    cycle_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    run_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    meta_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bars_cache (
                    symbol TEXT NOT NULL,
                    tf TEXT NOT NULL,
                    ts DATETIME NOT NULL,
                    o REAL,
                    h REAL,
                    l REAL,
                    c REAL,
                    v REAL,
                    PRIMARY KEY(symbol, tf, ts)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session TEXT,
                    type TEXT,
                    symbol TEXT,
                    value REAL,
                    meta_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session TEXT,
                    starting_equity REAL,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_ref TEXT,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    price REAL,
                    exec_id TEXT,
                    exchange TEXT,
                    commission REAL,
                    realized_pnl REAL,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions_live (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    avg_price REAL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.executescript(
                """
                CREATE VIEW IF NOT EXISTS v_latest_signals AS
                SELECT s1.* FROM signals s1
                INNER JOIN (
                    SELECT symbol, MAX(run_ts) as latest_ts
                    FROM signals
                    GROUP BY symbol
                ) s2 ON s1.symbol = s2.symbol AND s1.run_ts = s2.latest_ts;

                CREATE VIEW IF NOT EXISTS v_risk_events_today AS
                SELECT * FROM risk_events
                WHERE DATE(ts) = DATE('now');

                CREATE VIEW IF NOT EXISTS v_intraday_exposure AS
                SELECT
                    session,
                    SUM(realized_pnl + unrealized_pnl) as total_exposure,
                    (SELECT starting_equity FROM metrics_equity ORDER BY ts DESC LIMIT 1) as current_equity
                FROM metrics_equity
                WHERE DATE(ts) = DATE('now')
                GROUP BY session;

                CREATE VIEW IF NOT EXISTS v_daily_equity AS
                SELECT
                    ts,
                    session,
                    starting_equity,
                    realized_pnl,
                    unrealized_pnl,
                    (realized_pnl + unrealized_pnl) / starting_equity * 100 as drawdown_pct,
                    CASE WHEN ABS((realized_pnl + unrealized_pnl) / starting_equity * 100) >= 10 THEN 1 ELSE 0 END as halt_flag
                FROM metrics_equity
                WHERE DATE(ts) = DATE('now');
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_position(self, symbol: str, qty: float, avg_price: float) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO positions_live(symbol, qty, avg_price, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty=excluded.qty,
                    avg_price=excluded.avg_price,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (symbol, float(qty), float(avg_price)),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_trade_plan(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry: float,
        stop: float,
        target: float,
        order_ref: str,
        reasons: Optional[List[str]],
    ) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO trades_journal(order_ref, symbol, side, qty, price, exec_id, exchange, commission, realized_pnl, ts, reason)
                VALUES(?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, CURRENT_TIMESTAMP, ?)
                """,
                (order_ref, symbol, side, float(qty), "\n".join(reasons or [])),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_execution(
        self,
        order_ref: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        exec_id: str,
        exchange: str,
        commission: Optional[float] = None,
        realized_pnl: Optional[float] = None,
    ) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO trades_journal(order_ref, symbol, side, qty, price, exec_id, exchange, commission, realized_pnl, ts)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    order_ref,
                    symbol,
                    side,
                    float(qty),
                    float(price),
                    exec_id,
                    exchange,
                    commission,
                    realized_pnl,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_watchlist(self) -> List[Dict[str, Any]]:
        settings = Settings()
        db_path = settings.premarket_db
        if not os.path.exists(db_path):
            logger.warning("Premarket DB not found at %s", db_path)
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            query_today = (
                "SELECT * FROM v_watchlist "
                "WHERE DATE(generated_at) = DATE('now','localtime')"
            )
            try:
                rows = conn.execute(query_today).fetchall()
                if not rows:
                    query_prev = (
                        "SELECT * FROM v_watchlist "
                        "WHERE DATE(generated_at) >= DATE('now','localtime','-2 day') "
                        "AND DATE(generated_at) < DATE('now','localtime')"
                    )
                    rows = conn.execute(query_prev).fetchall()
                    if not rows:
                        rows = conn.execute("SELECT * FROM v_watchlist").fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute("SELECT * FROM v_watchlist").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def insert_signal(self, signal: Signal) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO signals
                (symbol, run_ts, features_json, rules_passed_json, base_score,
                 ai_adj_score, final_score, rank, reasons_text, cycle_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.symbol,
                    datetime.now(),
                    json.dumps(signal.features),
                    json.dumps(signal.rules_passed),
                    signal.base_score,
                    signal.ai_adj_score,
                    signal.final_score,
                    signal.rank,
                    "|".join(signal.reasons),
                    signal.cycle_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_bars_cache(self, symbol: str, tf: str, bars: List[BarData]) -> None:
        if not bars:
            return
        conn = self.get_connection()
        try:
            for bar in bars:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bars_cache (symbol, tf, ts, o, h, l, c, v)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        tf,
                        bar.timestamp.isoformat(),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_bars_from_cache(self, symbol: str, tf: str, lookback_min: int) -> List[BarData]:
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT symbol, tf, ts, o, h, l, c, v
                FROM bars_cache
                WHERE symbol = ? AND tf = ? AND ts >= datetime('now', ?)
                ORDER BY ts ASC
                """,
                (symbol, tf, f"-{int(lookback_min)} minutes"),
            )
            rows: List[BarData] = []
            for row in cursor.fetchall():
                rows.append(
                    BarData(
                        symbol=row["symbol"],
                        timestamp=datetime.fromisoformat(row["ts"]),
                        open=row["o"],
                        high=row["h"],
                        low=row["l"],
                        close=row["c"],
                        volume=row["v"],
                        timeframe=row["tf"],
                    )
                )
            return rows
        finally:
            conn.close()

    def insert_risk_event(
        self,
        event_type: str,
        session: str,
        symbol: Optional[str] = None,
        value: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO risk_events (ts, session, type, symbol, value, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(),
                    session,
                    event_type,
                    symbol,
                    value,
                    json.dumps(meta or {}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_todays_trade_count(self) -> int:
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM risk_events
                WHERE type = 'TRADE_OPENED' AND DATE(ts) = DATE('now')
                """
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_current_equity(self) -> Optional[float]:
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT starting_equity, realized_pnl, unrealized_pnl
                FROM metrics_equity
                ORDER BY ts DESC LIMIT 1
                """
            )
            row = cursor.fetchone()
            if row:
                return row[0] + row[1] + row[2]
            return None
        finally:
            conn.close()

    def get_latest_bars(self, symbol: str, timeframe: str, limit: int = 100) -> List[BarData]:
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM bars_cache
                WHERE symbol = ? AND tf = ?
                ORDER BY ts DESC LIMIT ?
                """,
                (symbol, timeframe, limit),
            )
            bars: List[BarData] = []
            for row in cursor:
                bars.append(
                    BarData(
                        symbol=row["symbol"],
                        timestamp=datetime.fromisoformat(row["ts"]),
                        open=row["o"],
                        high=row["h"],
                        low=row["l"],
                        close=row["c"],
                        volume=row["v"],
                        timeframe=row["tf"],
                    )
                )
            return sorted(bars, key=lambda bar: bar.timestamp)
        finally:
            conn.close()


__all__ = ["Database"]

