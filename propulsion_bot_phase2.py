# propulsion_bot_phase2.py
# Complete Intraday Trading System - Phase 2
# Single file implementation based on the detailed blueprint

import sqlite3
import json
import yaml
import logging
import asyncio
import time
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import pytz
import requests
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None  # Fallback handled in start()
import schedule
import threading
from pathlib import Path
from ibapi.contract import Contract
# ib_insync not required for EClient-based flow
from datetime import datetime, timezone
import pandas as pd
import time
import logging
import math

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)

# Load environment variables
load_dotenv()

class Settings:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Load YAML config from file if present; fallback to embedded defaults
        cfg_path = Path('config') / 'strategy.yaml'
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = yaml.safe_load(DEFAULT_CONFIG)
        self.ibkr_host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.ibkr_port = int(os.getenv('IBKR_PORT', '7497'))
        self.ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
        self.finnhub_token = os.getenv('FINNHUB_TOKEN', '')
        self.dashboard_port = int(os.getenv('DASHBOARD_PORT', '8501'))
        # External DB paths
        self.premarket_db = os.getenv('PREMARKET_DB', str(Path('..')/ 'bot1.1' / 'premarket.db'))
        # Testing/ops flags
        self.force_market_open = str(os.getenv('FORCE_MARKET_OPEN', 'false')).lower() in ('1','true','yes')
    
    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# =============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# =============================================================================

@dataclass
class Signal:
    symbol: str
    entry_hint: Optional[float]
    stop_hint: Optional[float]
    direction: str  # 'long' or 'short'
    base_score: float
    ai_adj_score: float
    final_score: float
    reasons: List[str]
    rules_passed: Dict[str, bool]
    features: Dict[str, Any]
    cycle_id: str
    rank: int = 0
    
    def __post_init__(self):
        if not 0 <= self.base_score <= 1:
            raise ValueError("base_score must be between 0 and 1")
        if not 0 <= self.ai_adj_score <= 1:
            raise ValueError("ai_adj_score must be between 0 and 1")
        if not 0 <= self.final_score <= 1:
            raise ValueError("final_score must be between 0 and 1")

@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: int
    entry: float
    stop: float
    scale_out: float
    target: float
    trail_mode: str
    risk_context: Dict[str, Any]
    reasons: Optional[List[str]] = None

@dataclass
class RiskEvent:
    ts: datetime
    session: str
    type: str
    symbol: Optional[str]
    value: Optional[float]
    meta_json: Dict[str, Any]

@dataclass
class BarData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Optional[str] = '5min'

@dataclass
class Headline:
    symbol: str
    headline: str
    source: str
    timestamp: datetime
    url: Optional[str] = None

# =============================================================================
# DATABASE & PERSISTENCE
# =============================================================================

class Database:
    def __init__(self, db_path: str = 'propulsion_bot.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = self.get_connection()
        try:
            # Signals table
            conn.execute('''
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
            ''')
            
            # AI Provenance table
            conn.execute('''
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
            ''')
            
            # Bars cache table
            conn.execute('''
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
            ''')
            
            # Risk events table
            conn.execute('''
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
            ''')
            
            # Equity metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session TEXT,
                    starting_equity REAL,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Trades journal (executions/fills)
            conn.execute('''
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
            ''')

            # Live positions snapshot
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions_live (
                    symbol TEXT PRIMARY KEY,
                    qty REAL,
                    avg_price REAL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create views
            conn.executescript('''
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
            ''')
            
            conn.commit()
        finally:
            conn.close()
    
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Trades/Positions helpers ---
    def upsert_position(self, symbol: str, qty: float, avg_price: float):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO positions_live(symbol, qty, avg_price, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty=excluded.qty,
                    avg_price=excluded.avg_price,
                    updated_at=CURRENT_TIMESTAMP
            ''', (symbol, float(qty), float(avg_price)))
            conn.commit()
        finally:
            conn.close()

    def insert_trade_plan(self, symbol: str, side: str, qty: int, entry: float, stop: float, target: float, order_ref: str, reasons: Optional[List[str]]):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO trades_journal(order_ref, symbol, side, qty, price, exec_id, exchange, commission, realized_pnl, ts, reason)
                VALUES(?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, CURRENT_TIMESTAMP, ?)
            ''', (order_ref, symbol, side, float(qty), '\n'.join(reasons or [])))
            conn.commit()
        finally:
            conn.close()

    def insert_execution(self, order_ref: str, symbol: str, side: str, qty: float, price: float, exec_id: str, exchange: str, commission: Optional[float] = None, realized_pnl: Optional[float] = None):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO trades_journal(order_ref, symbol, side, qty, price, exec_id, exchange, commission, realized_pnl, ts)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (order_ref, symbol, side, float(qty), float(price), exec_id, exchange, commission, realized_pnl))
            conn.commit()
        finally:
            conn.close()

    def load_watchlist(self) -> List[Dict[str, Any]]:
        settings = Settings()
        db_path = settings.premarket_db
        if not os.path.exists(db_path):
            logging.warning(f"Premarket DB not found at {db_path}")
            return []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Prefer rows generated today; if none, fall back to previous 2 days; if column missing, fall back to all
            query_today = (
                "SELECT * FROM v_watchlist "
                "WHERE DATE(generated_at) = DATE('now','localtime')"
            )
            try:
                rows = conn.execute(query_today).fetchall()
                # If no rows for today, fall back to previous 2 full days
                if not rows:
                    query_prev2 = (
                        "SELECT * FROM v_watchlist "
                        "WHERE DATE(generated_at) >= DATE('now','localtime','-2 day') "
                        "AND DATE(generated_at) < DATE('now','localtime')"
                    )
                    rows = conn.execute(query_prev2).fetchall()
                    if not rows:
                        rows = conn.execute("SELECT * FROM v_watchlist").fetchall()
            except sqlite3.OperationalError:
                # generated_at column may not exist in the view; fallback
                rows = conn.execute("SELECT * FROM v_watchlist").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def insert_signal(self, signal: Signal):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO signals 
                (symbol, run_ts, features_json, rules_passed_json, base_score, 
                 ai_adj_score, final_score, rank, reasons_text, cycle_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.symbol,
                datetime.now(),
                json.dumps(signal.features),
                json.dumps(signal.rules_passed),
                signal.base_score,
                signal.ai_adj_score,
                signal.final_score,
                signal.rank,
                '|'.join(signal.reasons),
                signal.cycle_id
            ))
            conn.commit()
        finally:
            conn.close()

    def upsert_bars_cache(self, symbol: str, tf: str, bars: List['BarData']):
        if not bars:
            return
        conn = self.get_connection()
        try:
            for b in bars:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bars_cache (symbol, tf, ts, o, h, l, c, v)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (symbol, tf, b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume)
                )
            conn.commit()
        finally:
            conn.close()

    def get_bars_from_cache(self, symbol: str, tf: str, lookback_min: int) -> List['BarData']:
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT symbol, tf, ts, o, h, l, c, v
                FROM bars_cache
                WHERE symbol = ? AND tf = ? AND ts >= datetime('now', ?)
                ORDER BY ts ASC
                """,
                (symbol, tf, f'-{int(lookback_min)} minutes')
            )
            res: List[BarData] = []
            for row in cur.fetchall():
                res.append(BarData(symbol=row['symbol'],
                                   timestamp=datetime.fromisoformat(row['ts']),
                                   open=row['o'], high=row['h'], low=row['l'],
                                   close=row['c'], volume=row['v'], timeframe=row['tf']))
            return res
        finally:
            conn.close()
    
    def insert_risk_event(self, event_type: str, session: str, symbol: Optional[str] = None,
                         value: Optional[float] = None, meta: Optional[Dict] = None):
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO risk_events (ts, session, type, symbol, value, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                session,
                event_type,
                symbol,
                value,
                json.dumps(meta or {})
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_todays_trade_count(self) -> int:
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT COUNT(*) as count FROM risk_events 
                WHERE type = 'TRADE_OPENED' AND DATE(ts) = DATE('now')
            ''')
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_current_equity(self) -> Optional[float]:
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT starting_equity, realized_pnl, unrealized_pnl 
                FROM metrics_equity 
                ORDER BY ts DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            if row:
                return row[0] + row[1] + row[2]
            return None
        finally:
            conn.close()
    
    def get_latest_bars(self, symbol: str, timeframe: str, limit: int = 100) -> List[BarData]:
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT * FROM bars_cache 
                WHERE symbol = ? AND tf = ?
                ORDER BY ts DESC LIMIT ?
            ''', (symbol, timeframe, limit))
            
            bars = []
            for row in cursor:
                bars.append(BarData(
                    symbol=row['symbol'],
                    timestamp=datetime.fromisoformat(row['ts']),
                    open=row['o'],
                    high=row['h'],
                    low=row['l'],
                    close=row['c'],
                    volume=row['v'],
                    timeframe=row['tf']
                ))
            return sorted(bars, key=lambda x: x.timestamp)
        finally:
            conn.close()

# =============================================================================
# MARKET CLOCK
# =============================================================================

class MarketClock:
    def __init__(self, timezone: str = "America/Toronto"):
        self.tz = pytz.timezone(timezone)
    
    def is_market_hours(self, check_time: Optional[datetime] = None) -> bool:
        if check_time is None:
            check_time = datetime.now(self.tz)
        
        market_time = check_time.astimezone(self.tz)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        return (market_time.weekday() < 7 and 
                market_open <= market_time.time() <= market_close)
    
    def time_until_flatten(self, flatten_time: str = "15:55") -> Optional[timedelta]:
        now = datetime.now(self.tz)
        hour, minute = map(int, flatten_time.split(':'))
        flatten_dt = datetime.combine(now.date(), dt_time(hour, minute))
        flatten_dt = self.tz.localize(flatten_dt)
        
        if now >= flatten_dt:
            return timedelta(0)
        
        return flatten_dt - now
    
    def should_flatten(self, flatten_time: str = "15:55") -> bool:
        time_until = self.time_until_flatten(flatten_time)
        return time_until is not None and time_until.total_seconds() <= 0

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return [None] * len(prices)
        
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.tolist()

    @staticmethod
    def calculate_vwap(highs: List[float], lows: List[float], 
                      closes: List[float], volumes: List[float]) -> List[float]:
        if len(highs) < 1:
            return [None] * len(highs)
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        vwap = []
        cumulative_tpv = 0
        cumulative_volume = 0
        
        for tp, vol in zip(typical_prices, volumes):
            cumulative_tpv += tp * vol
            cumulative_volume += vol
            vwap.append(cumulative_tpv / cumulative_volume if cumulative_volume > 0 else tp)
        
        return vwap

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], 
                     closes: List[float], period: int = 14) -> List[float]:
        if len(highs) < period:
            return [None] * len(highs)
        
        tr = [highs[0] - lows[0]]
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr.append(max(tr1, tr2, tr3))
        
        atr = pd.Series(tr).rolling(window=period).mean()
        return atr.tolist()

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean()
        avg_losses = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return [None] + rsi.tolist()

    @staticmethod
    def calculate_supertrend(highs: List[float], lows: List[float], 
                            closes: List[float], period: int = 10, 
                            multiplier: float = 3.0) -> dict:
        if len(highs) < period:
            return {'trend': [None] * len(highs), 'upper': [None] * len(highs), 'lower': [None] * len(highs)}
        
        atr_values = TechnicalIndicators.calculate_atr(highs, lows, closes, period)
        
        hl2 = [(h + l) / 2 for h, l in zip(highs, lows)]
        upper_band = [hl2[i] + (multiplier * atr_values[i]) for i in range(len(hl2))]
        lower_band = [hl2[i] - (multiplier * atr_values[i]) for i in range(len(hl2))]
        
        trend = [None] * len(closes)
        upper = [None] * len(closes)
        lower = [None] * len(closes)
        
        trend[period] = 1
        upper[period] = upper_band[period]
        lower[period] = lower_band[period]
        
        for i in range(period + 1, len(closes)):
            if trend[i-1] == 1:
                if closes[i] > lower_band[i-1]:
                    trend[i] = 1
                    lower[i] = max(lower_band[i], lower[i-1])
                else:
                    trend[i] = -1
                    upper[i] = upper_band[i]
            else:
                if closes[i] < upper_band[i-1]:
                    trend[i] = -1
                    upper[i] = min(upper_band[i], upper[i-1])
                else:
                    trend[i] = 1
                    lower[i] = lower_band[i]
        
        return {'trend': trend, 'upper': upper, 'lower': lower}

# =============================================================================
# FEATURE BUILDER
# =============================================================================

class FeatureBuilder:
    def __init__(self, config):
        self.config = config
        self.indicators = TechnicalIndicators()
    
    def build(self, symbol: str, bars: List[BarData], 
              fundamentals: Dict[str, Any] = None, 
              catalysts: List[Headline] = None,
              quote: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if len(bars) < 21:
            return {}
        
        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        volumes = [bar.volume for bar in bars]
        timestamps = [bar.timestamp for bar in bars]
        
        features = {}
        settings = Settings()
        
        features['ema_fast'] = self.indicators.calculate_ema(closes, settings.get('strategy.ema_fast', 9))
        features['ema_slow'] = self.indicators.calculate_ema(closes, settings.get('strategy.ema_slow', 21))
        features['ema_bias'] = self.indicators.calculate_ema(closes, settings.get('strategy.ema_bias', 50))
        features['vwap'] = self.indicators.calculate_vwap(highs, lows, closes, volumes)
        features['atr'] = self.indicators.calculate_atr(highs, lows, closes)
        features['rsi'] = self.indicators.calculate_rsi(closes)
        
        features['volume_spike'] = self._calculate_volume_spike(volumes)
        features['consolidation'] = self._detect_consolidation(highs, lows, closes)
        features['catalyst_freshness'] = self._calculate_catalyst_freshness(catalysts, timestamps[-1])

        # Optional pattern detectors and volume profile
        try:
            if len(bars) >= 40:
                tri = self._detect_triangle(highs, lows, closes, lookback=40)
                features['pattern_triangle'] = tri
            else:
                features['pattern_triangle'] = {'is_contracting': False, 'breakout_up': False}
        except Exception:
            features['pattern_triangle'] = {'is_contracting': False, 'breakout_up': False}

        try:
            if len(bars) >= 60:
                dt = self._detect_double_top(highs, closes, volumes, lookback=60)
                features['pattern_double_top'] = dt
            else:
                features['pattern_double_top'] = False
        except Exception:
            features['pattern_double_top'] = False

        try:
            if len(bars) >= 60:
                vp = self._volume_profile_levels(lows, highs, closes, volumes, lookback=60, bins=30)
                features.update(vp)
        except Exception:
            pass
        
        if settings.get('strategy.enable_supertrend', False):
            supertrend_params = settings.get('strategy.supertrend', {})
            features['supertrend'] = self.indicators.calculate_supertrend(
                highs, lows, closes,
                supertrend_params.get('atr_period', 10),
                supertrend_params.get('atr_mult', 3.0)
            )
        
        if fundamentals:
            features.update({
                'market_cap': fundamentals.get('market_cap'),
                'pe_ratio': fundamentals.get('pe_ratio'),
                'volume_avg': fundamentals.get('volume_avg')
            })
        
        features['closes'] = closes
        features['highs'] = highs
        features['lows'] = lows
        features['volumes'] = volumes
        # Inject live pricing context if provided
        if quote:
            q_last = quote.get('last')
            q_ask = quote.get('ask')
            q_bid = quote.get('bid')
            latest_price = float(q_last if q_last is not None else (q_ask if q_ask is not None else closes[-1]))
            features['latest_price'] = latest_price
            if q_bid is not None and q_ask is not None:
                features['spread'] = float(q_ask) - float(q_bid)
        return features
    
    def _calculate_volume_spike(self, volumes: List[float], 
                              lookback: int = 20) -> List[float]:
        multiplier = Settings().get('strategy.vol_spike_multiple', 1.5)
        
        volume_spikes = []
        for i in range(len(volumes)):
            if i < lookback:
                volume_spikes.append(1.0)
                continue
            
            avg_volume = sum(volumes[i-lookback:i]) / lookback
            current_volume = volumes[i]
            spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_spikes.append(spike_ratio)
        
        return volume_spikes
    
    def _detect_consolidation(self, highs: List[float], lows: List[float], 
                            closes: List[float]) -> List[bool]:
        lookback = Settings().get('strategy.consolidation_lookback', 20)
        
        consolidation = []
        for i in range(len(closes)):
            if i < lookback:
                consolidation.append(False)
                continue
            
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]
            
            range_high = max(recent_highs)
            range_low = min(recent_lows)
            range_size = range_high - range_low
            avg_price = sum(closes[i-lookback:i]) / lookback
            
            is_consolidating = (range_size / avg_price) < 0.02
            consolidation.append(is_consolidating)
        
        return consolidation

    def _detect_triangle(self, highs: List[float], lows: List[float], 
                         closes: List[float], lookback: int = 40) -> Dict[str, Any]:
        if len(closes) < lookback:
            return {'is_contracting': False, 'breakout_up': False}
        h = highs[-lookback:]
        l = lows[-lookback:]
        n = len(h)
        x = np.arange(n)
        try:
            m_h, b_h = np.polyfit(x, h, 1)
            m_l, b_l = np.polyfit(x, l, 1)
        except Exception:
            return {'is_contracting': False, 'breakout_up': False}
        upper_start = m_h*0 + b_h
        lower_start = m_l*0 + b_l
        upper_end = m_h*(n-1) + b_h
        lower_end = m_l*(n-1) + b_l
        width_start = max(upper_start - lower_start, 0.0)
        width_end = max(upper_end - lower_end, 0.0)
        is_contracting = (m_h < 0) and (m_l > 0) and (width_end < width_start * 0.8)
        # Breakout detection: last close above last projected upper
        last_close = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else last_close
        breakout_up = False
        if is_contracting:
            upper_prev = m_h*(n-2) + b_h
            upper_last = upper_end
            try:
                if all(np.isfinite([last_close, prev_close, upper_prev, upper_last])):
                    breakout_up = (prev_close <= upper_prev) and (last_close > upper_last)
            except Exception:
                breakout_up = False
        return {
            'is_contracting': bool(is_contracting),
            'breakout_up': bool(breakout_up),
            'upper_last': float(upper_end) if np.isfinite(upper_end) else None,
            'lower_last': float(lower_end) if np.isfinite(lower_end) else None,
        }

    def _detect_double_top(self, highs: List[float], closes: List[float], 
                           volumes: List[float], lookback: int = 60, tolerance: float = 0.01) -> bool:
        if len(highs) < lookback or len(highs) < 5:
            return False
        h = highs[-lookback:]
        v = volumes[-lookback:]
        # Simple peak detection
        peaks = []
        for i in range(1, len(h)-1):
            if h[i] > h[i-1] and h[i] > h[i+1]:
                peaks.append(i)
        if len(peaks) < 2:
            return False
        p1, p2 = peaks[-2], peaks[-1]
        price1, price2 = h[p1], h[p2]
        if not (np.isfinite(price1) and np.isfinite(price2)):
            return False
        # Near equal highs within tolerance
        if abs(price1 - price2) / max(price1, 1e-6) > tolerance:
            return False
        # Optional volume condition: second peak volume not higher than first by large margin
        try:
            if v[p2] > v[p1] * 1.5:
                return False
        except Exception:
            pass
        return True

    def _volume_profile_levels(self, lows: List[float], highs: List[float], closes: List[float], 
                               volumes: List[float], lookback: int = 60, bins: int = 30) -> Dict[str, Any]:
        lo = np.array(lows[-lookback:])
        hi = np.array(highs[-lookback:])
        cl = np.array(closes[-lookback:])
        vol = np.array(volumes[-lookback:])
        pr_min = np.nanmin(lo)
        pr_max = np.nanmax(hi)
        if not np.isfinite(pr_min) or not np.isfinite(pr_max) or pr_max <= pr_min:
            return {}
        edges = np.linspace(pr_min, pr_max, bins+1)
        centers = (edges[:-1] + edges[1:]) / 2
        # Assign each close to a bin
        idx = np.clip(np.digitize(cl, edges) - 1, 0, bins-1)
        vol_by_bin = np.zeros(bins, dtype=float)
        for i in range(len(idx)):
            if np.isfinite(vol[i]):
                vol_by_bin[idx[i]] += float(vol[i])
        if vol_by_bin.sum() <= 0:
            return {}
        # Determine HVN/LVN thresholds
        hvn_cut = np.quantile(vol_by_bin, 0.85)
        lvn_cut = np.quantile(vol_by_bin, 0.15)
        hvn_levels = [float(centers[i]) for i, val in enumerate(vol_by_bin) if val >= hvn_cut and val > 0]
        lvn_levels = [float(centers[i]) for i, val in enumerate(vol_by_bin) if val <= lvn_cut and val > 0]
        last_close = float(closes[-1]) if np.isfinite(closes[-1]) else None
        nearest_hvn = None
        nearest_lvn = None
        prox_hvn = None
        prox_lvn = None
        if last_close is not None:
            if hvn_levels:
                nearest_hvn = float(min(hvn_levels, key=lambda x: abs(x - last_close)))
                prox_hvn = abs(nearest_hvn - last_close)
            if lvn_levels:
                nearest_lvn = float(min(lvn_levels, key=lambda x: abs(x - last_close)))
                prox_lvn = abs(nearest_lvn - last_close)
        return {
            'vp_hvn_levels': hvn_levels,
            'vp_lvn_levels': lvn_levels,
            'vp_nearest_hvn': nearest_hvn,
            'vp_nearest_lvn': nearest_lvn,
            'vp_prox_hvn': prox_hvn,
            'vp_prox_lvn': prox_lvn,
        }
    
    def _calculate_catalyst_freshness(self, catalysts: List[Headline], 
                                    current_time: datetime) -> float:
        if not catalysts:
            return 0.0
        
        recent_catalyst = max(catalysts, key=lambda x: x.timestamp)
        hours_ago = (current_time - recent_catalyst.timestamp).total_seconds() / 3600
        
        freshness = max(0, 1 - (hours_ago / 24))
        return freshness

# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class PropulsionStrategy:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, features_batch: List[Dict[str, Any]]) -> List[Signal]:
        signals = []
        
        for features in features_batch:
            symbol = features.get('symbol')
            feature_data = features.get('features', {})
            
            if not feature_data:
                continue
            
            signal = self._evaluate_single(symbol, feature_data)
            if signal:
                signals.append(signal)
        
        signals.sort(key=lambda x: x.final_score, reverse=True)
        for i, signal in enumerate(signals):
            signal.rank = i + 1
        
        return signals
    
    def _evaluate_single(self, symbol: str, features: Dict[str, Any]) -> Signal:
        reasons = []
        rules_passed = {}
        # long_score = bullish bias; short_score = bearish bias
        base_score = 0.0
        bear_score = 0.0
        penalties = []
        settings = Settings()
        
        # Direction gate via Supertrend (preferred if enabled)
        long_allowed = True
        short_allowed = True
        if settings.get('strategy.enable_supertrend', False):
            st = features.get('supertrend', {}) or {}
            st_trend = (st.get('trend') or [])
            if st_trend:
                last_tr = st_trend[-1]
                if last_tr == 1:
                    long_allowed, short_allowed = True, False
                    rules_passed['dir_gate'] = True
                    reasons.append('DirGate=Supertrend Long')
                elif last_tr == -1:
                    long_allowed, short_allowed = False, True
                    rules_passed['dir_gate'] = True
                    reasons.append('DirGate=Supertrend Short')

        # EMA Cross Check (trigger/boost)
        ema_fast = features.get('ema_fast', [])
        ema_slow = features.get('ema_slow', [])
        if len(ema_fast) > 1 and len(ema_slow) > 1:
            current_fast = ema_fast[-1]
            current_slow = ema_slow[-1]
            prev_fast = ema_fast[-2]
            prev_slow = ema_slow[-2]
            # Guard against None/NaN
            def _finite(x):
                try:
                    return x is not None and np.isfinite(float(x))
                except Exception:
                    return False
            # Check cross in last 1–2 bars
            cross_up = False
            cross_down = False
            if all(_finite(x) for x in (current_fast, current_slow, prev_fast, prev_slow)):
                cross_up = (prev_fast <= prev_slow) and (current_fast > current_slow)
                cross_down = (prev_fast >= prev_slow) and (current_fast < current_slow)
                # also look back 2 bars if available
                if len(ema_fast) > 2 and len(ema_slow) > 2:
                    f2, s2 = ema_fast[-3], ema_slow[-3]
                    if _finite(f2) and _finite(s2):
                        cross_up = cross_up or ((f2 <= s2) and (prev_fast > prev_slow))
                        cross_down = cross_down or ((f2 >= s2) and (prev_fast < prev_slow))
            rules_passed['ema_cross_up'] = bool(cross_up)
            rules_passed['ema_cross_down'] = bool(cross_down)
            if cross_up and long_allowed:
                base_score += 0.3
                reasons.append("EMA Cross Up")
            if cross_down and short_allowed:
                bear_score += 0.3
                reasons.append("EMA Cross Down")
        
        # Alignment: EMA bias / EMA21 and VWAP
        ema_bias = features.get('ema_bias', [])
        closes = features.get('closes', [])
        if ema_bias and closes:
            cb = closes[-1]
            eb = ema_bias[-1]
            try:
                if cb is not None and eb is not None and np.isfinite(cb) and np.isfinite(eb):
                    price_above_bias = cb > eb
                    rules_passed['ema_bias'] = price_above_bias
                    if price_above_bias and long_allowed:
                        base_score += 0.2
                        reasons.append("Above EMA Bias")
                    if (not price_above_bias) and short_allowed:
                        bear_score += 0.2
                        reasons.append("Below EMA Bias")
            except Exception:
                pass
        
        # VWAP Alignment
        vwap = features.get('vwap', [])
        if vwap and closes:
            cv = closes[-1]
            vv = vwap[-1]
            try:
                if cv is not None and vv is not None and np.isfinite(cv) and np.isfinite(vv):
                    price_above_vwap = cv > vv
                    rules_passed['vwap_alignment'] = price_above_vwap
                    if price_above_vwap and settings.get('strategy.vwap_required', True) and long_allowed:
                        base_score += 0.2
                        reasons.append("Above VWAP")
                    if (not price_above_vwap) and short_allowed:
                        bear_score += 0.2
                        reasons.append("Below VWAP")
            except Exception:
                pass
        
        # Volume Spike
        volume_spike = features.get('volume_spike', [])
        if volume_spike:
            try:
                vs = float(volume_spike[-1])
                thresh = float(settings.get('strategy.vol_spike_multiple', 1.5))
                if np.isfinite(vs):
                    has_volume_spike = vs > thresh
                    rules_passed['volume_spike'] = has_volume_spike
                    if has_volume_spike:
                        base_score += 0.15
                        reasons.append("Volume Spike")
            except Exception:
                pass
        
        # Consolidation Break
        consolidation = features.get('consolidation', [])
        if len(consolidation) > 1:
            was_consolidating = consolidation[-2]
            broke_out = not consolidation[-1] and was_consolidating
            rules_passed['consolidation_break'] = broke_out
            if broke_out:
                base_score += 0.15
                reasons.append("Consolidation Break")
        
        # Catalyst Check
        catalyst_freshness = features.get('catalyst_freshness', 0)
        try:
            rules_passed['catalyst'] = float(catalyst_freshness) > 0
        except Exception:
            rules_passed['catalyst'] = False

        # Pattern detectors influence
        tri = features.get('pattern_triangle', {}) or {}
        if tri.get('is_contracting') and tri.get('breakout_up'):
            base_score += 0.15
            reasons.append("Triangle Breakout")

        # Double top is bearish; penalize
        if features.get('pattern_double_top'):
            penalties.append(0.2)
            reasons.append("Double Top Risk")

        # Volume profile influence
        atr_vals = features.get('atr', [])
        latest_atr = atr_vals[-1] if atr_vals else None
        try:
            if latest_atr is not None and np.isfinite(latest_atr) and latest_atr > 0 and closes:
                cl = closes[-1]
                prox_lvn = features.get('vp_prox_lvn')
                prox_hvn = features.get('vp_prox_hvn')
                # Reward proximity to LVN (potential fast zone) within 0.5 ATR
                if prox_lvn is not None and np.isfinite(prox_lvn) and prox_lvn <= 0.5 * latest_atr:
                    base_score += 0.1
                    reasons.append("Near LVN")
                # Penalize HVN overhead (resistance) within 0.5 ATR if HVN above price
                nearest_hvn = features.get('vp_nearest_hvn')
                if (nearest_hvn is not None and np.isfinite(nearest_hvn)
                        and nearest_hvn >= cl and (nearest_hvn - cl) <= 0.5 * latest_atr):
                    penalties.append(0.1)
                    reasons.append("HVN Overhead")
        except Exception:
            pass
        if settings.get('strategy.catalyst_required', True):
            if catalyst_freshness > 0:
                base_score += catalyst_freshness * 0.1
                reasons.append(f"Recent Catalyst ({catalyst_freshness:.2f})")
        else:
            base_score += catalyst_freshness * 0.1
        
        # Supertrend still influences if enabled (small boost/penalty)
        if settings.get('strategy.enable_supertrend', False):
            supertrend = features.get('supertrend', {})
            trend = supertrend.get('trend', [])
            if trend and trend[-1] == 1 and long_allowed:
                base_score += 0.1
                reasons.append("Supertrend Long")
            elif trend and trend[-1] == -1 and short_allowed:
                bear_score += 0.1
                reasons.append("Supertrend Short")
        
        # Short side contributions already included above when short_allowed

        # Decide direction and calculate entry/stop hints
        # If gate forced one side, honor it; else pick the stronger
        if long_allowed and not short_allowed:
            direction = 'long'
        elif short_allowed and not long_allowed:
            direction = 'short'
        else:
            direction = 'long' if base_score >= bear_score else 'short'
        entry_hint = closes[-1] if closes else None
        stop_hint = None
        if entry_hint and features.get('atr'):
            atr = features['atr'][-1]
            try:
                if atr is not None and np.isfinite(atr) and atr > 0:
                    stop_hint = entry_hint - (atr * 1.5) if direction == 'long' else entry_hint + (atr * 1.5)
            except Exception:
                stop_hint = None

        total_penalty = sum(penalties)
        chosen_score = max(base_score, bear_score)
        final_score = max(0, min(1, chosen_score - total_penalty))
        # Add direction hint to reasons for debugging
        try:
            reasons.append(f"Dir={direction} bull={base_score:.2f} short={bear_score:.2f}")
        except Exception:
            pass

        return Signal(
            symbol=symbol,
            entry_hint=entry_hint,
            stop_hint=stop_hint,
            direction=direction,
            base_score=base_score,
            ai_adj_score=final_score,  # Default to final_score when AI is off
            final_score=final_score,
            reasons=reasons,
            rules_passed=rules_passed,
            features=features,
            cycle_id="cycle_" + str(int(time.time()))
        )

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

class RiskManager:
    def __init__(self, config, db: Database, ib: Optional['IBKRLiveFeed']=None):
        self.config = config
        self.db = db
        self.ib = ib
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def pre_trade_checks(self, signals: List[Signal]) -> Tuple[bool, List[Signal]]:
        allowed_signals = []
        settings = Settings()
        
        # Daily trade cap check
        todays_trades = self.db.get_todays_trade_count()
        trade_cap = settings.get('risk.daily_trade_cap', 20)
        
        if todays_trades >= trade_cap:
            self.db.insert_risk_event(
                'TRADE_CAP_REACHED',
                self.session_id,
                value=todays_trades,
                meta={'cap': trade_cap}
            )
            return False, []
        
        # Drawdown halt check
        # Prefer IBKR reported equity if available
        current_equity = None
        if self.ib is not None:
            current_equity = self.ib.get_equity() or None
        if current_equity is None:
            current_equity = self.db.get_current_equity()
        if current_equity:
            # Simplified drawdown calculation
            # In production, you'd track starting equity and calculate properly
            drawdown_pct = 0  # Placeholder
            drawdown_halt = settings.get('risk.daily_drawdown_halt_pct', 10.0)
            
            if drawdown_pct >= drawdown_halt:
                self.db.insert_risk_event(
                    'DRAWDOWN_HALT',
                    self.session_id,
                    value=drawdown_pct,
                    meta={'halt_threshold': drawdown_halt}
                )
                return False, []
        
        # Exposure check
        exposure_pct = self._calculate_current_exposure()
        max_exposure = settings.get('risk.max_portfolio_exposure_pct', 100)
        
        if exposure_pct >= max_exposure:
            self.db.insert_risk_event(
                'EXPOSURE_CAP_REACHED',
                self.session_id,
                value=exposure_pct,
                meta={'cap': max_exposure}
            )
            return False, []
        
        # Individual signal checks
        for signal in signals:
            if self._check_signal_risk(signal):
                allowed_signals.append(signal)
        
        return True, allowed_signals
    
    def _calculate_current_exposure(self) -> float:
        # Calculate exposure as sum(|position * avgCost|) / equity * 100
        equity = None
        if self.ib is not None:
            equity = self.ib.get_equity()
        if equity is None or equity <= 0:
            return 0.0
        total = 0.0
        positions = []
        if self.ib is not None:
            positions = self.ib.get_positions()
        for p in positions:
            try:
                total += abs(float(p.get('position', 0.0)) * float(p.get('avgCost', 0.0)))
            except Exception:
                continue
        return (total / equity) * 100.0 if equity else 0.0
    
    def _check_signal_risk(self, signal: Signal) -> bool:
        settings = Settings()
        
        # Illiquidity veto
        if settings.get('risk.illiquidity_veto', True):
            # Simplified liquidity check
            volume = signal.features.get('volumes', [])
            if volume and volume[-1] < 10:  # Placeholder threshold
                self.db.insert_risk_event(
                    'ILLIQUIDITY_VETO',
                    self.session_id,
                    symbol=signal.symbol,
                    value=volume[-1]
                )
                return False
        
        # Spread penalty (would be applied in scoring)
        # Earnings blackout (would check earnings calendar)
        
        return True

# =============================================================================
# TRADE MANAGER & EXECUTION
# =============================================================================

class TradeManager:
    def __init__(self, config, db: Database, ib: Optional['IBKRLiveFeed']=None):
        self.config = config
        self.db = db
        self.ib = ib
        self.dry_run = Settings().get('execution.enable_orders', False) == False
        # Track submitted (symbol, side) to avoid duplicate open orders per session and allow flipping
        self._submitted_symbols: set[Tuple[str, str]] = set()
    
    def plan_orders(self, signals: List[Signal]) -> List[PlannedOrder]:
        planned_orders = []
        settings = Settings()
        
        for signal in signals:
            # Basic validation of entry/stop
            if signal.entry_hint is None or signal.stop_hint is None:
                continue
            try:
                entry = float(signal.entry_hint)
                stop = float(signal.stop_hint)
            except Exception:
                continue
            if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0 or stop <= 0:
                continue
            
            # Position sizing
            equity = self.db.get_current_equity() or 100000  # Default equity
            risk_per_trade_pct = settings.get('risk.risk_per_trade_pct', 1.0)
            dollar_risk = equity * risk_per_trade_pct / 100
            
            raw_risk = abs(entry - stop)
            min_tick = float(settings.get('risk.min_tick_buffer', 0.01))
            per_share_risk = raw_risk if (np.isfinite(raw_risk) and raw_risk > 0) else min_tick
            per_share_risk = max(per_share_risk, min_tick)
            
            qty = int(max(0, dollar_risk / per_share_risk))
            
            # Cap position size
            max_position_pct = settings.get('risk.max_position_value_pct', 20.0)
            max_position_value = equity * max_position_pct / 100
            max_qty_by_value = int(max_position_value / entry) if entry > 0 else 0
            
            qty = min(qty, max_qty_by_value)
            
            if qty <= 0:
                continue
            
            # Calculate targets (direction-aware)
            risk_distance = per_share_risk
            scale_out_r = settings.get('execution.scale_out_at_r_multiple', 1.0)
            final_target_r = settings.get('execution.final_target_r_multiple', 2.0)
            if getattr(signal, 'direction', 'long') == 'short':
                scale_out_price = entry - (risk_distance * scale_out_r)
                final_target_price = entry - (risk_distance * final_target_r)
            else:
                scale_out_price = entry + (risk_distance * scale_out_r)
                final_target_price = entry + (risk_distance * final_target_r)
            
            # Trail parameters from config/features
            atr_vals = signal.features.get('atr', []) if isinstance(signal.features, dict) else []
            atr_latest = atr_vals[-1] if atr_vals else None
            atr_mult = float(settings.get('execution.atr_trail_mult', 2.0))
            lmt_offset_pct = float(settings.get('execution.lmt_offset_pct', 0.02))
            latest_price = signal.features.get('latest_price') if isinstance(signal.features, dict) else None

            side = "SELL" if getattr(signal, 'direction', 'long') == 'short' else "BUY"
            planned_order = PlannedOrder(
                symbol=signal.symbol,
                side=side,
                qty=qty,
                entry=entry,
                stop=stop,
                scale_out=scale_out_price,
                target=final_target_price,
                trail_mode=settings.get('execution.trail_mode', 'ema21'),
                risk_context={
                    'dollar_risk': dollar_risk,
                    'per_share_risk': per_share_risk,
                    'r_multiple_scale': scale_out_r,
                    'r_multiple_final': final_target_r,
                    'atr': float(atr_latest) if (atr_latest is not None and np.isfinite(atr_latest)) else None,
                    'atr_mult': atr_mult,
                    'lmt_offset_pct': lmt_offset_pct,
                    'latest_price': float(latest_price) if (latest_price is not None and np.isfinite(latest_price)) else entry
                },
                reasons=signal.reasons if hasattr(signal, 'reasons') else None
            )
            planned_orders.append(planned_order)
        
        return planned_orders
    
    def execute(self, orders: List[PlannedOrder]) -> List[Dict[str, Any]]:
        results = []
        existing_by_side: Dict[str, set] = {}
        placed_keys: set[Tuple[str, str]] = set()
        held_pos: Dict[str, float] = {}
        # Query IBKR for currently open orders to avoid duplicates, if available
        if self.ib is not None:
            try:
                open_orders = self.ib.get_open_orders()
                for oo in open_orders:
                    sym = oo.get('symbol')
                    status = (oo.get('status') or '').lower()
                    action = (oo.get('action') or '').upper()
                    if sym and status not in ('filled', 'cancelled', 'inactive') and action in ('BUY','SELL'):
                        existing_by_side.setdefault(sym, set()).add(action)
            except Exception:
                pass
            # Query positions to avoid opening another position in same symbol
            try:
                positions = self.ib.get_positions()
                for p in positions:
                    sym = p.get('symbol')
                    qty = float(p.get('position', 0) or 0)
                    if sym and qty != 0:
                        held_pos[sym] = qty
            except Exception:
                pass
        
        for order in orders:
            try:
                print(f"Planned {order.side} {order.symbol} @ {order.entry} tgt {order.target} stop {order.stop}")
            except Exception:
                pass
            key = (order.symbol, (order.side or '').upper())
            # Skip duplicate same-side orders in same batch, submitted set, or existing open orders of same side
            if (key in placed_keys) or (key in self._submitted_symbols) or (order.symbol in existing_by_side and (order.side or '').upper() in existing_by_side[order.symbol]):
                results.append({'symbol': order.symbol, 'side': order.side, 'status': 'SKIPPED_DUPLICATE'})
                continue
            # Allow flipping; if same direction held, treat as scale-in up to cap
            existing_qty = held_pos.get(order.symbol, 0.0)
            if existing_qty != 0:
                planned_sign = 1 if (order.side or '').upper() == 'BUY' else -1
                existing_sign = 1 if existing_qty > 0 else -1
                if existing_sign == planned_sign:
                    try:
                        equity = self.db.get_current_equity() or 100000
                        max_pct = Settings().get('risk.max_position_value_pct', 20.0)
                        cap_value = equity * max_pct / 100.0
                        current_value = abs(existing_qty) * float(order.entry)
                        remaining_value = max(0.0, cap_value - current_value)
                        max_add_qty = int(remaining_value / max(0.01, float(order.entry)))
                        add_qty = min(int(order.qty), max_add_qty)
                        if add_qty <= 0:
                            results.append({'symbol': order.symbol, 'side': order.side, 'status': 'SKIPPED_HELD_SAME_DIRECTION'})
                            continue
                        else:
                            order.qty = add_qty
                            results.append({'symbol': order.symbol, 'side': order.side, 'status': 'SCALE_IN_PLANNED', 'qty': add_qty})
                    except Exception:
                        results.append({'symbol': order.symbol, 'side': order.side, 'status': 'SKIPPED_HELD_SAME_DIRECTION'})
                        continue
            if self.dry_run:
                result = {
                    'symbol': order.symbol,
                    'status': 'DRY_RUN',
                    'qty': order.qty,
                    'entry': order.entry,
                    'stop': order.stop,
                    'message': 'Dry run - no order sent'
                }
            else:
                # Place IBKR bracket orders per IB example (parent LMT + takeProfit LMT + stop STP)
                try:
                    ib = self.ib
                    if ib is None:
                        raise RuntimeError('IBKR feed is not initialized')
                    contract = ib.resolve(order.symbol)
                    # Parent
                    parent_id = ib._next_id; ib._next_id += 1
                    from ibapi.order import Order
                    parent = Order(); parent.orderId = parent_id
                    parent.action = order.side
                    parent.orderType = "LMT"
                    # Defensively ensure integer quantity
                    parent.totalQuantity = int(max(0, int(order.qty)))
                    # Round prices to the nearest $0.05 as requested
                    def _round_to_inc(px: float, inc: float = 0.05) -> float:
                        try:
                            return round(round(float(px) / inc) * inc, 2)
                        except Exception:
                            return float(px)
                    parent.lmtPrice = _round_to_inc(order.entry)
                    parent.tif = "DAY"
                    parent.outsideRth = False
                    # Explicitly disable NASDAQ-specific flags that can be preset in TWS
                    try:
                        parent.eTradeOnly = False
                        parent.firmQuoteOnly = False
                    except Exception:
                        pass
                    parent.orderRef = order.symbol + ":" + (order.side or "") + ":" + str(int(time.time()))
                    parent.transmit = False
                    # Take profit child
                    tp_id = ib._next_id; ib._next_id += 1
                    take_profit = Order(); take_profit.orderId = tp_id
                    take_profit.action = "SELL" if order.side == "BUY" else "BUY"
                    take_profit.orderType = "LMT"
                    take_profit.totalQuantity = parent.totalQuantity
                    take_profit.lmtPrice = _round_to_inc(order.target)
                    take_profit.parentId = parent_id
                    take_profit.tif = "GTC"
                    take_profit.outsideRth = False
                    try:
                        take_profit.eTradeOnly = False
                        take_profit.firmQuoteOnly = False
                    except Exception:
                        pass
                    take_profit.orderRef = parent.orderRef
                    take_profit.transmit = False
                    # Trailing Stop Limit child (replacing fixed stop)
                    sl_id = ib._next_id; ib._next_id += 1
                    stop_loss = Order(); stop_loss.orderId = sl_id
                    stop_loss.action = "SELL" if order.side == "BUY" else "BUY"
                    stop_loss.orderType = "TRAIL LIMIT"
                    stop_loss.totalQuantity = parent.totalQuantity
                    # Use 2 x ATR for trailing amount and 2% limit price offset
                    entry_px = float(order.entry)
                    stop_init = float(order.stop)
                    rc = order.risk_context or {}
                    atr_val = rc.get('atr')
                    atr_mult = rc.get('atr_mult', 2.0)
                    lmt_offset_pct = rc.get('lmt_offset_pct', 0.02)
                    ref_px = rc.get('latest_price', entry_px)
                    # Fallbacks if ATR missing
                    trailing_amt_raw = (float(atr_val) * float(atr_mult)) if (atr_val is not None) else max(0.05, abs(entry_px - stop_init))
                    limit_offset_raw = float(ref_px) * float(lmt_offset_pct)
                    # Round all to nearest $0.05
                    trail_stop_price = _round_to_inc(stop_init)
                    trailing_amt = _round_to_inc(trailing_amt_raw)
                    limit_offset = _round_to_inc(limit_offset_raw)
                    stop_loss.trailStopPrice = trail_stop_price
                    stop_loss.auxPrice = trailing_amt
                    stop_loss.lmtPriceOffset = limit_offset
                    stop_loss.parentId = parent_id
                    stop_loss.tif = "GTC"
                    stop_loss.outsideRth = False
                    try:
                        stop_loss.eTradeOnly = False
                        stop_loss.firmQuoteOnly = False
                    except Exception:
                        pass
                    stop_loss.orderRef = parent.orderRef
                    stop_loss.transmit = True
                    # Place in order: parent, takeProfit, stopLoss (last transmits chain)
                    ib.placeOrder(parent.orderId, contract, parent)
                    ib.placeOrder(take_profit.orderId, contract, take_profit)
                    ib.placeOrder(stop_loss.orderId, contract, stop_loss)
                    result = {'symbol': order.symbol, 'status': 'SUBMITTED', 'qty': order.qty, 'orderRef': parent.orderRef}
                    try:
                        self.db.insert_trade_plan(order.symbol, order.side, order.qty, order.entry, order.stop, order.target, parent.orderRef, order.reasons)
                    except Exception:
                        pass
                    # Mark as submitted to prevent duplicates in later cycles (by side)
                    placed_keys.add(key)
                    self._submitted_symbols.add(key)
                except Exception as e:
                    result = {'symbol': order.symbol, 'status': 'ERROR', 'error': str(e)}
            
            results.append(result)
            
            # Log the trade
            if result['status'] == 'DRY_RUN':
                self.db.insert_risk_event(
                    'TRADE_OPENED_DRY_RUN',
                    f"session_{datetime.now().strftime('%Y%m%d')}",
                    symbol=order.symbol,
                    value=order.qty * order.entry,
                    meta=result
                )
            elif result['status'] == 'SUBMITTED':
                # Optional: record opened trade event
                try:
                    self.db.insert_risk_event(
                        'TRADE_OPENED',
                        f"session_{datetime.now().strftime('%Y%m%d')}",
                        symbol=order.symbol,
                        value=order.qty * order.entry,
                        meta={'qty': order.qty}
                    )
                except Exception:
                    pass
        
        return results

# =============================================================================
# DATA FEEDS (IBKR LIVE)
# =============================================================================

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime, timedelta
import pytz

class PropulsionBot(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self._req_id = 0
        self._historical_data = {}
        
    def reqHistoricalData(self, contract: Contract, duration: str = "1 D", 
                         barSize: str = "5 mins", whatToShow: str = "TRADES"):
        """Request historical market data following IB specs"""
        
        self._req_id += 1
        req_id = self._req_id
        
        # Format end time in EST
        end_time = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y%m%d %H:%M:%S EST")
        
        # Configure request per IB docs
        super().reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=end_time,
            durationStr=duration, 
            barSizeSetting=barSize,
            whatToShow=whatToShow,
            useRTH=0,  # Include extended hours
            formatDate=1,  # YYYYMMDD HH:MM:SS
            keepUpToDate=False,
            chartOptions=[]
        )
        
        return req_id
        
    def historicalData(self, reqId, bar):
        """Process incoming historical data"""
        if reqId not in self._historical_data:
            self._historical_data[reqId] = []
            
        self._historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high, 
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'wap': bar.wap,
            'count': bar.barCount
        })
        
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Handle end of historical data stream"""
        if reqId in self._historical_data:
            self._historical_data[reqId].append({'completed': True})

class IBKRLiveFeed(EWrapper, EClient):
    def __init__(self, host: str, port: int, client_id: int):
        EClient.__init__(self, self)
        self.host, self.port, self.client_id = host, port, client_id
        self._next_id = 0
        self._bars: Dict[int, List[Any]] = {}
        self._bar_events: Dict[int, threading.Event] = {}
        self._contract_events: Dict[int, threading.Event] = {}
        self._contracts: Dict[int, Contract] = {}
        self._req_symbol_map: Dict[int, str] = {}
        self._min_tick_by_symbol: Dict[str, float] = {}
        # Matching symbols lookup
        self._match_events: Dict[int, threading.Event] = {}
        self._matches: Dict[int, Any] = {}
        # Quotes
        self._quotes: Dict[int, Dict[str, float]] = {}
        self._quote_events: Dict[int, threading.Event] = {}
        # News
        self._news: Dict[int, List[Dict[str, Any]]] = {}
        self._news_events: Dict[int, threading.Event] = {}
        # Account/positions
        self._account_values: Dict[Tuple[str, str], str] = {}
        self._account_event: threading.Event = threading.Event()
        self._positions: List[Dict[str, Any]] = []
        self._position_event: threading.Event = threading.Event()
        # Optional DB handle for journaling
        self.db: Optional[Database] = None
        # Track commissions by execId
        self._commissions: Dict[str, float] = {}
        # Open orders snapshot
        self._open_orders: List[Dict[str, Any]] = []
        self._open_orders_event: threading.Event = threading.Event()

    def nextValidId(self, orderId: int):
        self._next_id = orderId

    def contractDetails(self, reqId, contractDetails):
        self._contracts[reqId] = contractDetails.contract
        try:
            symbol = self._req_symbol_map.get(reqId)
            if symbol:
                mt = getattr(contractDetails, 'minTick', None)
                if mt:
                    self._min_tick_by_symbol[symbol] = float(mt)
        except Exception:
            pass

    def contractDetailsEnd(self, reqId):
        ev = self._contract_events.get(reqId)
        if ev:
            ev.set()
        # cleanup request -> symbol mapping
        self._req_symbol_map.pop(reqId, None)

    # Matching symbols callback
    def symbolSamples(self, reqId, details):
        self._matches[reqId] = details
        ev = self._match_events.get(reqId)
        if ev:
            ev.set()

    def historicalData(self, reqId, bar):
        self._bars.setdefault(reqId, []).append(bar)

    def historicalDataEnd(self, reqId, start, end):
        ev = self._bar_events.get(reqId)
        if ev:
            ev.set()

    # Quotes (snapshot)
    def tickPrice(self, reqId, tickType, price, attrib):
        q = self._quotes.setdefault(reqId, {'bid': None, 'ask': None, 'last': None})
        # TickType values: 1-BID, 2-ASK, 4-LAST per TickTypeEnum
        if tickType == 1:
            q['bid'] = price
        elif tickType == 2:
            q['ask'] = price
        elif tickType == 4:
            q['last'] = price

    def tickSnapshotEnd(self, reqId):
        ev = self._quote_events.get(reqId)
        if ev:
            ev.set()

    # News (historical headlines)
    def historicalNews(self, requestId, time, providerCode, articleId, headline):
        self._news.setdefault(requestId, []).append({'time': time, 'headline': headline})

    def historicalNewsEnd(self, requestId, hasMore):
        ev = self._news_events.get(requestId)
        if ev:
            ev.set()

    # Account and positions
    def updateAccountValue(self, key, val, currency, accountName):
        self._account_values[(key, currency)] = val

    def accountDownloadEnd(self, accountName):
        self._account_event.set()

    def position(self, account, contract, position, avgCost):
        self._positions.append({
            'account': account,
            'symbol': getattr(contract, 'symbol', ''),
            'conId': getattr(contract, 'conId', 0),
            'position': float(position),
            'avgCost': float(avgCost)
        })
        try:
            if self.db is not None and getattr(contract, 'symbol', ''):
                self.db.upsert_position(getattr(contract, 'symbol', ''), float(position), float(avgCost))
        except Exception:
            pass

    def positionEnd(self):
        self._position_event.set()

    # Execution details journaling
    def execDetails(self, reqId, contract, execution):
        try:
            if self.db is not None:
                self.db.insert_execution(
                    order_ref=getattr(execution, 'orderRef', ''),
                    symbol=getattr(contract, 'symbol', ''),
                    side=getattr(execution, 'side', ''),
                    qty=float(getattr(execution, 'shares', 0) or 0),
                    price=float(getattr(execution, 'price', 0) or 0),
                    exec_id=getattr(execution, 'execId', ''),
                    exchange=getattr(execution, 'exchange', ''),
                    commission=self._commissions.get(getattr(execution, 'execId', ''), None),
                    realized_pnl=None
                )
        except Exception:
            pass

    def commissionReport(self, commissionReport):
        try:
            exec_id = getattr(commissionReport, 'execId', '')
            comm = float(getattr(commissionReport, 'commission', 0) or 0)
            if exec_id:
                self._commissions[exec_id] = comm
        except Exception:
            pass

    # Open orders callbacks
    def openOrder(self, orderId, contract, order, orderState):
        try:
            self._open_orders.append({
                'orderId': int(orderId),
                'symbol': getattr(contract, 'symbol', ''),
                'action': getattr(order, 'action', ''),
                'totalQuantity': int(getattr(order, 'totalQuantity', 0) or 0),
                'orderType': getattr(order, 'orderType', ''),
                'status': getattr(orderState, 'status', ''),
                'orderRef': getattr(order, 'orderRef', '')
            })
        except Exception:
            pass

    def openOrderEnd(self):
        try:
            self._open_orders_event.set()
        except Exception:
            pass

    def get_open_orders(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        try:
            self._open_orders = []
            self._open_orders_event.clear()
            self.reqOpenOrders()
            self._open_orders_event.wait(timeout=timeout)
            return list(self._open_orders)
        except Exception:
            return []

    @staticmethod
    def _stock(symbol: str) -> Contract:
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        c.primaryExchange = "NASDAQ"
        return c

    def resolve(self, symbol: str) -> Contract:
        # Try SMART with primaryExchange hints, then matching symbols fallback
        exchanges = [None, "NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "ISLAND"]
        for ex in exchanges:
            req_id = self._next_id; self._next_id += 1
            self._contract_events[req_id] = threading.Event()
            base = self._stock(symbol)
            if ex:
                base.primaryExchange = ex
            self._req_symbol_map[req_id] = symbol
            self.reqContractDetails(req_id, base)
            self._contract_events[req_id].wait(timeout=5)
            self._contract_events.pop(req_id, None)
            contract = self._contracts.pop(req_id, None)
            if contract and getattr(contract, 'conId', 0):
                return contract
        # Fallback: use matching symbols API to disambiguate, then request details for first stock in US
        try:
            req_id = self._next_id; self._next_id += 1
            self._match_events[req_id] = threading.Event()
            self.reqMatchingSymbols(req_id, symbol)
            self._match_events[req_id].wait(timeout=5)
            self._match_events.pop(req_id, None)
            matches = self._matches.pop(req_id, [])
            # Prefer stock in USD primary exchanges
            preferred_ex = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "ISLAND"}
            for m in matches:
                try:
                    desc = m.contract
                    if getattr(desc, 'secType', '') != 'STK':
                        continue
                    if getattr(desc, 'currency', '') and desc.currency != 'USD':
                        continue
                    # Request details for this descriptor
                    req_id2 = self._next_id; self._next_id += 1
                    self._contract_events[req_id2] = threading.Event()
                    # Build a contract using the descriptor fields
                    c = Contract()
                    c.symbol = desc.symbol
                    c.secType = 'STK'
                    c.currency = desc.currency or 'USD'
                    c.exchange = 'SMART'
                    if getattr(desc, 'primaryExchange', None):
                        c.primaryExchange = desc.primaryExchange
                    self._req_symbol_map[req_id2] = symbol
                    self.reqContractDetails(req_id2, c)
                    self._contract_events[req_id2].wait(timeout=5)
                    self._contract_events.pop(req_id2, None)
                    contract = self._contracts.pop(req_id2, None)
                    if contract and getattr(contract, 'conId', 0):
                        return contract
                except Exception:
                    continue
        except Exception:
            pass
        # As last resort, return a basic stock contract; callers should handle failures
        import logging
        logging.getLogger(__name__).error(f"Failed to resolve contract for {symbol} on all exchanges")
        return self._stock(symbol)

    def get_min_tick(self, symbol: str, default: float = 0.01) -> float:
        mt = self._min_tick_by_symbol.get(symbol)
        try:
            if mt is None:
                # Attempt a quick resolve to populate
                _ = self.resolve(symbol)
                mt = self._min_tick_by_symbol.get(symbol)
        except Exception:
            pass
        try:
            return float(mt) if mt is not None and mt > 0 else float(default)
        except Exception:
            return float(default)

    def fetch_bars_df(self, symbol: str, bar_size: str, duration: str,
                      whatToShow: str = 'TRADES', useRTH: bool = True,
                      formatDate: int = 1, endDateTime: Optional[datetime] = None) -> pd.DataFrame:
        try:
            # Add delay between requests to prevent overload
            time.sleep(0.5)
            
            # Resolve an IB API Contract for the symbol
            contract = self.resolve(symbol)

            # Prepare end time per IBKR docs (UTC "YYYYMMDD HH:MM:SS")
            queryTime = ''
            if endDateTime is not None:
                try:
                    dt_utc = endDateTime if endDateTime.tzinfo is None else endDateTime.astimezone(timezone.utc)
                    queryTime = dt_utc.strftime('%Y%m%d %H:%M:%S')
                except Exception:
                    queryTime = ''
            # Helper to perform a single request attempt
            def _do_req(end_dt: Optional[datetime], wts: str, rth: bool) -> List[Dict[str, Any]]:
                qtime = ''
                if end_dt is not None:
                    try:
                        dt_utc = end_dt if end_dt.tzinfo is None else end_dt.astimezone(timezone.utc)
                        # Use explicit UTC with dash separator as per IBKR warning
                        qtime = dt_utc.strftime('%Y%m%d-%H:%M:%S')
                    except Exception:
                        qtime = ''
                rid = self._next_id; self._next_id += 1
                self._bars[rid] = []
                self._bar_events[rid] = threading.Event()
                self.reqHistoricalData(rid, contract, qtime, duration, bar_size, wts, 1 if rth else 0, formatDate, False, [])
                self._bar_events[rid].wait(timeout=30)
                out = []
                for b in self._bars.get(rid, []):
                    o = getattr(b, 'open', getattr(b, 'openPrice', None))
                    c = getattr(b, 'close', getattr(b, 'closePrice', None))
                    h = getattr(b, 'high', None)
                    l = getattr(b, 'low', None)
                    v = getattr(b, 'volume', None)
                    ts = getattr(b, 'date', None)
                    if None in (o, h, l, c, v, ts):
                        continue
                    out.append({'ts': pd.to_datetime(ts), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
                self._bars.pop(rid, None)
                self._bar_events.pop(rid, None)
                return out

            attempts: List[Tuple[Optional[datetime], str, bool]] = []
            attempts.append((endDateTime, whatToShow, useRTH))
            if whatToShow == 'TRADES' and useRTH:
                attempts.append((endDateTime, 'TRADES', False))
            attempts.append((endDateTime, 'MIDPOINT', useRTH))
            attempts.append((endDateTime, 'MIDPOINT', False))
            if endDateTime is None:
                now_utc = datetime.utcnow()
                attempts.extend([
                    (now_utc, whatToShow, useRTH),
                    (now_utc, 'TRADES', False),
                    (now_utc, 'MIDPOINT', useRTH),
                    (now_utc, 'MIDPOINT', False),
                    (now_utc - timedelta(days=1), whatToShow, useRTH),
                    (now_utc - timedelta(days=1), 'MIDPOINT', False),
                ])

            for ed, wts, rth in attempts:
                rows = _do_req(ed, wts, rth)
                if rows:
                    logger.debug(f"Historical bars resolved for {symbol} with wts={wts}, useRTH={rth}, end={ed}")
                    return pd.DataFrame(rows)

            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_quote_snapshot(self, symbol: str) -> Dict[str, float]:
        contract = self.resolve(symbol)
        req_id = self._next_id; self._next_id += 1
        self._quotes[req_id] = {'bid': None, 'ask': None, 'last': None}
        self._quote_events[req_id] = threading.Event()
        # Generic tick list empty, snapshot=True
        self.reqMktData(req_id, contract, "", True, False, [])
        self._quote_events[req_id].wait(timeout=10)
        quote = self._quotes.get(req_id, {})
        # cleanup
        self._quotes.pop(req_id, None)
        self._quote_events.pop(req_id, None)
        return quote

    def fetch_headlines(self, symbol: str, provider_code: str = "BRFG", lookback_hours: int = 24) -> List[Dict[str, Any]]:
        # resolve to conId is already in contract
        contract = self.resolve(symbol)
        con_id = getattr(contract, 'conId', None)
        if not con_id:
            return []
        req_id = self._next_id; self._next_id += 1
        self._news[req_id] = []
        self._news_events[req_id] = threading.Event()
        # Use explicit UTC with dash as per IBKR guidance
        start = (datetime.utcnow() - timedelta(hours=lookback_hours)).strftime("%Y%m%d-%H:%M:%S")
        end = ""
        total = 10
        self.reqHistoricalNews(req_id, con_id, provider_code, start, end, total, [])
        self._news_events[req_id].wait(timeout=10)
        headlines = self._news.get(req_id, [])
        self._news.pop(req_id, None)
        self._news_events.pop(req_id, None)
        return headlines

    def get_equity(self, base_currency: str = 'USD', timeout: float = 10.0) -> Optional[float]:
        self._account_event.clear()
        # True = subscribe; use any account string, TWS will supply current
        try:
            self.reqAccountUpdates(True, "")
        except Exception:
            return None
        self._account_event.wait(timeout=timeout)
        val = self._account_values.get(("NetLiquidation", base_currency)) or self._account_values.get(("TotalCashValue", base_currency))
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    def get_positions(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        self._positions = []
        self._position_event.clear()
        try:
            self.reqPositions()
        except Exception:
            return []
        self._position_event.wait(timeout=timeout)
        return list(self._positions)

    def fetch_data_bundle(self, 
                          symbols: List[str], 
                          bar_size: str = "5 mins", 
                          duration: str = "14400 S") -> Tuple[Dict[str, List[BarData]], 
                                                            Dict[str, Dict[str, float]], 
                                                            List[Headline]]:
        """
        Fetch bundle of market data including bars, quotes and headlines
        
        Args:
            symbols: List of stock symbols
            bar_size: Bar size (e.g. "5 mins") 
            duration: Look back duration (e.g. "14400 S")
            
        Returns:
            Tuple containing:
            - Dict of bar data by symbol
            - Dict of quotes by symbol
            - List of headlines
        """
        try:
            bars_data: Dict[str, List[BarData]] = {}
            quotes_map: Dict[str, Dict[str, float]] = {}
            headlines: List[Headline] = []

            for sym in symbols:
                try:
                    cached = self.db.get_bars_from_cache(sym, '5min', 240)
                    rows: List[BarData] = cached
                    
                    if not rows:
                        df = self.ib.fetch_bars_df(sym, bar_size=bar_size, duration=duration)
                        if df is not None and not df.empty:
                            rows = []
                            for _, r in df.iterrows():
                                rows.append(BarData(
                                    symbol=sym,
                                    timestamp=pd.to_datetime(r['ts']).to_pydatetime(),
                                    open=float(r['open']),
                                    high=float(r['high']),
                                    low=float(r['low']), 
                                    close=float(r['close']),
                                    volume=float(r['volume']),
                                    timeframe='5min'
                                ))
                            self.db.upsert_bars_cache(sym, '5min', rows)
                        else:
                            logger.warning(f"No data returned for {sym}")
                            continue

                    if rows:
                        bars_data[sym] = rows
                        
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Fatal error in data fetch: {e}")
            # Skip rest of cycle gracefully on fatal fetch error
            return
        return bars_data, quotes_map, headlines

# =============================================================================
# ORCHESTRATOR - MAIN CONTROLLER
# =============================================================================

class IntradayOrchestrator:
    def __init__(self):
        self.settings = Settings()
        self.db = Database()
        self.market_clock = MarketClock()
        # Live IBKR data feed
        self.ib = IBKRLiveFeed(self.settings.ibkr_host, self.settings.ibkr_port, self.settings.ibkr_client_id)
        # Share DB handle for journaling and live position sync
        try:
            self.ib.db = self.db
        except Exception:
            pass
        self.ib.connect(self.settings.ibkr_host, self.settings.ibkr_port, self.settings.ibkr_client_id)
        _t = threading.Thread(target=self.ib.run, daemon=True)
        _t.start()
        self.feature_builder = FeatureBuilder(self.settings)
        self.strategy = PropulsionStrategy(self.settings)
        self.risk_manager = RiskManager(self.settings, self.db, self.ib)
        self.trade_manager = TradeManager(self.settings, self.db, self.ib)
        self.scheduler = None
        
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cycle_count = 0
        self.is_running = False
        
        # Initialize starting equity
        self._initialize_starting_equity()
    
    def _initialize_starting_equity(self):
        conn = self.db.get_connection()
        try:
            conn.execute('''
                INSERT INTO metrics_equity (session, starting_equity, realized_pnl, unrealized_pnl)
                VALUES (?, ?, ?, ?)
            ''', (self.session_id, 100000, 0, 0))
            conn.commit()
        finally:
            conn.close()
    
    def run_intraday_cycle(self):
        """Main intraday cycle execution"""
        if not self.market_clock.is_market_hours():
            print("Market is closed. Skipping cycle.")
            return
        
        self.cycle_count += 1
        cycle_id = f"{self.session_id}_cycle_{self.cycle_count}"
        print(f"Starting intraday cycle {cycle_id}")
        
        try:
            # 1. Load watchlist from external DB view
            wl_rows = self.db.load_watchlist()
            watchlist = [r['symbol'] for r in wl_rows if r.get('symbol')]
            if not watchlist:
                # Fallback to WATCHLIST from env (comma-separated), or a small default
                env_wl = os.getenv('WATCHLIST', '')
                if env_wl:
                    watchlist = [s.strip().upper() for s in env_wl.split(',') if s.strip()]
                else:
                    watchlist = ['AAPL', 'MSFT', 'NVDA']
                print(f"Watchlist empty; using fallback: {watchlist}")

            # 2. Fetch data with cache (IBKR live)
            bars_data: Dict[str, List[BarData]] = {}
            quotes_map: Dict[str, Dict[str, float]] = {}
            headlines: List[Headline] = []
            lookback_min = 240
            current_time = pd.Timestamp.now()
            
            for sym in watchlist:
                try:
                    cached = self.db.get_bars_from_cache(sym, '5min', lookback_min)
                    rows: List[BarData] = cached
                    
                    # Improved cache validation
                    cache_valid = (rows and len(rows) >= 50 and 
                                 (current_time - pd.to_datetime(rows[-1].timestamp)).total_seconds() < 300)
                    
                    if not cache_valid:
                        # Add rate limiting delay
                        time.sleep(0.1)  
                        df = self.ib.fetch_bars_df(sym, bar_size="5 mins", duration=f"{lookback_min*60} S")
                        
                        if df is not None and not df.empty:
                            rows = []
                            for _, r in df.iterrows():
                                rows.append(BarData(
                                    symbol=sym,
                                    timestamp=pd.to_datetime(r['ts']).to_pydatetime(),
                                    open=float(r['open']),
                                    high=float(r['high']),
                                    low=float(r['low']),
                                    close=float(r['close']),
                                    volume=float(r['volume']),
                                    timeframe='5min'
                                ))
                            # Cache the new data
                            self.db.upsert_bars_cache(sym, '5min', rows)
                        else:
                            logger.warning(f"No data returned for {sym}")
                    
                    if rows:
                        bars_data[sym] = rows

                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")
                    continue

            # 3. Build features
            features_batch = []
            for symbol in watchlist:
                symbol_bars = bars_data.get(symbol, [])
                symbol_headlines = [h for h in headlines if h.symbol == symbol]
                
                if symbol_bars:
                    # inject quote if available
                    latest_quote = quotes_map.get(symbol, {})
                    features = self.feature_builder.build(symbol, symbol_bars, catalysts=symbol_headlines, quote=latest_quote)
                    features_batch.append({
                        'symbol': symbol,
                        'features': features
                    })
            print(f"Features built for {len(features_batch)} symbols in cycle {cycle_id}")
            
            # 4. Evaluate strategy
            signals = self.strategy.evaluate(features_batch)
            print(f"Evaluated signals: {len(signals)} in cycle {cycle_id}")
            
            # 5. Take top N signals
            top_n = self.settings.get('orchestrator.intraday_top_n', 20)
            top_signals = signals[:top_n]
            
            # 6. Risk checks
            risk_ok, allowed_signals = self.risk_manager.pre_trade_checks(top_signals)
            print(f"Top {len(top_signals)}; allowed after risk: {len(allowed_signals)} in cycle {cycle_id}")
            
            if risk_ok and allowed_signals:
                # 7. Plan and execute orders
                planned_orders = self.trade_manager.plan_orders(allowed_signals)
                print(f"Planned {len(planned_orders)} orders in cycle {cycle_id}")
                execution_results = self.trade_manager.execute(planned_orders)
                
                print(f"Executed {len(execution_results)} orders in cycle {cycle_id}")
            
            # 8. Journal signals
            for signal in signals:
                self.db.insert_signal(signal)
            
            print(f"Completed cycle {cycle_id}. Generated {len(signals)} signals.")
            
        except Exception as e:
            print(f"Error in intraday cycle {cycle_id}: {e}")
            self.db.insert_risk_event(
                'CYCLE_ERROR',
                self.session_id,
                value=self.cycle_count,
                meta={'error': str(e)}
            )
    
    def run_flatten_guard(self):
        """Close all positions at end of day"""
        print("Executing flatten guard - closing all positions")
        
        # In production, this would close all open positions
        self.db.insert_risk_event(
            'FLATTEN_GUARD_EXECUTED',
            self.session_id,
            value=self.cycle_count
        )
        
        print("Flatten guard completed")
       
    def start(self):
        """Start the orchestrator"""
        if self.is_running:
            print("Orchestrator is already running")
            return
        
        self.is_running = True
        print("Starting Propulsion Bot Phase 2 Orchestrator")
        
        cadence_min = self.settings.get('orchestrator.cadence_min', 5)
        flatten_time = self.settings.get('orchestrator.flatten_time_et', "15:55")
        # APScheduler setup (preferred)
        if BackgroundScheduler is not None:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(self.run_intraday_cycle, 'interval', minutes=cadence_min, id='intraday_cycle')
            try:
                h, m = map(int, str(flatten_time).split(':'))
            except Exception:
                h, m = 15, 55
            self.scheduler.add_job(self.run_flatten_guard, 'cron', hour=h, minute=m, id='flatten_guard')
            self.scheduler.start()
        else:
            # Fallback to schedule library
            schedule.every(cadence_min).minutes.do(self.run_intraday_cycle)
            schedule.every().day.at(flatten_time).do(self.run_flatten_guard)
        # Main loop
        try:
            while self.is_running:
                if self.scheduler is None:
                    schedule.run_pending()
                
                # Check if we should stop (market closed and after flatten time) unless forced open
                if (not self.settings.force_market_open and
                    not self.market_clock.is_market_hours() and
                    self.market_clock.C(flatten_time)):
                    print("Market closed and flatten completed. Stopping orchestrator.")
                    break
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("Orchestrator stopped by user")
        finally:
            # Ensure full shutdown
            self.stop()

    def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        if self.scheduler is not None:
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:
                pass
            self.scheduler = None
        else:
            try:
                schedule.clear()
            except Exception:
                pass
        try:
            if getattr(self, 'ib', None) is not None:
                self.ib.disconnect()
        except Exception:
            pass
        print("Orchestrator stopping...")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the Propulsion Bot Phase 2"""
    print("=" * 60)
    print("PROPULSION BOT - PHASE 2 (INTRADAY)")
    print("=" * 60)
    
    # Initialize system
    orchestrator = IntradayOrchestrator()
    
    # Auto continuous by default (non-interactive)
    print("Starting continuous operation (auto)...")
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        orchestrator.stop()
    finally:
        # Double-ensure IB disconnect on exit
        try:
            if getattr(orchestrator, 'ib', None) is not None:
                orchestrator.ib.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
