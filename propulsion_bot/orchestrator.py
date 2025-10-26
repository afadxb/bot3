"""High level orchestration for the Propulsion bot."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - optional dependency
    BackgroundScheduler = None  # type: ignore

import schedule

from . import logger
from .config import Settings
from .database import Database
from .dtos import BarData, Headline
from .features import FeatureBuilder
from .ibkr import IBKRLiveFeed
from .market import MarketClock
from .risk import RiskManager
from .strategy import PropulsionStrategy
from .trade import TradeManager


class IntradayOrchestrator:
    def __init__(self) -> None:
        self.settings = Settings()
        self.db = Database()
        self.market_clock = MarketClock()
        self.feature_builder = FeatureBuilder(self.settings.config)
        self.strategy = PropulsionStrategy(self.settings.config)
        self.ib = IBKRLiveFeed(
            self.settings.ibkr_host, self.settings.ibkr_port, self.settings.ibkr_client_id
        )
        self.ib.db = self.db
        self.risk_manager = RiskManager(self.settings.config, self.db, self.ib)
        self.trade_manager = TradeManager(self.settings.config, self.db, self.ib)
        self.scheduler = None
        self.is_running = False
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cycle_count = 0

    def run_intraday_cycle(self) -> None:
        self.cycle_count += 1
        cycle_id = f"cycle_{self.cycle_count}"
        print(f"Starting intraday cycle {cycle_id}")

        try:
            watchlist = [row.get("symbol") for row in self.db.load_watchlist() if row.get("symbol")]
            print(f"Loaded watchlist with {len(watchlist)} symbols")

            if not watchlist:
                print("Watchlist is empty; skipping cycle")
                return

            market_open_required = not self.settings.force_market_open
            if market_open_required and not self.market_clock.is_market_hours():
                print("Market is closed; skipping cycle")
                return

            lookback_min = self.settings.get("data.lookback_minutes", 240)
            bars_data: Dict[str, List[BarData]] = {}
            quotes_map: Dict[str, Dict[str, float]] = {}
            headlines: List[Headline] = []

            for sym in watchlist:
                try:
                    cached = self.db.get_bars_from_cache(sym, "5min", lookback_min)
                    rows: List[BarData] = cached

                    cache_valid = (
                        rows
                        and len(rows) >= 50
                        and (datetime.utcnow() - pd.to_datetime(rows[-1].timestamp)).total_seconds() < 300
                    )

                    if not cache_valid:
                        time.sleep(0.1)
                        df = self.ib.fetch_bars_df(sym, bar_size="5 mins", duration=f"{lookback_min * 60} S")

                        if df is not None and not df.empty:
                            rows = []
                            for _, record in df.iterrows():
                                rows.append(
                                    BarData(
                                        symbol=sym,
                                        timestamp=pd.to_datetime(record["ts"]).to_pydatetime(),
                                        open=float(record["open"]),
                                        high=float(record["high"]),
                                        low=float(record["low"]),
                                        close=float(record["close"]),
                                        volume=float(record["volume"]),
                                        timeframe="5min",
                                    )
                                )
                            self.db.upsert_bars_cache(sym, "5min", rows)
                        else:
                            logger.warning("No data returned for %s", sym)

                    if rows:
                        bars_data[sym] = rows

                except Exception as exc:
                    logger.error("Error processing %s: %s", sym, exc)
                    continue

            features_batch: List[Dict[str, object]] = []
            for symbol in watchlist:
                symbol_bars = bars_data.get(symbol, [])
                symbol_headlines = [headline for headline in headlines if headline.symbol == symbol]

                if symbol_bars:
                    latest_quote = quotes_map.get(symbol, {})
                    features = self.feature_builder.build(
                        symbol, symbol_bars, catalysts=symbol_headlines, quote=latest_quote
                    )
                    features_batch.append({"symbol": symbol, "features": features})
            print(f"Features built for {len(features_batch)} symbols in cycle {cycle_id}")

            signals = self.strategy.evaluate(features_batch)
            print(f"Evaluated signals: {len(signals)} in cycle {cycle_id}")

            top_n = self.settings.get("orchestrator.intraday_top_n", 20)
            top_signals = signals[:top_n]

            risk_ok, allowed_signals = self.risk_manager.pre_trade_checks(top_signals)
            print(f"Top {len(top_signals)}; allowed after risk: {len(allowed_signals)} in cycle {cycle_id}")

            if risk_ok and allowed_signals:
                planned_orders = self.trade_manager.plan_orders(allowed_signals)
                print(f"Planned {len(planned_orders)} orders in cycle {cycle_id}")
                execution_results = self.trade_manager.execute(planned_orders)

                print(f"Executed {len(execution_results)} orders in cycle {cycle_id}")

            for signal in signals:
                self.db.insert_signal(signal)

            print(f"Completed cycle {cycle_id}. Generated {len(signals)} signals.")

        except Exception as exc:
            print(f"Error in intraday cycle {cycle_id}: {exc}")
            self.db.insert_risk_event(
                "CYCLE_ERROR",
                self.session_id,
                value=self.cycle_count,
                meta={"error": str(exc)},
            )

    def run_flatten_guard(self) -> None:
        print("Executing flatten guard - closing all positions")

        self.db.insert_risk_event(
            "FLATTEN_GUARD_EXECUTED",
            self.session_id,
            value=self.cycle_count,
        )

        print("Flatten guard completed")

    def start(self) -> None:
        if self.is_running:
            print("Orchestrator is already running")
            return

        self.is_running = True
        print("Starting Propulsion Bot Phase 2 Orchestrator")

        cadence_min = self.settings.get("orchestrator.cadence_min", 5)
        flatten_time = self.settings.get("orchestrator.flatten_time_et", "15:55")
        if BackgroundScheduler is not None:
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(self.run_intraday_cycle, "interval", minutes=cadence_min, id="intraday_cycle")
            try:
                hour, minute = map(int, str(flatten_time).split(":"))
            except Exception:
                hour, minute = 15, 55
            self.scheduler.add_job(self.run_flatten_guard, "cron", hour=hour, minute=minute, id="flatten_guard")
            self.scheduler.start()
        else:
            schedule.every(cadence_min).minutes.do(self.run_intraday_cycle)
            schedule.every().day.at(flatten_time).do(self.run_flatten_guard)

        try:
            while self.is_running:
                if self.scheduler is None:
                    schedule.run_pending()

                if (
                    not self.settings.force_market_open
                    and not self.market_clock.is_market_hours()
                    and self.market_clock.should_flatten(flatten_time)
                ):
                    print("Market closed and flatten completed. Stopping orchestrator.")
                    break

                time.sleep(30)

        except KeyboardInterrupt:
            print("Orchestrator stopped by user")
        finally:
            self.stop()

    def stop(self) -> None:
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
            if getattr(self, "ib", None) is not None:
                self.ib.disconnect()
        except Exception:
            pass
        print("Orchestrator stopping...")


__all__ = ["IntradayOrchestrator"]

