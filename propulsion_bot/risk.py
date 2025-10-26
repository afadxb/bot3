"""Risk management utilities."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple, TYPE_CHECKING

from .config import Settings
from .database import Database
from .dtos import Signal

if TYPE_CHECKING:
    from .ibkr import IBKRLiveFeed


class RiskManager:
    def __init__(self, config, db: Database, ib: Optional["IBKRLiveFeed"] = None) -> None:
        self.config = config
        self.db = db
        self.ib = ib
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def pre_trade_checks(self, signals: List[Signal]) -> Tuple[bool, List[Signal]]:
        allowed_signals: List[Signal] = []
        settings = Settings()

        todays_trades = self.db.get_todays_trade_count()
        trade_cap = settings.get("risk.daily_trade_cap", 20)

        if todays_trades >= trade_cap:
            self.db.insert_risk_event(
                "TRADE_CAP_REACHED",
                self.session_id,
                value=todays_trades,
                meta={"cap": trade_cap},
            )
            return False, []

        current_equity = None
        if self.ib is not None:
            current_equity = self.ib.get_equity() or None
        if current_equity is None:
            current_equity = self.db.get_current_equity()
        if current_equity:
            drawdown_pct = 0
            drawdown_halt = settings.get("risk.daily_drawdown_halt_pct", 10.0)

            if drawdown_pct >= drawdown_halt:
                self.db.insert_risk_event(
                    "DRAWDOWN_HALT",
                    self.session_id,
                    value=drawdown_pct,
                    meta={"halt_threshold": drawdown_halt},
                )
                return False, []

        exposure_pct = self._calculate_current_exposure()
        max_exposure = settings.get("risk.max_portfolio_exposure_pct", 100)

        if exposure_pct >= max_exposure:
            self.db.insert_risk_event(
                "EXPOSURE_CAP_REACHED",
                self.session_id,
                value=exposure_pct,
                meta={"cap": max_exposure},
            )
            return False, []

        for signal in signals:
            if self._check_signal_risk(signal):
                allowed_signals.append(signal)

        return True, allowed_signals

    def _calculate_current_exposure(self) -> float:
        equity = None
        if self.ib is not None:
            equity = self.ib.get_equity()
        if equity is None or equity <= 0:
            return 0.0
        total = 0.0
        positions = self.ib.get_positions() if self.ib is not None else []
        for position in positions:
            try:
                total += abs(float(position.get("position", 0.0)) * float(position.get("avgCost", 0.0)))
            except Exception:
                continue
        return (total / equity) * 100.0 if equity else 0.0

    def _check_signal_risk(self, signal: Signal) -> bool:
        settings = Settings()

        if settings.get("risk.illiquidity_veto", True):
            volume = signal.features.get("volumes", [])
            if volume and volume[-1] < 10:
                self.db.insert_risk_event(
                    "ILLIQUIDITY_VETO",
                    self.session_id,
                    symbol=signal.symbol,
                    value=volume[-1],
                )
                return False

        return True


__all__ = ["RiskManager"]

