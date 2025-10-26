"""Data transfer objects used throughout the Propulsion bot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    symbol: str
    entry_hint: Optional[float]
    stop_hint: Optional[float]
    direction: str  # "long" or "short"
    base_score: float
    ai_adj_score: float
    final_score: float
    reasons: List[str]
    rules_passed: Dict[str, bool]
    features: Dict[str, Any]
    cycle_id: str
    rank: int = 0

    def __post_init__(self) -> None:
        for field_name in ("base_score", "ai_adj_score", "final_score"):
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")


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
    timeframe: Optional[str] = "5min"


@dataclass
class Headline:
    symbol: str
    headline: str
    source: str
    timestamp: datetime
    url: Optional[str] = None


__all__ = [
    "Signal",
    "PlannedOrder",
    "RiskEvent",
    "BarData",
    "Headline",
]

