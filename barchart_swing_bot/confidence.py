"""Utility functions for evaluating trade confidence.

The production system would normally rely on an ML model fed with a large
feature set.  For the purposes of the test suite we keep the implementation
transparent and deterministic while still modelling some of the behaviour one
would expect from a richer pipeline.  The module now supports additional
technical indicators, multiple timeframes and a small back‑testing helper to
evaluate a moving‑average strategy against historical prices.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple


# Weights for individual indicators.  Values are expressed as percentages and
# sum to 100 so the returned confidence is naturally in the ``0..100`` range.
WEIGHTS: Mapping[str, float] = {
    "wtd_alpha": 10,
    "momentum_daily": 15,
    "momentum_weekly": 15,
    "inst_score": 20,
    "rsi": 10,
    "stochastic": 10,
    "macd": 10,
    "ma_signal": 10,
}


def compute_confidence(features: Mapping[str, float]) -> float:
    """Return a confidence percentage derived from weighted indicators.

    ``features`` is expected to contain values in the ``0..1`` range for each
    key in :data:`WEIGHTS`.  Missing features contribute ``0`` to the final
    score.  The result is clipped to ``0..100`` for safety.
    """

    score = 0.0
    for key, weight in WEIGHTS.items():
        score += float(features.get(key, 0)) * weight
    return max(0.0, min(100.0, score))


def is_actionable(features: Mapping[str, float]) -> bool:
    """Return ``True`` when the computed confidence meets the 80 % threshold."""

    return compute_confidence(features) >= 80


def determine_best_timeframe(
    daily: Mapping[str, float], weekly: Mapping[str, float]
) -> Tuple[str, float, bool]:
    """Select the timeframe with the highest confidence.

    Returns a tuple of ``(timeframe, confidence, passes)`` where ``passes``
    indicates whether the winning timeframe meets the 80 % actionable
    threshold.
    """

    daily_conf = compute_confidence(daily)
    weekly_conf = compute_confidence(weekly)
    if daily_conf >= weekly_conf:
        return "daily", daily_conf, daily_conf >= 80
    return "weekly", weekly_conf, weekly_conf >= 80


def backtest_moving_average(
    prices: Sequence[float], short: int = 20, long: int = 50
) -> float:
    """Back‑test a simple moving‑average crossover strategy.

    The function iterates over ``prices`` and records the cumulative return of a
    long‑only strategy that buys when the short moving average crosses above the
    long moving average and sells on the inverse crossover.  The function
    returns the total return percentage (e.g. ``0.5`` for ``50 %``).
    """

    if long <= 0 or short <= 0 or long <= short:
        raise ValueError("long window must be greater than short window")
    if len(prices) < long + 1:
        return 0.0
    total_return = 0.0
    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    for idx in range(long, len(prices)):
        short_ma = sum(prices[idx - short : idx]) / short
        long_ma = sum(prices[idx - long : idx]) / long
        price = prices[idx]
        if short_ma > long_ma and position == 0:
            position = 1
            entry_price = price
        elif short_ma < long_ma and position == 1:
            total_return += price / entry_price - 1
            position = 0
    if position == 1:  # close any open position at the last price
        total_return += prices[-1] / entry_price - 1
    return total_return
