"""Collection of technical indicator utilities."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return [None] * len(prices)

        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.tolist()

    @staticmethod
    def calculate_vwap(
        highs: List[float], lows: List[float], closes: List[float], volumes: List[float]
    ) -> List[float]:
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
    def calculate_atr(
        highs: List[float], lows: List[float], closes: List[float], period: int = 14
    ) -> List[float]:
        if len(highs) < period:
            return [None] * len(highs)

        tr = [highs[0] - lows[0]]

        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
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
    def calculate_supertrend(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 10,
        multiplier: float = 3.0,
    ) -> dict:
        if len(highs) < period:
            return {
                "trend": [None] * len(highs),
                "upper": [None] * len(highs),
                "lower": [None] * len(highs),
            }

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
            if trend[i - 1] == 1:
                if closes[i] > lower_band[i - 1]:
                    trend[i] = 1
                    lower[i] = max(lower_band[i], lower[i - 1])
                else:
                    trend[i] = -1
                    upper[i] = upper_band[i]
            else:
                if closes[i] < upper_band[i - 1]:
                    trend[i] = -1
                    upper[i] = min(upper_band[i], upper[i - 1])
                else:
                    trend[i] = 1
                    lower[i] = lower_band[i]

        return {"trend": trend, "upper": upper, "lower": lower}


__all__ = ["TechnicalIndicators"]

