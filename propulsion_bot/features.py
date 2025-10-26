"""Feature engineering utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Settings
from .dtos import BarData, Headline
from .indicators import TechnicalIndicators


class FeatureBuilder:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.indicators = TechnicalIndicators()

    def build(
        self,
        symbol: str,
        bars: List[BarData],
        fundamentals: Optional[Dict[str, Any]] = None,
        catalysts: Optional[List[Headline]] = None,
        quote: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if len(bars) < 21:
            return {}

        closes = [bar.close for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        volumes = [bar.volume for bar in bars]
        timestamps = [bar.timestamp for bar in bars]

        features: Dict[str, Any] = {}
        settings = Settings()

        features["ema_fast"] = self.indicators.calculate_ema(
            closes, settings.get("strategy.ema_fast", 9)
        )
        features["ema_slow"] = self.indicators.calculate_ema(
            closes, settings.get("strategy.ema_slow", 21)
        )
        features["ema_bias"] = self.indicators.calculate_ema(
            closes, settings.get("strategy.ema_bias", 50)
        )
        features["vwap"] = self.indicators.calculate_vwap(highs, lows, closes, volumes)
        features["atr"] = self.indicators.calculate_atr(highs, lows, closes)
        features["rsi"] = self.indicators.calculate_rsi(closes)

        features["volume_spike"] = self._calculate_volume_spike(volumes)
        features["consolidation"] = self._detect_consolidation(highs, lows, closes)
        features["catalyst_freshness"] = self._calculate_catalyst_freshness(
            catalysts, timestamps[-1]
        )

        try:
            if len(bars) >= 40:
                features["pattern_triangle"] = self._detect_triangle(highs, lows, closes, lookback=40)
            else:
                features["pattern_triangle"] = {"is_contracting": False, "breakout_up": False}
        except Exception:
            features["pattern_triangle"] = {"is_contracting": False, "breakout_up": False}

        try:
            if len(bars) >= 60:
                features["pattern_double_top"] = self._detect_double_top(highs, closes, volumes, lookback=60)
            else:
                features["pattern_double_top"] = False
        except Exception:
            features["pattern_double_top"] = False

        try:
            if len(bars) >= 60:
                features.update(
                    self._volume_profile_levels(lows, highs, closes, volumes, lookback=60, bins=30)
                )
        except Exception:
            pass

        if settings.get("strategy.enable_supertrend", False):
            supertrend_params = settings.get("strategy.supertrend", {})
            features["supertrend"] = self.indicators.calculate_supertrend(
                highs,
                lows,
                closes,
                supertrend_params.get("atr_period", 10),
                supertrend_params.get("atr_mult", 3.0),
            )

        if fundamentals:
            features.update(
                {
                    "market_cap": fundamentals.get("market_cap"),
                    "pe_ratio": fundamentals.get("pe_ratio"),
                    "volume_avg": fundamentals.get("volume_avg"),
                }
            )

        features["closes"] = closes
        features["highs"] = highs
        features["lows"] = lows
        features["volumes"] = volumes

        if quote:
            q_last = quote.get("last")
            q_ask = quote.get("ask")
            q_bid = quote.get("bid")
            latest_price = float(
                q_last if q_last is not None else (q_ask if q_ask is not None else closes[-1])
            )
            features["latest_price"] = latest_price
            if q_bid is not None and q_ask is not None:
                features["spread"] = float(q_ask) - float(q_bid)

        return features

    def _calculate_volume_spike(self, volumes: List[float], lookback: int = 20) -> List[float]:
        multiplier = Settings().get("strategy.vol_spike_multiple", 1.5)

        volume_spikes = []
        for i in range(len(volumes)):
            if i < lookback:
                volume_spikes.append(1.0)
                continue

            avg_volume = sum(volumes[i - lookback : i]) / lookback
            current_volume = volumes[i]
            spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_spikes.append(spike_ratio)

        return volume_spikes

    def _detect_consolidation(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> List[bool]:
        lookback = Settings().get("strategy.consolidation_lookback", 20)

        consolidation = []
        for i in range(len(closes)):
            if i < lookback:
                consolidation.append(False)
                continue

            recent_highs = highs[i - lookback : i]
            recent_lows = lows[i - lookback : i]

            range_high = max(recent_highs)
            range_low = min(recent_lows)
            range_size = range_high - range_low
            avg_price = sum(closes[i - lookback : i]) / lookback

            is_consolidating = (range_size / avg_price) < 0.02 if avg_price else False
            consolidation.append(is_consolidating)

        return consolidation

    def _detect_triangle(
        self, highs: List[float], lows: List[float], closes: List[float], lookback: int = 40
    ) -> Dict[str, Any]:
        if len(closes) < lookback:
            return {"is_contracting": False, "breakout_up": False}
        h = highs[-lookback:]
        l = lows[-lookback:]
        n = len(h)
        x = np.arange(n)
        try:
            m_h, b_h = np.polyfit(x, h, 1)
            m_l, b_l = np.polyfit(x, l, 1)
        except Exception:
            return {"is_contracting": False, "breakout_up": False}
        upper_start = m_h * 0 + b_h
        lower_start = m_l * 0 + b_l
        upper_end = m_h * (n - 1) + b_h
        lower_end = m_l * (n - 1) + b_l
        width_start = max(upper_start - lower_start, 0.0)
        width_end = max(upper_end - lower_end, 0.0)
        is_contracting = (m_h < 0) and (m_l > 0) and (width_end < width_start * 0.8)
        last_close = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else last_close
        breakout_up = False
        if is_contracting:
            upper_prev = m_h * (n - 2) + b_h
            upper_last = upper_end
            try:
                if all(np.isfinite([last_close, prev_close, upper_prev, upper_last])):
                    breakout_up = (prev_close <= upper_prev) and (last_close > upper_last)
            except Exception:
                breakout_up = False
        return {
            "is_contracting": bool(is_contracting),
            "breakout_up": bool(breakout_up),
            "upper_last": float(upper_end) if np.isfinite(upper_end) else None,
            "lower_last": float(lower_end) if np.isfinite(lower_end) else None,
        }

    def _detect_double_top(
        self,
        highs: List[float],
        closes: List[float],
        volumes: List[float],
        lookback: int = 60,
        tolerance: float = 0.01,
    ) -> bool:
        if len(highs) < lookback or len(highs) < 5:
            return False
        h = highs[-lookback:]
        v = volumes[-lookback:]
        peaks = []
        for i in range(1, len(h) - 1):
            if h[i] > h[i - 1] and h[i] > h[i + 1]:
                peaks.append(i)
        if len(peaks) < 2:
            return False
        p1, p2 = peaks[-2], peaks[-1]
        price1, price2 = h[p1], h[p2]
        if not (np.isfinite(price1) and np.isfinite(price2)):
            return False
        if abs(price1 - price2) / max(price1, 1e-6) > tolerance:
            return False
        try:
            if v[p2] > v[p1] * 1.5:
                return False
        except Exception:
            pass
        return True

    def _volume_profile_levels(
        self,
        lows: List[float],
        highs: List[float],
        closes: List[float],
        volumes: List[float],
        lookback: int = 60,
        bins: int = 30,
    ) -> Dict[str, Any]:
        lo = np.array(lows[-lookback:])
        hi = np.array(highs[-lookback:])
        cl = np.array(closes[-lookback:])
        vol = np.array(volumes[-lookback:])
        pr_min = np.nanmin(lo)
        pr_max = np.nanmax(hi)
        if not np.isfinite(pr_min) or not np.isfinite(pr_max) or pr_max <= pr_min:
            return {}
        edges = np.linspace(pr_min, pr_max, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        idx = np.clip(np.digitize(cl, edges) - 1, 0, bins - 1)
        vol_by_bin = np.zeros(bins, dtype=float)
        for i in range(len(idx)):
            if np.isfinite(vol[i]):
                vol_by_bin[idx[i]] += float(vol[i])
        if vol_by_bin.sum() <= 0:
            return {}
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
            "vp_hvn_levels": hvn_levels,
            "vp_lvn_levels": lvn_levels,
            "vp_nearest_hvn": nearest_hvn,
            "vp_nearest_lvn": nearest_lvn,
            "vp_prox_hvn": prox_hvn,
            "vp_prox_lvn": prox_lvn,
        }

    def _calculate_catalyst_freshness(
        self, catalysts: Optional[List[Headline]], current_time: datetime
    ) -> float:
        if not catalysts:
            return 0.0

        recent_catalyst = max(catalysts, key=lambda item: item.timestamp)
        hours_ago = (current_time - recent_catalyst.timestamp).total_seconds() / 3600

        freshness = max(0, 1 - (hours_ago / 24))
        return freshness


__all__ = ["FeatureBuilder"]

