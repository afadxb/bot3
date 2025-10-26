"""Strategy evaluation logic for generating trading signals."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from .config import Settings
from .dtos import Signal


class PropulsionStrategy:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def evaluate(self, features_batch: List[Dict[str, Any]]) -> List[Signal]:
        signals: List[Signal] = []

        for features in features_batch:
            symbol = features.get("symbol")
            feature_data = features.get("features", {})

            if not feature_data:
                continue

            signal = self._evaluate_single(symbol, feature_data)
            if signal:
                signals.append(signal)

        signals.sort(key=lambda sig: sig.final_score, reverse=True)
        for index, signal in enumerate(signals):
            signal.rank = index + 1

        return signals

    def _evaluate_single(self, symbol: str, features: Dict[str, Any]) -> Signal:
        reasons: List[str] = []
        rules_passed: Dict[str, bool] = {}
        base_score = 0.0
        bear_score = 0.0
        penalties: List[float] = []
        settings = Settings()

        long_allowed = True
        short_allowed = True
        if settings.get("strategy.enable_supertrend", False):
            st = features.get("supertrend", {}) or {}
            st_trend = st.get("trend") or []
            if st_trend:
                last_tr = st_trend[-1]
                if last_tr == 1:
                    long_allowed, short_allowed = True, False
                    rules_passed["dir_gate"] = True
                    reasons.append("DirGate=Supertrend Long")
                elif last_tr == -1:
                    long_allowed, short_allowed = False, True
                    rules_passed["dir_gate"] = True
                    reasons.append("DirGate=Supertrend Short")

        ema_fast = features.get("ema_fast", [])
        ema_slow = features.get("ema_slow", [])
        if len(ema_fast) > 1 and len(ema_slow) > 1:
            current_fast = ema_fast[-1]
            current_slow = ema_slow[-1]
            prev_fast = ema_fast[-2]
            prev_slow = ema_slow[-2]

            def _finite(value: Any) -> bool:
                try:
                    return value is not None and np.isfinite(float(value))
                except Exception:
                    return False

            cross_up = False
            cross_down = False
            if all(_finite(val) for val in (current_fast, current_slow, prev_fast, prev_slow)):
                cross_up = (prev_fast <= prev_slow) and (current_fast > current_slow)
                cross_down = (prev_fast >= prev_slow) and (current_fast < current_slow)
                if len(ema_fast) > 2 and len(ema_slow) > 2:
                    f2, s2 = ema_fast[-3], ema_slow[-3]
                    if _finite(f2) and _finite(s2):
                        cross_up = cross_up or ((f2 <= s2) and (prev_fast > prev_slow))
                        cross_down = cross_down or ((f2 >= s2) and (prev_fast < prev_slow))
            rules_passed["ema_cross_up"] = bool(cross_up)
            rules_passed["ema_cross_down"] = bool(cross_down)
            if cross_up and long_allowed:
                base_score += 0.3
                reasons.append("EMA Cross Up")
            if cross_down and short_allowed:
                bear_score += 0.3
                reasons.append("EMA Cross Down")

        ema_bias = features.get("ema_bias", [])
        closes = features.get("closes", [])
        if ema_bias and closes:
            cb = closes[-1]
            eb = ema_bias[-1]
            try:
                if cb is not None and eb is not None and np.isfinite(cb) and np.isfinite(eb):
                    price_above_bias = cb > eb
                    rules_passed["ema_bias"] = price_above_bias
                    if price_above_bias and long_allowed:
                        base_score += 0.2
                        reasons.append("Above EMA Bias")
                    if (not price_above_bias) and short_allowed:
                        bear_score += 0.2
                        reasons.append("Below EMA Bias")
            except Exception:
                pass

        vwap = features.get("vwap", [])
        if vwap and closes:
            cv = closes[-1]
            vv = vwap[-1]
            try:
                if cv is not None and vv is not None and np.isfinite(cv) and np.isfinite(vv):
                    price_above_vwap = cv > vv
                    rules_passed["vwap_alignment"] = price_above_vwap
                    if price_above_vwap and settings.get("strategy.vwap_required", True) and long_allowed:
                        base_score += 0.2
                        reasons.append("Above VWAP")
                    if (not price_above_vwap) and short_allowed:
                        bear_score += 0.2
                        reasons.append("Below VWAP")
            except Exception:
                pass

        volume_spike = features.get("volume_spike", [])
        if volume_spike:
            try:
                vs = float(volume_spike[-1])
                thresh = float(settings.get("strategy.vol_spike_multiple", 1.5))
                if np.isfinite(vs):
                    has_volume_spike = vs > thresh
                    rules_passed["volume_spike"] = has_volume_spike
                    if has_volume_spike:
                        base_score += 0.15
                        reasons.append("Volume Spike")
            except Exception:
                pass

        consolidation = features.get("consolidation", [])
        if len(consolidation) > 1:
            was_consolidating = consolidation[-2]
            broke_out = not consolidation[-1] and was_consolidating
            rules_passed["consolidation_break"] = broke_out
            if broke_out:
                base_score += 0.15
                reasons.append("Consolidation Break")

        catalyst_freshness = features.get("catalyst_freshness", 0)
        try:
            rules_passed["catalyst"] = float(catalyst_freshness) > 0
        except Exception:
            rules_passed["catalyst"] = False

        tri = features.get("pattern_triangle", {}) or {}
        if tri.get("is_contracting") and tri.get("breakout_up"):
            base_score += 0.15
            reasons.append("Triangle Breakout")

        if features.get("pattern_double_top"):
            penalties.append(0.2)
            reasons.append("Double Top Risk")

        atr_vals = features.get("atr", [])
        latest_atr = atr_vals[-1] if atr_vals else None
        try:
            if latest_atr is not None and np.isfinite(latest_atr) and latest_atr > 0 and closes:
                cl = closes[-1]
                prox_lvn = features.get("vp_prox_lvn")
                prox_hvn = features.get("vp_prox_hvn")
                if prox_lvn is not None and np.isfinite(prox_lvn) and prox_lvn <= 0.5 * latest_atr:
                    base_score += 0.1
                    reasons.append("Near LVN")
                nearest_hvn = features.get("vp_nearest_hvn")
                if (
                    nearest_hvn is not None
                    and np.isfinite(nearest_hvn)
                    and nearest_hvn >= cl
                    and (nearest_hvn - cl) <= 0.5 * latest_atr
                ):
                    penalties.append(0.1)
                    reasons.append("HVN Overhead")
        except Exception:
            pass
        if settings.get("strategy.catalyst_required", True):
            if catalyst_freshness > 0:
                base_score += catalyst_freshness * 0.1
                reasons.append(f"Recent Catalyst ({catalyst_freshness:.2f})")
        else:
            base_score += catalyst_freshness * 0.1

        if settings.get("strategy.enable_supertrend", False):
            supertrend = features.get("supertrend", {})
            trend = supertrend.get("trend", [])
            if trend and trend[-1] == 1 and long_allowed:
                base_score += 0.1
                reasons.append("Supertrend Long")
            elif trend and trend[-1] == -1 and short_allowed:
                bear_score += 0.1
                reasons.append("Supertrend Short")

        if long_allowed and not short_allowed:
            direction = "long"
        elif short_allowed and not long_allowed:
            direction = "short"
        else:
            direction = "long" if base_score >= bear_score else "short"
        entry_hint = closes[-1] if closes else None
        stop_hint = None
        if entry_hint and features.get("atr"):
            atr = features["atr"][-1]
            try:
                if atr is not None and np.isfinite(atr) and atr > 0:
                    stop_hint = (
                        entry_hint - (atr * 1.5)
                        if direction == "long"
                        else entry_hint + (atr * 1.5)
                    )
            except Exception:
                stop_hint = None

        total_penalty = sum(penalties)
        chosen_score = max(base_score, bear_score)
        final_score = max(0, min(1, chosen_score - total_penalty))
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
            ai_adj_score=final_score,
            final_score=final_score,
            reasons=reasons,
            rules_passed=rules_passed,
            features=features,
            cycle_id="cycle_" + str(int(time.time())),
        )


__all__ = ["PropulsionStrategy"]

