"""Order planning and execution helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .config import Settings
from .database import Database
from .dtos import PlannedOrder, Signal

if TYPE_CHECKING:
    from .ibkr import IBKRLiveFeed

logger = logging.getLogger("propulsion_bot.trade")


class TradeManager:
    def __init__(self, config, db: Database, ib: Optional["IBKRLiveFeed"] = None) -> None:
        self.config = config
        self.db = db
        self.ib = ib
        self.dry_run = Settings().get("execution.enable_orders", False) is False
        self._submitted_symbols: set[Tuple[str, str]] = set()

    def plan_orders(self, signals: List[Signal]) -> List[PlannedOrder]:
        planned_orders: List[PlannedOrder] = []
        settings = Settings()

        for signal in signals:
            if signal.entry_hint is None or signal.stop_hint is None:
                continue
            try:
                entry = float(signal.entry_hint)
                stop = float(signal.stop_hint)
            except Exception:
                continue
            if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0 or stop <= 0:
                continue

            equity = self.db.get_current_equity() or 100000
            risk_per_trade_pct = settings.get("risk.risk_per_trade_pct", 1.0)
            dollar_risk = equity * risk_per_trade_pct / 100

            raw_risk = abs(entry - stop)
            min_tick = float(settings.get("risk.min_tick_buffer", 0.01))
            per_share_risk = raw_risk if (np.isfinite(raw_risk) and raw_risk > 0) else min_tick
            per_share_risk = max(per_share_risk, min_tick)

            qty = int(max(0, dollar_risk / per_share_risk))

            max_position_pct = settings.get("risk.max_position_value_pct", 20.0)
            max_position_value = equity * max_position_pct / 100
            max_qty_by_value = int(max_position_value / entry) if entry > 0 else 0

            qty = min(qty, max_qty_by_value)

            if qty <= 0:
                continue

            risk_distance = per_share_risk
            scale_out_r = settings.get("execution.scale_out_at_r_multiple", 1.0)
            final_target_r = settings.get("execution.final_target_r_multiple", 2.0)
            if getattr(signal, "direction", "long") == "short":
                scale_out_price = entry - (risk_distance * scale_out_r)
                final_target_price = entry - (risk_distance * final_target_r)
            else:
                scale_out_price = entry + (risk_distance * scale_out_r)
                final_target_price = entry + (risk_distance * final_target_r)

            atr_vals = signal.features.get("atr", []) if isinstance(signal.features, dict) else []
            atr_latest = atr_vals[-1] if atr_vals else None
            atr_mult = float(settings.get("execution.atr_trail_mult", 2.0))
            lmt_offset_pct = float(settings.get("execution.lmt_offset_pct", 0.02))
            latest_price = signal.features.get("latest_price") if isinstance(signal.features, dict) else None

            side = "SELL" if getattr(signal, "direction", "long") == "short" else "BUY"
            planned_order = PlannedOrder(
                symbol=signal.symbol,
                side=side,
                qty=qty,
                entry=entry,
                stop=stop,
                scale_out=scale_out_price,
                target=final_target_price,
                trail_mode=settings.get("execution.trail_mode", "ema21"),
                risk_context={
                    "dollar_risk": dollar_risk,
                    "per_share_risk": per_share_risk,
                    "r_multiple_scale": scale_out_r,
                    "r_multiple_final": final_target_r,
                    "atr": float(atr_latest) if (atr_latest is not None and np.isfinite(atr_latest)) else None,
                    "atr_mult": atr_mult,
                    "lmt_offset_pct": lmt_offset_pct,
                    "latest_price": float(latest_price) if (latest_price is not None and np.isfinite(latest_price)) else entry,
                },
                reasons=signal.reasons if hasattr(signal, "reasons") else None,
            )
            planned_orders.append(planned_order)

        return planned_orders

    def execute(self, orders: List[PlannedOrder]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        existing_by_side: Dict[str, set] = {}
        placed_keys: set[Tuple[str, str]] = set()
        held_pos: Dict[str, float] = {}

        if self.ib is not None:
            try:
                open_orders = self.ib.get_open_orders()
                for order in open_orders:
                    sym = order.get("symbol")
                    status = (order.get("status") or "").lower()
                    action = (order.get("action") or "").upper()
                    if sym and status not in ("filled", "cancelled", "inactive") and action in ("BUY", "SELL"):
                        existing_by_side.setdefault(sym, set()).add(action)
            except Exception:
                pass
            try:
                positions = self.ib.get_positions()
                for position in positions:
                    sym = position.get("symbol")
                    qty = float(position.get("position", 0) or 0)
                    if sym and qty != 0:
                        held_pos[sym] = qty
            except Exception:
                pass

        for order in orders:
            try:
                logger.info(
                    "Planned %s %s @ %s tgt %s stop %s", order.side, order.symbol, order.entry, order.target, order.stop
                )
            except Exception:
                pass
            key = (order.symbol, (order.side or "").upper())
            if (
                key in placed_keys
                or key in self._submitted_symbols
                or (order.symbol in existing_by_side and (order.side or "").upper() in existing_by_side[order.symbol])
            ):
                results.append({"symbol": order.symbol, "side": order.side, "status": "SKIPPED_DUPLICATE"})
                continue
            existing_qty = held_pos.get(order.symbol, 0.0)
            if existing_qty != 0:
                planned_sign = 1 if (order.side or "").upper() == "BUY" else -1
                existing_sign = 1 if existing_qty > 0 else -1
                if existing_sign == planned_sign:
                    try:
                        equity = self.db.get_current_equity() or 100000
                        max_pct = Settings().get("risk.max_position_value_pct", 20.0)
                        cap_value = equity * max_pct / 100.0
                        current_value = abs(existing_qty) * float(order.entry)
                        remaining_value = max(0.0, cap_value - current_value)
                        max_add_qty = int(remaining_value / max(0.01, float(order.entry)))
                        add_qty = min(int(order.qty), max_add_qty)
                        if add_qty <= 0:
                            results.append({"symbol": order.symbol, "side": order.side, "status": "SKIPPED_HELD_SAME_DIRECTION"})
                            continue
                        else:
                            order.qty = add_qty
                            results.append({"symbol": order.symbol, "side": order.side, "status": "SCALE_IN_PLANNED", "qty": add_qty})
                    except Exception:
                        results.append({"symbol": order.symbol, "side": order.side, "status": "SKIPPED_HELD_SAME_DIRECTION"})
                        continue
            if self.dry_run:
                result = {
                    "symbol": order.symbol,
                    "status": "DRY_RUN",
                    "qty": order.qty,
                    "entry": order.entry,
                    "stop": order.stop,
                    "message": "Dry run - no order sent",
                }
            else:
                try:
                    ib = self.ib
                    if ib is None:
                        raise RuntimeError("IBKR feed is not initialized")
                    contract = ib.resolve(order.symbol)
                    parent_id = ib._next_id
                    ib._next_id += 1
                    from ibapi.order import Order

                    parent = Order()
                    parent.orderId = parent_id
                    parent.action = order.side
                    parent.orderType = "LMT"
                    parent.totalQuantity = int(max(0, int(order.qty)))

                    def _round_to_inc(px: float, inc: float = 0.05) -> float:
                        try:
                            return round(round(float(px) / inc) * inc, 2)
                        except Exception:
                            return float(px)

                    parent.lmtPrice = _round_to_inc(order.entry)
                    parent.tif = "DAY"
                    parent.outsideRth = False
                    try:
                        parent.eTradeOnly = False
                        parent.firmQuoteOnly = False
                    except Exception:
                        pass
                    parent.orderRef = order.symbol + ":" + (order.side or "") + ":" + str(int(time.time()))
                    parent.transmit = False

                    tp_id = ib._next_id
                    ib._next_id += 1
                    take_profit = Order()
                    take_profit.orderId = tp_id
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

                    sl_id = ib._next_id
                    ib._next_id += 1
                    stop_loss = Order()
                    stop_loss.orderId = sl_id
                    stop_loss.action = "SELL" if order.side == "BUY" else "BUY"
                    stop_loss.orderType = "TRAIL LIMIT"
                    stop_loss.totalQuantity = parent.totalQuantity
                    entry_px = float(order.entry)
                    stop_init = float(order.stop)
                    rc = order.risk_context or {}
                    atr_val = rc.get("atr")
                    atr_mult = rc.get("atr_mult", 2.0)
                    lmt_offset_pct = rc.get("lmt_offset_pct", 0.02)
                    ref_px = rc.get("latest_price", entry_px)
                    trailing_amt_raw = (
                        float(atr_val) * float(atr_mult)
                        if (atr_val is not None)
                        else max(0.05, abs(entry_px - stop_init))
                    )
                    limit_offset_raw = float(ref_px) * float(lmt_offset_pct)
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
                    ib.placeOrder(parent.orderId, contract, parent)
                    ib.placeOrder(take_profit.orderId, contract, take_profit)
                    ib.placeOrder(stop_loss.orderId, contract, stop_loss)
                    result = {
                        "symbol": order.symbol,
                        "status": "SUBMITTED",
                        "qty": order.qty,
                        "orderRef": parent.orderRef,
                    }
                    try:
                        self.db.insert_trade_plan(
                            order.symbol,
                            order.side,
                            order.qty,
                            order.entry,
                            order.stop,
                            order.target,
                            parent.orderRef,
                            order.reasons,
                        )
                    except Exception:
                        pass
                    placed_keys.add(key)
                    self._submitted_symbols.add(key)
                except Exception as exc:
                    result = {"symbol": order.symbol, "status": "ERROR", "error": str(exc)}

            results.append(result)

            if result["status"] == "DRY_RUN":
                self.db.insert_risk_event(
                    "TRADE_OPENED_DRY_RUN",
                    f"session_{time.strftime('%Y%m%d')}",
                    symbol=order.symbol,
                    value=order.qty * order.entry,
                    meta=result,
                )
            elif result["status"] == "SUBMITTED":
                try:
                    self.db.insert_risk_event(
                        "TRADE_OPENED",
                        f"session_{time.strftime('%Y%m%d')}",
                        symbol=order.symbol,
                        value=order.qty * order.entry,
                        meta={"qty": order.qty},
                    )
                except Exception:
                    pass

        return results


__all__ = ["TradeManager"]

