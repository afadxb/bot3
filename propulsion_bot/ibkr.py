"""Interactive Brokers client helpers."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

from .database import Database
from .dtos import BarData, Headline

logger = logging.getLogger("propulsion_bot.ibkr")


class PropulsionBot(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self._req_id = 0
        self._historical_data: Dict[int, List[Dict[str, Any]]] = {}

    def reqHistoricalData(
        self,
        contract: Contract,
        duration: str = "1 D",
        barSize: str = "5 mins",
        whatToShow: str = "TRADES",
    ) -> int:
        self._req_id += 1
        req_id = self._req_id
        end_time = datetime.now(pytz.timezone("US/Eastern")).strftime("%Y%m%d %H:%M:%S EST")

        super().reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=end_time,
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow=whatToShow,
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )

        return req_id

    def historicalData(self, reqId: int, bar) -> None:  # noqa: N802 - IBKR callback
        if reqId not in self._historical_data:
            self._historical_data[reqId] = []

        self._historical_data[reqId].append(
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "wap": bar.wap,
                "count": bar.barCount,
            }
        )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        if reqId in self._historical_data:
            self._historical_data[reqId].append({"completed": True})


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
        self._match_events: Dict[int, threading.Event] = {}
        self._matches: Dict[int, Any] = {}
        self._quotes: Dict[int, Dict[str, float]] = {}
        self._quote_events: Dict[int, threading.Event] = {}
        self._news: Dict[int, List[Dict[str, Any]]] = {}
        self._news_events: Dict[int, threading.Event] = {}
        self._account_values: Dict[Tuple[str, str], str] = {}
        self._account_event: threading.Event = threading.Event()
        self._positions: List[Dict[str, Any]] = []
        self._position_event: threading.Event = threading.Event()
        self.db: Optional[Database] = None
        self._commissions: Dict[str, float] = {}
        self._open_orders: List[Dict[str, Any]] = []
        self._open_orders_event: threading.Event = threading.Event()

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self._next_id = orderId

    def contractDetails(self, reqId: int, contractDetails) -> None:  # noqa: N802
        self._contracts[reqId] = contractDetails.contract
        try:
            symbol = self._req_symbol_map.get(reqId)
            if symbol:
                min_tick = getattr(contractDetails, "minTick", None)
                if min_tick:
                    self._min_tick_by_symbol[symbol] = float(min_tick)
        except Exception:
            pass

    def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
        event = self._contract_events.get(reqId)
        if event:
            event.set()
        self._req_symbol_map.pop(reqId, None)

    def symbolSamples(self, reqId: int, details) -> None:  # noqa: N802
        self._matches[reqId] = details
        event = self._match_events.get(reqId)
        if event:
            event.set()

    def historicalData(self, reqId: int, bar) -> None:  # noqa: N802
        self._bars.setdefault(reqId, []).append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        event = self._bar_events.get(reqId)
        if event:
            event.set()

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:  # noqa: N802
        quote = self._quotes.setdefault(reqId, {"bid": None, "ask": None, "last": None})
        if tickType == 1:
            quote["bid"] = price
        elif tickType == 2:
            quote["ask"] = price
        elif tickType == 4:
            quote["last"] = price

    def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
        event = self._quote_events.get(reqId)
        if event:
            event.set()

    def historicalNews(self, requestId: int, time_: str, providerCode: str, articleId: str, headline: str) -> None:  # noqa: N802,E501
        self._news.setdefault(requestId, []).append({"time": time_, "headline": headline})

    def historicalNewsEnd(self, requestId: int, hasMore: bool) -> None:  # noqa: N802
        event = self._news_events.get(requestId)
        if event:
            event.set()

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str) -> None:  # noqa: N802
        self._account_values[(key, currency)] = val

    def accountDownloadEnd(self, accountName: str) -> None:  # noqa: N802
        self._account_event.set()

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:  # noqa: N802,E501
        self._positions.append(
            {
                "account": account,
                "symbol": getattr(contract, "symbol", ""),
                "conId": getattr(contract, "conId", 0),
                "position": float(position),
                "avgCost": float(avgCost),
            }
        )
        try:
            if self.db is not None and getattr(contract, "symbol", ""):
                self.db.upsert_position(getattr(contract, "symbol", ""), float(position), float(avgCost))
        except Exception:
            pass

    def positionEnd(self) -> None:  # noqa: N802
        self._position_event.set()

    def execDetails(self, reqId: int, contract: Contract, execution) -> None:  # noqa: N802
        try:
            if self.db is not None:
                self.db.insert_execution(
                    order_ref=getattr(execution, "orderRef", ""),
                    symbol=getattr(contract, "symbol", ""),
                    side=getattr(execution, "side", ""),
                    qty=float(getattr(execution, "shares", 0) or 0),
                    price=float(getattr(execution, "price", 0) or 0),
                    exec_id=getattr(execution, "execId", ""),
                    exchange=getattr(execution, "exchange", ""),
                    commission=self._commissions.get(getattr(execution, "execId", ""), None),
                    realized_pnl=None,
                )
        except Exception:
            pass

    def commissionReport(self, commissionReport) -> None:  # noqa: N802
        try:
            exec_id = getattr(commissionReport, "execId", "")
            commission = float(getattr(commissionReport, "commission", 0) or 0)
            if exec_id:
                self._commissions[exec_id] = commission
        except Exception:
            pass

    def openOrder(self, orderId: int, contract: Contract, order, orderState) -> None:  # noqa: N802,E501
        try:
            self._open_orders.append(
                {
                    "orderId": int(orderId),
                    "symbol": getattr(contract, "symbol", ""),
                    "action": getattr(order, "action", ""),
                    "totalQuantity": int(getattr(order, "totalQuantity", 0) or 0),
                    "orderType": getattr(order, "orderType", ""),
                    "status": getattr(orderState, "status", ""),
                    "orderRef": getattr(order, "orderRef", ""),
                }
            )
        except Exception:
            pass

    def openOrderEnd(self) -> None:  # noqa: N802
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
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.primaryExchange = "NASDAQ"
        return contract

    def resolve(self, symbol: str) -> Contract:
        exchanges = [None, "NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "ISLAND"]
        for exchange_name in exchanges:
            req_id = self._next_id
            self._next_id += 1
            self._contract_events[req_id] = threading.Event()
            base = self._stock(symbol)
            if exchange_name:
                base.primaryExchange = exchange_name
            self._req_symbol_map[req_id] = symbol
            self.reqContractDetails(req_id, base)
            self._contract_events[req_id].wait(timeout=5)
            self._contract_events.pop(req_id, None)
            contract = self._contracts.pop(req_id, None)
            if contract and getattr(contract, "conId", 0):
                return contract

        try:
            req_id = self._next_id
            self._next_id += 1
            self._match_events[req_id] = threading.Event()
            self.reqMatchingSymbols(req_id, symbol)
            self._match_events[req_id].wait(timeout=5)
            self._match_events.pop(req_id, None)
            matches = self._matches.pop(req_id, [])
            for match in matches:
                try:
                    desc = match.contract
                    if getattr(desc, "secType", "") != "STK":
                        continue
                    if getattr(desc, "currency", "") and desc.currency != "USD":
                        continue
                    req_id2 = self._next_id
                    self._next_id += 1
                    self._contract_events[req_id2] = threading.Event()
                    contract = Contract()
                    contract.symbol = desc.symbol
                    contract.secType = "STK"
                    contract.currency = desc.currency or "USD"
                    contract.exchange = "SMART"
                    if getattr(desc, "primaryExchange", None):
                        contract.primaryExchange = desc.primaryExchange
                    self._req_symbol_map[req_id2] = symbol
                    self.reqContractDetails(req_id2, contract)
                    self._contract_events[req_id2].wait(timeout=5)
                    self._contract_events.pop(req_id2, None)
                    resolved = self._contracts.pop(req_id2, None)
                    if resolved and getattr(resolved, "conId", 0):
                        return resolved
                except Exception:
                    continue
        except Exception:
            pass

        logger.error("Failed to resolve contract for %s on all exchanges", symbol)
        return self._stock(symbol)

    def get_min_tick(self, symbol: str, default: float = 0.01) -> float:
        min_tick = self._min_tick_by_symbol.get(symbol)
        try:
            if min_tick is None:
                _ = self.resolve(symbol)
                min_tick = self._min_tick_by_symbol.get(symbol)
        except Exception:
            pass
        try:
            return float(min_tick) if min_tick is not None and min_tick > 0 else float(default)
        except Exception:
            return float(default)

    def fetch_bars_df(
        self,
        symbol: str,
        bar_size: str,
        duration: str,
        whatToShow: str = "TRADES",
        useRTH: bool = True,
        formatDate: int = 1,
        endDateTime: Optional[datetime] = None,
    ) -> pd.DataFrame:
        try:
            time.sleep(0.5)
            contract = self.resolve(symbol)
            queryTime = ""
            if endDateTime is not None:
                try:
                    dt_utc = endDateTime if endDateTime.tzinfo is None else endDateTime.astimezone(timezone.utc)
                    queryTime = dt_utc.strftime("%Y%m%d %H:%M:%S")
                except Exception:
                    queryTime = ""

            def _do_req(end_dt: Optional[datetime], wts: str, rth: bool) -> List[Dict[str, Any]]:
                qtime = ""
                if end_dt is not None:
                    try:
                        dt_utc = end_dt if end_dt.tzinfo is None else end_dt.astimezone(timezone.utc)
                        qtime = dt_utc.strftime("%Y%m%d-%H:%M:%S")
                    except Exception:
                        qtime = ""
                rid = self._next_id
                self._next_id += 1
                self._bars[rid] = []
                self._bar_events[rid] = threading.Event()
                self.reqHistoricalData(rid, contract, qtime, duration, bar_size, wts, 1 if rth else 0, formatDate, False, [])
                self._bar_events[rid].wait(timeout=30)
                out: List[Dict[str, Any]] = []
                for bar in self._bars.get(rid, []):
                    open_price = getattr(bar, "open", getattr(bar, "openPrice", None))
                    close_price = getattr(bar, "close", getattr(bar, "closePrice", None))
                    high_price = getattr(bar, "high", None)
                    low_price = getattr(bar, "low", None)
                    volume = getattr(bar, "volume", None)
                    ts = getattr(bar, "date", None)
                    if None in (open_price, high_price, low_price, close_price, volume, ts):
                        continue
                    out.append(
                        {
                            "ts": pd.to_datetime(ts),
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                        }
                    )
                self._bars.pop(rid, None)
                self._bar_events.pop(rid, None)
                return out

            attempts: List[Tuple[Optional[datetime], str, bool]] = []
            attempts.append((endDateTime, whatToShow, useRTH))
            if whatToShow == "TRADES" and useRTH:
                attempts.append((endDateTime, "TRADES", False))
            attempts.append((endDateTime, "MIDPOINT", useRTH))
            attempts.append((endDateTime, "MIDPOINT", False))
            if endDateTime is None:
                now_utc = datetime.utcnow()
                attempts.extend(
                    [
                        (now_utc, whatToShow, useRTH),
                        (now_utc, "TRADES", False),
                        (now_utc, "MIDPOINT", useRTH),
                        (now_utc, "MIDPOINT", False),
                        (now_utc - timedelta(days=1), whatToShow, useRTH),
                        (now_utc - timedelta(days=1), "MIDPOINT", False),
                    ]
                )

            for end_dt, wts, rth in attempts:
                rows = _do_req(end_dt, wts, rth)
                if rows:
                    logger.debug(
                        "Historical bars resolved for %s with wts=%s, useRTH=%s, end=%s",
                        symbol,
                        wts,
                        rth,
                        end_dt,
                    )
                    return pd.DataFrame(rows)

            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        except Exception as exc:
            logger.exception("Error fetching %s: %s", symbol, exc)
            return pd.DataFrame()

    def fetch_quote_snapshot(self, symbol: str) -> Dict[str, float]:
        contract = self.resolve(symbol)
        req_id = self._next_id
        self._next_id += 1
        self._quotes[req_id] = {"bid": None, "ask": None, "last": None}
        self._quote_events[req_id] = threading.Event()
        self.reqMktData(req_id, contract, "", True, False, [])
        self._quote_events[req_id].wait(timeout=10)
        quote = self._quotes.get(req_id, {})
        self._quotes.pop(req_id, None)
        self._quote_events.pop(req_id, None)
        return quote

    def fetch_headlines(
        self, symbol: str, provider_code: str = "BRFG", lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        contract = self.resolve(symbol)
        con_id = getattr(contract, "conId", None)
        if not con_id:
            return []
        req_id = self._next_id
        self._next_id += 1
        self._news[req_id] = []
        self._news_events[req_id] = threading.Event()
        start = (datetime.utcnow() - timedelta(hours=lookback_hours)).strftime("%Y%m%d-%H:%M:%S")
        end = ""
        total = 10
        self.reqHistoricalNews(req_id, con_id, provider_code, start, end, total, [])
        self._news_events[req_id].wait(timeout=10)
        headlines = self._news.get(req_id, [])
        self._news.pop(req_id, None)
        self._news_events.pop(req_id, None)
        return headlines

    def get_equity(self, base_currency: str = "USD", timeout: float = 10.0) -> Optional[float]:
        self._account_event.clear()
        try:
            self.reqAccountUpdates(True, "")
        except Exception:
            return None
        self._account_event.wait(timeout=timeout)
        val = self._account_values.get(("NetLiquidation", base_currency)) or self._account_values.get(
            ("TotalCashValue", base_currency)
        )
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

    def fetch_data_bundle(
        self,
        symbols: List[str],
        bar_size: str = "5 mins",
        duration: str = "14400 S",
    ) -> Tuple[Dict[str, List[BarData]], Dict[str, Dict[str, float]], List[Headline]]:
        try:
            bars_data: Dict[str, List[BarData]] = {}
            quotes_map: Dict[str, Dict[str, float]] = {}
            headlines: List[Headline] = []

            for sym in symbols:
                try:
                    cached = self.db.get_bars_from_cache(sym, "5min", 240) if self.db else []
                    rows: List[BarData] = cached

                    if not rows:
                        df = self.fetch_bars_df(sym, bar_size=bar_size, duration=duration)
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
                            if self.db:
                                self.db.upsert_bars_cache(sym, "5min", rows)
                        else:
                            logger.warning("No data returned for %s", sym)
                            continue

                    if rows:
                        bars_data[sym] = rows

                except Exception as exc:
                    logger.error("Error processing %s: %s", sym, exc)
                    continue

            return bars_data, quotes_map, headlines
        except Exception as exc:
            logger.error("Fatal error in data fetch: %s", exc)
            return {}, {}, []


__all__ = ["PropulsionBot", "IBKRLiveFeed"]

