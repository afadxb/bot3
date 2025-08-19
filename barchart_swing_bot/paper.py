from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .config import get_settings
from . import notifier


settings = get_settings()


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    high_since_entry: float
    atr: float
    epsilon: float
    atr_lambda: float
    trail: float = field(init=False)

    def __post_init__(self) -> None:
        self.trail = self.entry_price + self.epsilon

    def update(self, current_price: float) -> None:
        if current_price > self.high_since_entry:
            self.high_since_entry = current_price
        candidate = self.high_since_entry - self.atr_lambda * self.atr
        new_trail = max(self.trail, candidate)
        if new_trail > self.trail:
            notifier.send("trail raised", f"{self.symbol} {new_trail:.2f}")
        self.trail = new_trail


class PaperBroker:
    """Very small paper trading simulator."""

    def __init__(self, balance: float) -> None:
        self.balance = balance
        self.positions: Dict[str, Position] = {}

    def buy(
        self,
        symbol: str,
        qty: int,
        price: float,
        atr: float,
        epsilon: float,
        atr_lambda: float,
    ) -> Position:
        notifier.send("order sent", f"BUY {qty} {symbol} @ {price}")
        cost = qty * price
        if cost > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= cost
        pos = Position(symbol, qty, price, price, atr, epsilon, atr_lambda)
        self.positions[symbol] = pos
        notifier.send("order filled", f"BUY {qty} {symbol} @ {price}")
        return pos

    def mark(self, symbol: str, price: float) -> None:
        pos = self.positions[symbol]
        pos.update(price)

    def should_exit(self, symbol: str, price: float) -> bool:
        pos = self.positions[symbol]
        return price <= pos.trail

    def sell(self, symbol: str, price: float) -> None:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * price
        self.balance += proceeds
        notifier.send("exit executed", f"SELL {pos.qty} {symbol} @ {price}")

    def check_risk(self) -> None:
        start = settings.paper_start_balance
        loss_pct = (start - self.balance) / start * 100
        if loss_pct >= settings.max_daily_loss_pct:
            notifier.send("risk warning", f"loss {loss_pct:.2f}%")
