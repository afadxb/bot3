from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


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
        self.trail = max(self.trail, candidate)


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
        cost = qty * price
        if cost > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= cost
        pos = Position(symbol, qty, price, price, atr, epsilon, atr_lambda)
        self.positions[symbol] = pos
        return pos

    def mark(self, symbol: str, price: float) -> None:
        pos = self.positions[symbol]
        pos.update(price)

    def should_exit(self, symbol: str, price: float) -> bool:
        pos = self.positions[symbol]
        return price <= pos.trail
