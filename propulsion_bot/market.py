"""Market calendar helpers."""

from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta
from typing import Optional

import pytz


class MarketClock:
    def __init__(self, timezone: str = "America/Toronto") -> None:
        self.tz = pytz.timezone(timezone)

    def is_market_hours(self, check_time: Optional[datetime] = None) -> bool:
        if check_time is None:
            check_time = datetime.now(self.tz)

        market_time = check_time.astimezone(self.tz)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)

        return market_time.weekday() < 7 and market_open <= market_time.time() <= market_close

    def time_until_flatten(self, flatten_time: str = "15:55") -> Optional[timedelta]:
        now = datetime.now(self.tz)
        hour, minute = map(int, flatten_time.split(":"))
        flatten_dt = datetime.combine(now.date(), dt_time(hour, minute))
        flatten_dt = self.tz.localize(flatten_dt)

        if now >= flatten_dt:
            return timedelta(0)

        return flatten_dt - now

    def should_flatten(self, flatten_time: str = "15:55") -> bool:
        time_until = self.time_until_flatten(flatten_time)
        return time_until is not None and time_until.total_seconds() <= 0


__all__ = ["MarketClock"]

