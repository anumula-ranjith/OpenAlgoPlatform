"""Rolling support/resistance signals and long-call policy. No order APIs."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "NIFTY01DEC2625350CE"
QUANTITY = 65
LOOKBACK = 20
STOP = 0.15
TARGET = 0.30


def breakout_signals(bars, now):
    """Levels use the previous 20 completed same-session candles, never the signal bar."""
    now = pd.Timestamp(now)
    if now.tzinfo is None or not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("Timezone-aware historical and evaluation timestamps are required.")
    frame = bars.sort_index().copy()
    frame.index = frame.index.tz_convert(IST)
    now = now.tz_convert(IST)
    frame = frame.loc[
        (frame.index.date == now.date())
        & (frame.index.strftime("%H:%M") >= "09:15")
        & (frame.index + pd.Timedelta(minutes=5) <= now)
    ]
    if frame.index.has_duplicates:
        raise ValueError("Duplicate index candles are not allowed.")
    frame["resistance"] = frame.high.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).max()
    frame["support"] = frame.low.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).min()
    closed = frame.index + pd.Timedelta(minutes=5)
    frame["breakout"] = (
        (frame.close > frame.resistance)
        & (closed.strftime("%H:%M") >= "11:00")
        & (closed.strftime("%H:%M") < "14:00")
    )
    frame["breakdown"] = frame.close < frame.support
    return frame


@dataclass
class Policy:
    quantity: int = 0
    entry_price: float = 0
    entries: int = 0
    last_signal: str = ""
    support: float = 0
    resistance: float = 0

    def decide(self, signal, premium, now):
        if now.tzinfo is None or not isfinite(premium) or premium <= 0:
            raise ValueError("A timezone and a finite positive premium are required.")
        now = now.astimezone(IST)
        clock = now.strftime("%H:%M")
        if self.quantity not in (0, QUANTITY):
            raise ValueError("Unexpected modeled quantity.")
        if self.quantity:
            if clock >= "15:05":
                return "SELL", "time_exit"
            if premium <= self.entry_price * (1 - STOP):
                return "SELL", "premium_stop"
            if premium >= self.entry_price * (1 + TARGET):
                return "SELL", "premium_target"
        closed = datetime.fromisoformat(signal["known_at_ist"])
        if closed.tzinfo is None:
            raise ValueError("The signal completion timestamp must include a timezone.")
        if self.last_signal == signal["known_at_ist"]:
            return "WAIT", "already_processed_signal"
        self.last_signal = signal["known_at_ist"]
        if (
            closed.astimezone(IST).date() != now.date()
            or not 0 <= (now - closed).total_seconds() < 300
        ):
            return "WAIT", "stale_or_future_signal"
        if self.quantity and signal["close"] < self.support:
            return "SELL", "frozen_support_break"
        if (
            not self.quantity
            and self.entries == 0
            and "11:00" <= clock < "14:00"
            and signal["breakout"]
        ):
            return "BUY", "resistance_breakout"
        return "WAIT", "no_entry_or_exit"

    def filled(self, action, price, support=None, resistance=None):
        if not isfinite(price) or price <= 0:
            raise ValueError("Fill must be finite and positive.")
        if action == "BUY" and self.quantity == 0 and self.entries == 0:
            if (
                support is None
                or resistance is None
                or not (isfinite(support) and isfinite(resistance) and 0 < support <= resistance)
            ):
                raise ValueError("A valid entry support/resistance range is required.")
            self.support, self.resistance = support, resistance
            self.quantity, self.entry_price, self.entries = QUANTITY, price, 1
        elif action == "SELL" and self.quantity == QUANTITY:
            self.quantity, self.entry_price = 0, 0
        else:
            raise ValueError("Invalid modeled fill transition.")
