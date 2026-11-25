"""Offline data checks, modeled fills and accounting for the fixed NIFTY call.

No account access or order submission. Brokerage/statutory arithmetic delegates
only to the platform's pure charge engine.
"""

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR
from decimal import Decimal as D

import numpy as np
import pandas as pd

from sandbox.charges import compute_charges, paisa
from strategies.examples.nifty_sr_paper import IST, QUANTITY, SYMBOL


@dataclass(frozen=True)
class Settings:
    option_minutes: int = 5
    slippage_bps: str = "10"
    brokerage_per_order: str = "20"


def validate_bars(frame, minutes):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Historical timestamps must include a timezone.")
    frame = frame.sort_index().copy()
    frame.index = frame.index.tz_convert(IST)
    if frame.empty or frame.index.has_duplicates:
        raise ValueError("Empty or duplicate historical candles.")
    columns = ["open", "high", "low", "close", "volume"]
    if not set(columns).issubset(frame.columns):
        raise ValueError("OHLC and volume are required.")
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if (
        not np.isfinite(frame[columns]).all().all()
        or (frame[columns[:4]] <= 0).any().any()
        or (frame.volume < 0).any()
    ):
        raise ValueError("Invalid historical price or volume.")
    if (frame.high < frame[["open", "low", "close"]].max(axis=1)).any() or (
        frame.low > frame[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC prices.")
    if any(t.minute % minutes or t.second or t.microsecond for t in frame.index):
        raise ValueError("Timestamps must align with the selected candle interval.")
    return frame


def fill(raw, side, settings):
    multiplier = D(1) + D(settings.slippage_bps) / 10000 * (1 if side == "BUY" else -1)
    return (D(str(raw)) * multiplier / D("0.05")).to_integral_value(
        rounding=ROUND_CEILING if side == "BUY" else ROUND_FLOOR
    ) * D("0.05")


def charges(price, side, settings):
    return compute_charges(
        "fno_options", "NFO", side, price * QUANTITY, D(settings.brokerage_per_order), D(0)
    )


def summary(trades):
    net_values = [D(t["net_pnl"]) for t in trades]
    gross = sum((D(t["gross_pnl"]) for t in trades), D(0))
    fees = sum((D(t["total_charges"]) for t in trades), D(0))
    wins, losses = sum(n > 0 for n in net_values), sum(n < 0 for n in net_values)
    equity = peak = max_drawdown = D(0)
    for value in net_values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(trades),
        "wins_after_charges": wins,
        "losses_after_charges": losses,
        "breakeven_after_charges": len(trades) - wins - losses,
        "win_rate_percent": round(100 * wins / len(trades), 2) if trades else None,
        "gross_pnl": str(paisa(gross)),
        "total_charges": str(paisa(fees)),
        "net_pnl": str(paisa(gross - fees)),
        "closed_trade_max_drawdown": str(paisa(max_drawdown)),
    }


def read_inputs(folder, option_minutes):
    manifest = json.loads((folder / "download_manifest.json").read_text(encoding="utf-8"))
    frames = []
    for name, expected, interval in [
        ("underlying_5m.csv", "NIFTY", "5m"),
        (f"option_{option_minutes}m.csv", SYMBOL, f"{option_minutes}m"),
    ]:
        path = folder / name
        meta = manifest["files"].get(name, {})
        if (
            meta.get("symbol") != expected
            or meta.get("interval") != interval
            or meta.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Historical instrument, interval or hash does not match its manifest.")
        frames.append(pd.read_csv(path, index_col=0, parse_dates=True))
    return *frames, manifest
