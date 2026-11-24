"""First NIFTY strategy: five-minute EMA 5/20 signals, with no order submission.

Run --mode replay for historical signals or --mode check for readiness.
The OpenAlgo host injects OPENALGO_API_KEY. For a local terminal, --local-key
uses the existing encrypted OpenAlgo key in memory without printing it.
NIFTY/NSE_INDEX is the index, not an executable futures/options contract.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "app.py").is_file())
SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "5m"
BAR_LENGTH = pd.Timedelta(minutes=5)
FAST_PERIOD = 5
SLOW_PERIOD = 20


class StrategyError(RuntimeError):
    """A safe, actionable error message containing no credentials."""


def require_analyzer(response):
    if not isinstance(response, dict) or response.get("status") != "success":
        raise StrategyError(
            "Could not verify Analyzer mode. Check the broker session and OpenAlgo API key."
        )
    data = response.get("data")
    if not isinstance(data, dict) or data.get("analyze_mode") is not True:
        raise StrategyError(
            "Analyzer mode is disabled. Switch the platform to Analyze before running this example."
        )


def indicators(bars, now):
    """Calculate only from completed, chronological, timezone-aware candles."""
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise StrategyError("Historical candles are unavailable. Check broker history access.")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise StrategyError("Five-minute candles must have timezone-aware timestamps.")
    if not {"open", "high", "low", "close"}.issubset(bars.columns):
        raise StrategyError("Historical data is missing OHLC columns.")
    current_time = pd.Timestamp(now)
    if current_time.tzinfo is None:
        raise StrategyError("The evaluation time must include a timezone.")
    frame = bars.sort_index().copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[frame.index + BAR_LENGTH <= current_time]
    if len(frame) < SLOW_PERIOD + 1:
        raise StrategyError(
            "At least 21 completed candles are needed for the EMA crossover example."
        )
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all() or (frame[column] <= 0).any():
            raise StrategyError("Historical candles contain invalid prices.")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise StrategyError("A candle high is inconsistent with its OHLC prices.")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise StrategyError("A candle low is inconsistent with its OHLC prices.")
    frame["fast_ema"] = (
        frame["close"].ewm(span=FAST_PERIOD, adjust=False, min_periods=FAST_PERIOD).mean()
    )
    frame["slow_ema"] = (
        frame["close"].ewm(span=SLOW_PERIOD, adjust=False, min_periods=SLOW_PERIOD).mean()
    )
    previous_fast = frame["fast_ema"].shift(1)
    previous_slow = frame["slow_ema"].shift(1)
    frame["bullish_cross"] = (previous_fast <= previous_slow) & (
        frame["fast_ema"] > frame["slow_ema"]
    )
    frame["bearish_cross"] = (previous_fast >= previous_slow) & (
        frame["fast_ema"] < frame["slow_ema"]
    )
    frame["signal"] = np.select(
        [frame["bullish_cross"], frame["bearish_cross"]],
        ["BULLISH_CROSS", "BEARISH_CROSS"],
        default="WAIT",
    )
    return frame


def snapshot(frame):
    latest = frame.iloc[-1]
    return {
        "candle_open_ist": frame.index[-1].tz_convert(IST).isoformat(),
        "candle_close_ist": (frame.index[-1] + BAR_LENGTH).tz_convert(IST).isoformat(),
        "close": round(float(latest["close"]), 2),
        "ema_5": round(float(latest["fast_ema"]), 2),
        "ema_20": round(float(latest["slow_ema"]), 2),
        "signal": latest["signal"],
        "orders_submitted": 0,
    }


@dataclass
class CandleTracker:
    last_seen: object = None

    def observe(self, frame):
        timestamp = frame.index[-1]
        if self.last_seen is not None and timestamp <= self.last_seen:
            return None
        self.last_seen = timestamp
        return snapshot(frame)


def load_key(use_local_key):
    existing = os.environ.get("OPENALGO_API_KEY")
    if existing:
        return existing
    if not use_local_key:
        raise StrategyError(
            "OPENALGO_API_KEY is missing. Use the Python host or --local-key on this installed machine."
        )
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    from database.auth_db import get_api_key_for_tradingview

    database = ROOT / "db" / "openalgo.db"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        accounts = connection.execute("SELECT user_id FROM api_keys").fetchall()
    if len(accounts) != 1:
        raise StrategyError(
            "Expected one existing OpenAlgo API key. Generate your key in Profile > API Key."
        )
    key = get_api_key_for_tradingview(accounts[0][0])
    if not key:
        raise StrategyError(
            "Could not read the saved OpenAlgo API key. Verify this installation's original .env."
        )
    return key


def fetch_candles(client, start_date, end_date):
    bars = client.history(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        interval=INTERVAL,
        start_date=start_date,
        end_date=end_date,
    )
    if not isinstance(bars, pd.DataFrame):
        raise StrategyError(
            "History request failed. Check broker login, supported intervals, and history entitlement."
        )
    return indicators(bars, datetime.now(IST))


def evaluate(client, mode, start_date, end_date):
    require_analyzer(client.analyzerstatus())
    frame = fetch_candles(client, start_date, end_date)
    timings = client.timings(date=datetime.now(IST).date().isoformat())
    if not isinstance(timings, dict) or timings.get("status") != "success":
        raise StrategyError("Could not verify the exchange calendar.")
    sessions = [session for session in timings.get("data", []) if session.get("exchange") == "NSE"]
    now_ms = int(datetime.now(IST).timestamp() * 1000)
    is_open = any(session["start_time"] <= now_ms < session["end_time"] for session in sessions)
    crosses = frame.loc[frame["signal"] != "WAIT"]
    return frame, {
        "strategy": "First NIFTY EMA 5/20",
        "test_type": "historical signal replay"
        if mode == "replay"
        else "read-only Analyzer readiness check",
        "instrument": {"symbol": SYMBOL, "exchange": EXCHANGE, "interval": INTERVAL},
        "analyzer_mode": True,
        "nse_session_open_now": is_open,
        "completed_candles": len(frame),
        "history_first_candle": frame.index[0].isoformat(),
        "history_last_candle": frame.index[-1].isoformat(),
        "latest": snapshot(frame),
        "bullish_crossovers": int(frame["bullish_cross"].sum()),
        "bearish_crossovers": int(frame["bearish_cross"].sum()),
        "recent_crossovers": [snapshot(frame.loc[:timestamp]) for timestamp in crosses.index[-10:]],
        "orders_submitted": 0,
        "limitations": [
            "Signals only: NIFTY index is not an executable contract.",
            "This report is not an Analyzer order/fill test or a profitability backtest.",
            "A futures/options contract and its lot size must be selected before adding orders.",
        ],
    }


def main(argv=None):
    today = datetime.now(IST).date()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "replay", "monitor"), default="check")
    parser.add_argument(
        "--local-key",
        action="store_true",
        help="Use this installation's saved OpenAlgo key in memory",
    )
    parser.add_argument("--start-date", default=(today - timedelta(days=10)).isoformat())
    parser.add_argument("--end-date", default=today.isoformat())
    parser.add_argument(
        "--minutes", type=int, default=15, help="Bounded duration for read-only monitoring (1-60)"
    )
    parser.add_argument("--report", type=Path, help="Save the credential-free JSON signal report")
    args = parser.parse_args(argv)
    if not 1 <= args.minutes <= 60:
        parser.error("--minutes must be between 1 and 60")
    if args.start_date > args.end_date:
        parser.error("--start-date must not follow --end-date")
    from openalgo import api

    with api(
        api_key=load_key(args.local_key),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
        timeout=30,
    ) as client:
        frame, report = evaluate(client, args.mode, args.start_date, args.end_date)
        print(json.dumps(report, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.mode != "monitor":
            return 0
        if not report["nse_session_open_now"]:
            print("Exchange is closed. No monitor started and no orders submitted.", flush=True)
            return 0
        tracker = CandleTracker()
        tracker.observe(frame)  # Seed with the current candle; do not re-emit it.
        deadline = time.monotonic() + args.minutes * 60
        while time.monotonic() < deadline:
            time.sleep(min(15, max(0, deadline - time.monotonic())))
            frame, current = evaluate(
                client, "monitor", args.start_date, datetime.now(IST).date().isoformat()
            )
            if not current["nse_session_open_now"]:
                print("Exchange session ended; monitor stopped.", flush=True)
                break
            event = tracker.observe(frame)
            if event:
                print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StrategyError as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Signal monitor stopped. No orders were submitted.", flush=True)
