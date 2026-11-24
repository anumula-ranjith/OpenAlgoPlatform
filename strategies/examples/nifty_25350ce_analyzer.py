"""NIFTY 01-Dec-2026 25350 CE: a local, Analyzer-only learning strategy.

Signals: NIFTY index EMA 5/20 on completed five-minute candles.
Execution: buy one lot of the selected call, never short or add to a position.
Order writes call OpenAlgo's LOCAL sandbox service, never the live order API.
Run --mode check, replay, plumbing, or paper. Hosted runs default to paper;
terminal runs default to check. A paper run is bounded to 15 minutes by default.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
from filelock import FileLock, Timeout

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "app.py").is_file())
sys.path.insert(0, str(ROOT))

from strategies.examples.first_nifty_strategy import (
    StrategyError,
    fetch_candles,
    indicators,
    load_key,
    require_analyzer,
    snapshot,
)

IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "NIFTY01DEC2625350CE"
EXCHANGE = "NFO"
EXPIRY = date(2026, 12, 1)
STRATEGY = "First_NIFTY_25350CE_Analyzer"
PRODUCT = "MIS"
EXPECTED_LOT = 65
ENTRY_START = "09:30"
ENTRY_END = "14:30"
EXIT_TIME = "15:05"
STOP_FRACTION = 0.10
TARGET_FRACTION = 0.20


def check_contract(response, today):
    if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
        raise StrategyError(
            "The selected call contract could not be verified in the current master contract."
        )
    data = response["data"]
    expected = {
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "instrumenttype": "CE",
        "expiry": "01-DEC-26",
    }
    if response.get("status") != "success" or any(data.get(k) != v for k, v in expected.items()):
        raise StrategyError(
            "The selected call contract could not be verified in the current master contract."
        )
    if int(data.get("lotsize", 0)) != EXPECTED_LOT or Decimal(str(data.get("tick_size", 0))) <= 0:
        raise StrategyError(
            "Contract lot size or tick size changed. Review the strategy before running."
        )
    if today > EXPIRY:
        raise StrategyError(
            "The selected option expired. This strategy will not roll to another contract."
        )
    return {
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "expiry": EXPIRY.isoformat(),
        "quantity": EXPECTED_LOT,
        "lots": 1,
        "tick_size": float(data["tick_size"]),
    }


def session_open(client, now):
    response = client.timings(date=now.date().isoformat())
    if not isinstance(response, dict) or response.get("status") != "success":
        raise StrategyError("Could not verify the NFO exchange calendar.")
    epoch_ms = int(now.timestamp() * 1000)
    return any(
        row.get("exchange") == EXCHANGE and row["start_time"] <= epoch_ms < row["end_time"]
        for row in response.get("data", [])
    )


@dataclass
class Policy:
    """Small, testable long/flat state machine; state changes only after fills."""

    quantity: int = 0
    entry_price: float = 0.0
    entries: int = 0
    last_signal_bar: str = ""

    def decide(self, signal, premium, now, is_open):
        if not is_open:
            return "WAIT", "exchange_closed"
        if self.quantity not in (0, EXPECTED_LOT):
            raise StrategyError(
                "Unexpected simulated position size. Reconcile the sandbox position."
            )
        clock = now.strftime("%H:%M")
        if self.quantity:
            if clock >= EXIT_TIME:
                return "SELL", "time_exit"
            if premium <= self.entry_price * (1 - STOP_FRACTION):
                return "SELL", "premium_stop"
            if premium >= self.entry_price * (1 + TARGET_FRACTION):
                return "SELL", "premium_target"
        bar = signal["candle_close_ist"]
        if bar == self.last_signal_bar:
            return "WAIT", "already_processed_candle"
        self.last_signal_bar = bar
        closed_at = datetime.fromisoformat(bar)
        age = (now - closed_at).total_seconds()
        if closed_at.date() != now.date() or not 0 <= age <= 360:
            return "WAIT", "stale_or_future_signal"
        if self.quantity and signal["signal"] == "BEARISH_CROSS":
            return "SELL", "bearish_crossover"
        if (
            not self.quantity
            and self.entries == 0
            and ENTRY_START <= clock < ENTRY_END
            and signal["signal"] == "BULLISH_CROSS"
        ):
            return "BUY", "bullish_crossover"
        return "WAIT", "no_new_entry_or_exit"

    def filled(self, action, price):
        if action == "BUY" and self.quantity == 0 and self.entries == 0:
            self.quantity, self.entry_price = EXPECTED_LOT, price
            self.entries += 1
        elif action == "SELL" and self.quantity == EXPECTED_LOT:
            self.quantity, self.entry_price = 0, 0.0
        else:
            raise StrategyError("Unexpected fill transition; stopped without retrying.")


class SandboxOnlyBroker:
    """Local sandbox writes cannot fall through to a live broker order path."""

    def __init__(self, client, key, tick_size):
        from database.auth_db import verify_api_key
        from services.sandbox_service import (
            sandbox_cancel_order,
            sandbox_get_order_status,
            sandbox_place_order,
        )
        from utils.broker_keys import apply_logged_in_broker_keys

        self.client, self._key, self.tick = client, key, Decimal(str(tick_size))
        self.user_id = verify_api_key(key)
        if not self.user_id:
            raise StrategyError("OpenAlgo API key verification failed.")
        # Match app startup: UI-saved broker keys live in the encrypted DB,
        # not in .env. Sandbox price checks in this separate process need them.
        if apply_logged_in_broker_keys() is None:
            raise StrategyError(
                "No configured broker session is available for sandbox quote checks."
            )
        configured = os.getenv("SANDBOX_DATABASE_URL", "sqlite:///db/sandbox.db")
        if configured != "sqlite:///db/sandbox.db":
            raise StrategyError("This local example expects the standard sandbox database path.")
        self.database = ROOT / "db" / "sandbox.db"
        self._place, self._status, self._cancel = (
            sandbox_place_order,
            sandbox_get_order_status,
            sandbox_cancel_order,
        )

    def read(self, query, parameters=()):
        with sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, parameters)]

    def position(self):
        rows = self.read(
            "SELECT quantity FROM sandbox_positions WHERE user_id=? AND symbol=? AND exchange=? AND product=?",
            (self.user_id, SYMBOL, EXCHANGE, PRODUCT),
        )
        return sum(row["quantity"] for row in rows)

    def pending(self):
        return self.read(
            "SELECT orderid FROM sandbox_orders WHERE user_id=? AND symbol=? AND exchange=? AND product=? AND order_status IN ('open','trigger pending')",
            (self.user_id, SYMBOL, EXCHANGE, PRODUCT),
        )

    def previous_entries(self):
        rows = self.read(
            "SELECT COUNT(*) AS count FROM sandbox_orders WHERE user_id=? AND symbol=? AND strategy=? AND action='BUY' AND filled_quantity>0 AND order_timestamp>=?",
            (self.user_id, SYMBOL, STRATEGY, datetime.now(IST).date().isoformat()),
        )
        return rows[0]["count"]

    def preflight(self):
        require_analyzer(self.client.analyzerstatus())
        if self.position() != 0 or self.pending():
            raise StrategyError(
                "This contract already has an open sandbox position or order. Reconcile it before starting."
            )

    def _call(self, function, *args):
        from database.sandbox_db import db_session

        try:
            success, response, _ = function(*args, self._key, {})
        finally:
            db_session.remove()  # Read fresh fills made by the separate execution engine.
        if not success or response.get("status") != "success" or response.get("mode") != "analyze":
            raise StrategyError(
                "Sandbox operation did not confirm success in Analyze mode; inspect orders before retrying."
            )
        return response

    def quote(self):
        response = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
        if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
            raise StrategyError("A valid option premium could not be obtained.")
        data = response["data"]
        if response.get("status") != "success" or float(data.get("ltp", 0)) <= 0:
            raise StrategyError("A valid option premium could not be obtained.")
        return float(data["ltp"])

    def submit_limit(self, action, price, tag=STRATEGY):
        require_analyzer(self.client.analyzerstatus())
        if action not in ("BUY", "SELL") or price <= 0:
            raise StrategyError("Invalid simulated order parameters.")
        price_decimal = Decimal(str(price))
        if price_decimal % self.tick:
            raise StrategyError("Limit price must be a multiple of the verified tick size.")
        payload = {
            "strategy": tag,
            "symbol": SYMBOL,
            "exchange": EXCHANGE,
            "action": action,
            "quantity": EXPECTED_LOT,
            "pricetype": "LIMIT",
            "product": PRODUCT,
            "price": str(price_decimal),
        }
        response = self._call(self._place, payload)
        if not response.get("orderid"):
            raise StrategyError(
                "Order outcome is unknown. Inspect the sandbox order book; do not resend."
            )
        return response["orderid"]

    def status(self, orderid):
        return self._call(self._status, {"orderid": orderid})["data"]

    def cancel(self, orderid):
        # Cancellation stays sandbox-only even if the UI changes mode during cleanup.
        current = self.status(orderid)
        if current["order_status"] in ("open", "trigger pending"):
            try:
                self._call(self._cancel, {"orderid": orderid})
            except StrategyError:
                if self.status(orderid)["order_status"] != "complete":
                    raise
        return self.status(orderid)

    def fill_limit(self, action):
        expected = 0 if action == "BUY" else EXPECTED_LOT
        if self.position() != expected or self.pending():
            raise StrategyError(
                "Unexpected sandbox exposure before submission; stopped without trading."
            )
        if action == "BUY" and self.previous_entries() != 0:
            raise StrategyError("This strategy already entered today. A second entry is refused.")
        premium = Decimal(str(self.quote()))
        raw = premium * (Decimal("1.001") if action == "BUY" else Decimal("0.999"))
        rounding = ROUND_CEILING if action == "BUY" else ROUND_FLOOR
        price = max(self.tick, (raw / self.tick).to_integral_value(rounding=rounding) * self.tick)
        orderid = self.submit_limit(action, price)
        deadline = time.monotonic() + 10
        current = None
        try:
            current = self.status(orderid)
            while (
                current["order_status"] in ("open", "trigger pending")
                and time.monotonic() < deadline
            ):
                time.sleep(1)
                current = self.status(orderid)
        finally:
            if current is None or current["order_status"] in ("open", "trigger pending"):
                current = self.cancel(orderid)
        if current["order_status"] != "complete" or current["filled_quantity"] != EXPECTED_LOT:
            raise StrategyError(
                f"Sandbox order {orderid} was not filled; stopped without resubmitting."
            )
        return {
            "orderid": orderid,
            "action": action,
            "filled_quantity": current["filled_quantity"],
            "average_price": current["average_price"],
            "mode": "analyze",
        }


def paper_run(client, broker, minutes):
    broker.preflight()
    now = datetime.now(IST)
    if not session_open(client, now):
        return {
            "test": "paper session guard",
            "mode": "analyze",
            "result": "exchange_closed",
            "orders_submitted": 0,
            "position_quantity": broker.position(),
        }
    state = Policy(entries=broker.previous_entries())
    start = (now.date() - timedelta(days=10)).isoformat()
    initial = fetch_candles(client, start, now.date().isoformat())
    state.last_signal_bar = snapshot(initial)[
        "candle_close_ist"
    ]  # No startup trade on an old crossover.
    deadline, fills = time.monotonic() + minutes * 60, []
    try:
        while time.monotonic() < deadline:
            now = datetime.now(IST)
            require_analyzer(client.analyzerstatus())
            if broker.position() != state.quantity or broker.pending():
                raise StrategyError(
                    "Sandbox exposure changed outside this strategy. Reconcile it before restarting."
                )
            is_open = session_open(client, now)
            if not is_open or now.strftime("%H:%M") >= EXIT_TIME:
                break
            frame = fetch_candles(client, start, now.date().isoformat())
            latest = snapshot(frame)
            action, reason = state.decide(latest, broker.quote(), now, is_open)
            print(
                json.dumps(
                    {
                        "time_ist": now.isoformat(),
                        "signal": latest["signal"],
                        "decision": action,
                        "reason": reason,
                        "paper_quantity": state.quantity,
                    }
                ),
                flush=True,
            )
            if action != "WAIT":
                fill = broker.fill_limit(action)
                state.filled(action, fill["average_price"])
                fills.append({**fill, "reason": reason})
                print(json.dumps(fills[-1]), flush=True)
                if state.quantity == 0:
                    break  # One completed entry/exit cycle is enough for the first lesson.
            time.sleep(min(15, max(0, deadline - time.monotonic())))
    finally:
        if state.quantity == EXPECTED_LOT:
            require_analyzer(client.analyzerstatus())
            if broker.position() != EXPECTED_LOT or broker.pending():
                raise StrategyError(
                    "Exit not retried: inspect this contract's sandbox orders and position."
                )
            if not session_open(client, datetime.now(IST)):
                raise StrategyError(
                    "Session closed before cleanup. Inspect the remaining sandbox position."
                )
            fill = broker.fill_limit("SELL")
            state.filled("SELL", fill["average_price"])
            fills.append({**fill, "reason": "bounded_run_cleanup"})
    return {
        "test": "bounded paper strategy run",
        "mode": "analyze",
        "fills": fills,
        "position_quantity": broker.position(),
        "maximum_entries_per_day": 1,
    }


def plumbing_test(client, broker):
    """Actual sandbox create/cancel test, deliberately without a simulated fill."""
    broker.preflight()
    premium = broker.quote()
    if premium <= float(broker.tick) * 10:
        raise StrategyError("Premium is too close to the lowest tick for a non-marketable test.")
    before = broker.read(
        "SELECT available_balance,used_margin FROM sandbox_funds WHERE user_id=?", (broker.user_id,)
    )
    orderid = broker.submit_limit("BUY", broker.tick, tag=STRATEGY + "_Plumbing")
    current = None
    try:
        current = broker.status(orderid)
        if current["order_status"] != "open" or current["filled_quantity"] != 0:
            raise StrategyError(
                "The diagnostic order did not remain unfilled. Inspect this sandbox order."
            )
    finally:
        current = broker.cancel(orderid)
    after = broker.read(
        "SELECT available_balance,used_margin FROM sandbox_funds WHERE user_id=?", (broker.user_id,)
    )
    if (
        current["order_status"] != "cancelled"
        or current["filled_quantity"] != 0
        or broker.position() != 0
    ):
        raise StrategyError(
            "Diagnostic cleanup did not confirm a cancelled, unfilled order and flat position."
        )
    if before != after:
        raise StrategyError(
            "Sandbox balances did not return to their pre-test values. Inspect the fund ledger."
        )
    return {
        "test": "actual Analyzer order create/cancel",
        "mode": "analyze",
        "symbol": SYMBOL,
        "orderid": orderid,
        "quantity": EXPECTED_LOT,
        "limit_price": float(broker.tick),
        "last_available_quote_ltp": premium,
        "quote_is_not_a_fresh_market_fill": True,
        "initial_status": "open",
        "final_status": current["order_status"],
        "filled_quantity": 0,
        "position_quantity": 0,
        "sandbox_balances_restored": True,
        "live_orders_submitted": 0,
    }


def replay(client, start, end):
    """Historical index signals paired with next option opens; no broker orders."""
    frame = fetch_candles(client, start, end)
    options = client.history(
        symbol=SYMBOL, exchange=EXCHANGE, interval="5m", start_date=start, end_date=end
    )
    option_frame = indicators(options, datetime.now(IST))
    references = []
    for timestamp in frame.index[frame["signal"] != "WAIT"]:
        available_at = timestamp + pd.Timedelta(minutes=5)
        candidates = option_frame.loc[
            (option_frame.index >= available_at) & (option_frame.index.date == available_at.date())
        ]
        if candidates.empty or (candidates.index[0] - available_at).total_seconds() > 300:
            continue
        references.append(
            {
                "signal_available_ist": available_at.isoformat(),
                "signal": frame.loc[timestamp, "signal"],
                "next_option_bar_ist": candidates.index[0].isoformat(),
                "next_option_open_reference": float(candidates.iloc[0]["open"]),
            }
        )
    return {
        "test": "historical signal/option-data alignment",
        "symbol": SYMBOL,
        "index_candles": len(frame),
        "option_candles": len(option_frame),
        "signal_references": references,
        "orders_submitted": 0,
        "limitations": "Execution references only; no fills, costs, or profitability backtest are implied.",
    }


def main(argv=None):
    os.chdir(ROOT)
    today = datetime.now(IST).date()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("check", "replay", "plumbing", "paper"),
        default="paper" if os.getenv("STRATEGY_ID") else "check",
    )
    parser.add_argument("--local-key", action="store_true")
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument(
        "--session-date",
        type=date.fromisoformat,
        help="Refuse execution on any date other than this IST trading date",
    )
    parser.add_argument("--start-date", default=(today - timedelta(days=10)).isoformat())
    parser.add_argument("--end-date", default=today.isoformat())
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.minutes <= 360:
        parser.error("--minutes must be between 1 and 360")
    if args.session_date is not None and args.session_date != today:
        raise StrategyError(
            "This run is restricted to its specified IST session date. No orders submitted."
        )
    for value in (args.start_date, args.end_date):
        date.fromisoformat(value)
    if args.start_date > args.end_date:
        parser.error("The start date must not follow the end date")
    key = load_key(args.local_key)
    host = os.getenv("OPENALGO_HOST", os.getenv("HOST_SERVER", "http://127.0.0.1:5000"))
    address = urlsplit(host)
    if address.hostname not in ("127.0.0.1", "localhost") or (address.port or 80) != int(
        os.getenv("FLASK_PORT", "5000")
    ):
        raise StrategyError(
            "This sandbox-only example must run against this local OpenAlgo installation."
        )
    from openalgo import api

    with api(api_key=key, host=host, timeout=30) as client:
        require_analyzer(client.analyzerstatus())
        contract = check_contract(client.symbol(symbol=SYMBOL, exchange=EXCHANGE), today)
        broker = SandboxOnlyBroker(client, key, contract["tick_size"])
        if args.mode == "check":
            broker.preflight()
            report = {
                "test": "Analyzer readiness",
                "mode": "analyze",
                "contract": contract,
                "nfo_session_open": session_open(client, datetime.now(IST)),
                "existing_contract_position": broker.position(),
                "orders_submitted": 0,
            }
        elif args.mode == "replay":
            report = replay(client, args.start_date, args.end_date)
        else:
            (ROOT / "tmp").mkdir(exist_ok=True)
            try:
                with FileLock(str(ROOT / "tmp" / "first_nifty_25350ce.lock"), timeout=0):
                    report = (
                        plumbing_test(client, broker)
                        if args.mode == "plumbing"
                        else paper_run(client, broker, args.minutes)
                    )
            except Timeout:
                raise StrategyError(
                    "Another copy of this strategy or its diagnostic test is already running."
                ) from None
    report["verified_at_ist"] = datetime.now(IST).isoformat()
    report["contract"] = contract
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StrategyError as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Stopped. Review the selected contract's sandbox orders and position.", flush=True)
