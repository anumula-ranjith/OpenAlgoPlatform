"""Read-only entry/exit and P&L reconciliation from actual sandbox records."""

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategies.examples.nifty_25350ce_analyzer import (
    EXCHANGE,
    EXPECTED_LOT,
    PRODUCT,
    STRATEGY,
    SYMBOL,
)

IST = ZoneInfo("Asia/Kolkata")


class VerificationError(RuntimeError):
    pass


def amount(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise VerificationError("A required price or charge is missing or invalid.") from None
    if not number.is_finite():
        raise VerificationError("Non-finite price or charges in sandbox records.")
    return number


def money(value):
    return format(value.quantize(Decimal("0.01")), ".2f")


def reconcile(
    orders, trades, position_quantity, pending_orders, analyze_mode, session_date, now=None
):
    now = now or datetime.now(IST)
    if analyze_mode is not True:
        raise VerificationError(
            "Analyze mode is not enabled. No live account operations were attempted."
        )
    by_id = {order["orderid"]: order for order in orders}
    if len(by_id) != len(orders):
        raise VerificationError("Duplicate order IDs prevent reconciliation.")
    trade_ids = [trade["tradeid"] for trade in trades]
    if len(trade_ids) != len(set(trade_ids)):
        raise VerificationError("Duplicate trade IDs prevent reconciliation.")
    for record in [*orders, *trades]:
        if (
            record["symbol"] != SYMBOL
            or record["exchange"] != EXCHANGE
            or record["product"] != PRODUCT
            or record["strategy"] != STRATEGY
            or record["action"] not in ("BUY", "SELL")
        ):
            raise VerificationError("Unexpected contract, product, action or strategy in this run.")
    for order in orders:
        if order["quantity"] != EXPECTED_LOT:
            raise VerificationError("An order quantity differs from the authorized one-lot run.")
        fills = [trade for trade in trades if trade["orderid"] == order["orderid"]]
        filled = sum(trade["quantity"] for trade in fills)
        if filled != order["filled_quantity"]:
            raise VerificationError("Order fill quantity and trade records disagree.")
        if order["order_status"] == "complete" and filled != EXPECTED_LOT:
            raise VerificationError("A completed order does not have a verified full-lot fill.")
        if order["order_status"] != "complete" and filled:
            raise VerificationError("A non-complete order unexpectedly contains fills.")
    for trade in trades:
        if trade["orderid"] not in by_id or trade["action"] != by_id[trade["orderid"]]["action"]:
            raise VerificationError("A trade is not associated with the expected order/action.")
        if (
            trade["quantity"] <= 0
            or amount(trade["price"]) <= 0
            or amount(trade["total_charges"]) < 0
        ):
            raise VerificationError("A fill contains invalid quantity, premium or charges.")
    completed = sorted(
        (order for order in orders if order["order_status"] == "complete"),
        key=lambda order: (order["order_timestamp"], order["orderid"]),
    )
    fees = sum((amount(trade["total_charges"]) for trade in trades), Decimal("0"))
    report = {
        "session_date_ist": session_date.isoformat(),
        "verified_at_ist": now.isoformat(),
        "mode": "analyze",
        "symbol": SYMBOL,
        "strategy": STRATEGY,
        "quantity_per_order": EXPECTED_LOT,
        "orders_seen": len(orders),
        "completed_order_ids": [order["orderid"] for order in completed],
        "position_quantity": position_quantity,
        "pending_orders": pending_orders,
        "entry_verified": False,
        "exit_verified": False,
        "charges_inr": money(fees),
        "gross_pnl_inr": None,
        "net_pnl_inr": None,
        "live_account_operations": 0,
    }
    if not completed:
        if trades or position_quantity or pending_orders:
            report["status"] = "attention_required"
        elif session_date > now.date():
            report["status"] = "awaiting_session"
        elif session_date == now.date() and now.strftime("%H:%M") < "15:05":
            report["status"] = "waiting_for_signal"
        else:
            report["status"] = "no_filled_trade"
        report["pnl_note"] = "No completed entry/exit pair: trade P&L is not verified."
        return report
    first = completed[0]
    if first["action"] != "BUY" or len(completed) > 2:
        raise VerificationError(
            "Expected exactly one long entry followed by one exit, with no repeated entries."
        )
    report["entry_verified"] = True
    report["entry_order_id"] = first["orderid"]
    buys = [trade for trade in trades if trade["action"] == "BUY"]
    buy_value = sum((amount(trade["price"]) * trade["quantity"] for trade in buys), Decimal("0"))
    report["entry_average_premium"] = money(buy_value / EXPECTED_LOT)
    if len(completed) == 1:
        report["status"] = (
            "entry_open"
            if position_quantity == EXPECTED_LOT and not pending_orders
            else "attention_required"
        )
        report["pnl_note"] = "Entry filled; exit and realized net P&L remain unverified."
        return report
    second = completed[1]
    if second["action"] != "SELL":
        raise VerificationError("The second completed order is not a closing SELL.")
    sells = [trade for trade in trades if trade["action"] == "SELL"]
    sell_value = sum((amount(trade["price"]) * trade["quantity"] for trade in sells), Decimal("0"))
    if position_quantity or pending_orders:
        report["status"] = "attention_required"
        report["pnl_note"] = (
            "Both fills exist, but the selected contract is not confirmed flat with no pending orders."
        )
        return report
    report.update(
        {
            "status": "round_trip_verified",
            "exit_verified": True,
            "exit_order_id": second["orderid"],
            "exit_average_premium": money(sell_value / EXPECTED_LOT),
            "gross_pnl_inr": money(sell_value - buy_value),
            "net_pnl_inr": money(sell_value - buy_value - fees),
            "pnl_note": "Actual sandbox fills less their recorded charges; not live-account P&L.",
        }
    )
    return report


def read_records(session_date):
    def rows(connection, sql, parameters=()):
        return [dict(row) for row in connection.execute(sql, parameters)]

    with sqlite3.connect((ROOT / "db/openalgo.db").as_uri() + "?mode=ro", uri=True) as application:
        users = application.execute("SELECT user_id FROM api_keys").fetchall()
        if len(users) != 1:
            raise VerificationError("Expected exactly one configured OpenAlgo account.")
        mode = application.execute("SELECT analyze_mode FROM settings LIMIT 1").fetchone()
    start, stop = session_date.isoformat(), (session_date + timedelta(days=1)).isoformat()
    user = users[0][0]
    parameters = (user, SYMBOL, EXCHANGE, PRODUCT, STRATEGY, start, stop)
    with sqlite3.connect((ROOT / "db/sandbox.db").as_uri() + "?mode=ro", uri=True) as sandbox:
        sandbox.row_factory = sqlite3.Row
        orders = rows(
            sandbox,
            "SELECT * FROM sandbox_orders WHERE user_id=? AND symbol=? AND exchange=? AND product=? AND strategy=? AND order_timestamp>=? AND order_timestamp<? ORDER BY order_timestamp,orderid",
            parameters,
        )
        trades = rows(
            sandbox,
            "SELECT t.* FROM sandbox_trades t JOIN sandbox_orders o ON o.orderid=t.orderid WHERE o.user_id=? AND o.symbol=? AND o.exchange=? AND o.product=? AND o.strategy=? AND o.order_timestamp>=? AND o.order_timestamp<?",
            parameters,
        )
        selected = (user, SYMBOL, EXCHANGE, PRODUCT)
        quantity = sandbox.execute(
            "SELECT COALESCE(SUM(quantity),0) FROM sandbox_positions WHERE user_id=? AND symbol=? AND exchange=? AND product=?",
            selected,
        ).fetchone()[0]
        pending = sandbox.execute(
            "SELECT COUNT(*) FROM sandbox_orders WHERE user_id=? AND symbol=? AND exchange=? AND product=? AND order_status IN ('open','trigger pending')",
            selected,
        ).fetchone()[0]
    return reconcile(orders, trades, quantity, pending, bool(mode and mode[0]), session_date)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = read_records(args.date)
    except (VerificationError, sqlite3.Error) as error:
        report = {
            "status": "attention_required",
            "session_date_ist": args.date.isoformat(),
            "message": str(error),
        }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["status"] == "attention_required" else 0


if __name__ == "__main__":
    raise SystemExit(main())
