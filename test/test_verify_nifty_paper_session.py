"""Synthetic accounting fixtures, never actual account P&L or broker calls."""

from copy import deepcopy
from datetime import date, datetime

import pytest

from scripts import verify_nifty_paper_session as verifier
from strategies.examples import nifty_25350ce_analyzer as strategy

DAY = date(2026, 11, 25)
AFTER = datetime(2026, 11, 25, 15, 10, tzinfo=verifier.IST)


def fixtures(exit_price=110):
    common = {
        "symbol": strategy.SYMBOL,
        "exchange": strategy.EXCHANGE,
        "product": strategy.PRODUCT,
        "strategy": strategy.STRATEGY,
        "quantity": 65,
    }
    orders = [
        {
            **common,
            "orderid": "BUY1",
            "action": "BUY",
            "order_status": "complete",
            "filled_quantity": 65,
            "order_timestamp": "2026-11-25 10:00:00",
        },
        {
            **common,
            "orderid": "SELL1",
            "action": "SELL",
            "order_status": "complete",
            "filled_quantity": 65,
            "order_timestamp": "2026-11-25 10:15:00",
        },
    ]
    trades = [
        {
            **common,
            "orderid": "BUY1",
            "tradeid": "T1",
            "action": "BUY",
            "price": 100,
            "total_charges": 25.10,
        },
        {
            **common,
            "orderid": "SELL1",
            "tradeid": "T2",
            "action": "SELL",
            "price": exit_price,
            "total_charges": 26.20,
        },
    ]
    return orders, trades


def check(orders, trades, quantity=0, pending=0, mode=True):
    return verifier.reconcile(orders, trades, quantity, pending, mode, DAY, AFTER)


def test_round_trip_reconciles_gross_fees_and_net_from_fills():
    report = check(*fixtures())
    assert report["status"] == "round_trip_verified"
    assert report["entry_verified"] and report["exit_verified"]
    assert report["gross_pnl_inr"] == "650.00"
    assert report["charges_inr"] == "51.30"
    assert report["net_pnl_inr"] == "598.70"


def test_loss_includes_both_sides_charges():
    report = check(*fixtures(exit_price=90))
    assert report["gross_pnl_inr"] == "-650.00"
    assert report["net_pnl_inr"] == "-701.30"


def test_entry_only_does_not_report_invested_cash_as_realized_loss():
    orders, trades = fixtures()
    report = check(orders[:1], trades[:1], quantity=65)
    assert report["status"] == "entry_open"
    assert report["entry_verified"] and not report["exit_verified"]
    assert report["gross_pnl_inr"] is None and report["net_pnl_inr"] is None


def test_no_fills_are_not_reported_as_verified_entry_exit():
    report = check([], [])
    assert report["status"] == "no_filled_trade"
    assert not report["entry_verified"] and not report["exit_verified"]
    assert report["net_pnl_inr"] is None


def test_future_session_is_explicitly_pending():
    before = datetime(2026, 11, 24, 12, 0, tzinfo=verifier.IST)
    report = verifier.reconcile([], [], 0, 0, True, DAY, before)
    assert report["status"] == "awaiting_session"


def test_cancelled_unfilled_order_is_not_an_entry():
    orders, _ = fixtures()
    orders = [dict(orders[0], order_status="cancelled", filled_quantity=0)]
    assert not check(orders, [])["entry_verified"]


@pytest.mark.parametrize(("quantity", "pending"), [(65, 0), (0, 1)])
def test_exposure_or_pending_order_prevents_success(quantity, pending):
    report = check(*fixtures(), quantity=quantity, pending=pending)
    assert report["status"] == "attention_required" and not report["exit_verified"]


def test_live_mode_is_refused():
    with pytest.raises(verifier.VerificationError, match="Analyze mode"):
        check(*fixtures(), mode=False)


def test_order_trade_quantity_mismatch_is_refused():
    orders, trades = fixtures()
    trades[0]["quantity"] = 64
    with pytest.raises(verifier.VerificationError, match="disagree"):
        check(orders, trades)


def test_duplicate_trade_is_not_counted_twice():
    orders, trades = fixtures()
    trades.append(deepcopy(trades[0]))
    with pytest.raises(verifier.VerificationError, match="Duplicate trade"):
        check(orders, trades)


def test_missing_charges_are_not_assumed_zero():
    orders, trades = fixtures()
    trades[0]["total_charges"] = None
    with pytest.raises(verifier.VerificationError, match="missing or invalid"):
        check(orders, trades)


def test_wrong_contract_is_refused():
    orders, trades = fixtures()
    orders[0]["symbol"] = "OTHER"
    with pytest.raises(verifier.VerificationError, match="Unexpected contract"):
        check(orders, trades)


def test_short_first_sequence_is_refused():
    orders, trades = fixtures()
    orders.reverse()
    orders[0]["order_timestamp"] = "2026-11-25 09:59:00"
    with pytest.raises(verifier.VerificationError, match="long entry"):
        check(orders, trades)


def test_wrong_session_date_stops_before_accessing_keys(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 24, 12, 0, tzinfo=verifier.IST)

    monkeypatch.setattr(strategy, "datetime", Clock)

    def keys_must_not_be_read(value):
        raise AssertionError("No key access is allowed on the wrong session date")

    monkeypatch.setattr(strategy, "load_key", keys_must_not_be_read)
    with pytest.raises(strategy.StrategyError, match="specified IST session date"):
        strategy.main(["--session-date", "2026-11-25", "--minutes", "335"])
