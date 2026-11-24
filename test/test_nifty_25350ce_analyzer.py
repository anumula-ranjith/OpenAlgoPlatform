"""Offline policy and order-lifecycle tests; no application database writes."""

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from strategies.examples import nifty_25350ce_analyzer as strategy

NOW = datetime(2026, 11, 25, 10, 0, tzinfo=strategy.IST)


def signal(kind="BULLISH_CROSS", when=NOW):
    return {"signal": kind, "candle_close_ist": when.isoformat()}


def test_fresh_bullish_cross_buys_one_lot_only_after_fill():
    policy = strategy.Policy()
    assert policy.decide(signal(), 100, NOW, True)[0] == "BUY"
    assert policy.quantity == 0
    policy.filled("BUY", 100)
    assert policy.quantity == 65 and policy.entries == 1
    assert (
        policy.decide(
            signal(when=NOW + timedelta(minutes=5)), 100, NOW + timedelta(minutes=5), True
        )[0]
        == "WAIT"
    )


def test_bearish_signal_never_opens_a_short():
    assert strategy.Policy().decide(signal("BEARISH_CROSS"), 100, NOW, True)[0] == "WAIT"


def test_bearish_signal_exits_existing_long():
    policy = strategy.Policy(quantity=65, entry_price=100, entries=1)
    assert policy.decide(signal("BEARISH_CROSS"), 100, NOW, True) == ("SELL", "bearish_crossover")
    policy.filled("SELL", 99)
    assert policy.quantity == 0 and policy.entries == 1
    assert (
        policy.decide(
            signal(when=NOW + timedelta(minutes=5)), 100, NOW + timedelta(minutes=5), True
        )[0]
        == "WAIT"
    )


@pytest.mark.parametrize(("premium", "reason"), [(90, "premium_stop"), (120, "premium_target")])
def test_premium_exits_do_not_wait_for_another_candle(premium, reason):
    policy = strategy.Policy(
        quantity=65, entry_price=100, entries=1, last_signal_bar=NOW.isoformat()
    )
    assert policy.decide(signal("WAIT"), premium, NOW, True) == ("SELL", reason)


def test_time_exit():
    now = NOW.replace(hour=15, minute=5)
    assert strategy.Policy(quantity=65, entry_price=100, entries=1).decide(
        signal("WAIT"), 100, now, True
    ) == ("SELL", "time_exit")


@pytest.mark.parametrize("minutes", [-10, 7, 1440])
def test_old_or_future_signals_cannot_open_positions(minutes):
    old = signal(when=NOW - timedelta(minutes=minutes))
    assert strategy.Policy().decide(old, 100, NOW, True)[0] == "WAIT"


def test_duplicate_candle_is_not_retried():
    policy = strategy.Policy()
    assert policy.decide(signal(), 100, NOW, True)[0] == "BUY"
    assert policy.decide(signal(), 100, NOW, True) == ("WAIT", "already_processed_candle")


@pytest.mark.parametrize(("hour", "minute"), [(9, 25), (14, 30), (15, 0)])
def test_entry_window(hour, minute):
    now = NOW.replace(hour=hour, minute=minute)
    assert strategy.Policy().decide(signal(when=now), 100, now, True)[0] == "WAIT"


def test_closed_exchange_cannot_trade():
    assert strategy.Policy().decide(signal(), 100, NOW, False) == ("WAIT", "exchange_closed")


def contract(lot=65):
    return {
        "status": "success",
        "data": {
            "symbol": strategy.SYMBOL,
            "exchange": "NFO",
            "instrumenttype": "CE",
            "expiry": "01-DEC-26",
            "lotsize": lot,
            "tick_size": 0.05,
        },
    }


def test_contract_metadata_is_verified():
    assert strategy.check_contract(contract(), date(2026, 11, 24))["quantity"] == 65


@pytest.mark.parametrize("response", [None, {"status": "error"}, contract(50)])
def test_missing_or_changed_contract_fails_closed(response):
    with pytest.raises(strategy.StrategyError):
        strategy.check_contract(response, date(2026, 11, 24))


def test_expired_contract_is_not_rolled_automatically():
    with pytest.raises(strategy.StrategyError, match="expired"):
        strategy.check_contract(contract(), date(2026, 12, 2))


class FakeClient:
    def __init__(self, analyze=True):
        self.analyze = analyze

    def analyzerstatus(self):
        return {"status": "success", "data": {"analyze_mode": self.analyze}}

    def timings(self, **kwargs):
        return {"status": "success", "data": []}


class FakeBroker(strategy.SandboxOnlyBroker):
    def __init__(self, state="complete", quantity=0, analyze=True, entries=0):
        self.client, self.tick = FakeClient(analyze), Decimal("0.05")
        self.quantity, self.state, self.entries = quantity, state, entries
        self.submissions, self.cancellations = [], []
        self._place = object()

    def preflight(self):
        strategy.require_analyzer(self.client.analyzerstatus())

    def position(self):
        return self.quantity

    def pending(self):
        return []

    def previous_entries(self):
        return self.entries

    def quote(self):
        return 100

    def _call(self, function, payload):
        self.submissions.append(payload)
        return {"status": "success", "mode": "analyze", "orderid": "TEST-ORDER"}

    def status(self, orderid):
        return {
            "order_status": self.state,
            "filled_quantity": 65 if self.state == "complete" else 0,
            "average_price": 100,
        }

    def cancel(self, orderid):
        self.cancellations.append(orderid)
        return {"order_status": "cancelled", "filled_quantity": 0, "average_price": 0}


def test_live_mode_cannot_reach_the_sandbox_submission():
    broker = FakeBroker(analyze=False)
    with pytest.raises(strategy.StrategyError, match="Analyzer mode is disabled"):
        broker.submit_limit("BUY", Decimal("100"))
    assert broker.submissions == []


def test_limit_price_tick_validation():
    broker = FakeBroker()
    with pytest.raises(strategy.StrategyError, match="tick size"):
        broker.submit_limit("BUY", Decimal("100.01"))
    assert broker.submissions == []


def test_only_verified_complete_fill_is_accepted():
    broker = FakeBroker()
    fill = broker.fill_limit("BUY")
    assert fill["filled_quantity"] == 65 and fill["mode"] == "analyze"
    assert len(broker.submissions) == 1
    assert broker.submissions[0]["pricetype"] == "LIMIT"
    assert broker.submissions[0]["price"] == "100.10"


def test_pending_order_is_cancelled_and_not_resubmitted(monkeypatch):
    broker = FakeBroker(state="open")
    clock = iter([0, 11])
    monkeypatch.setattr(strategy.time, "monotonic", lambda: next(clock))
    with pytest.raises(strategy.StrategyError, match="not filled"):
        broker.fill_limit("BUY")
    assert len(broker.submissions) == 1 and broker.cancellations == ["TEST-ORDER"]


def test_status_failure_still_attempts_known_order_cleanup():
    broker = FakeBroker()

    def status_failure(orderid):
        raise strategy.StrategyError("status unavailable")

    broker.status = status_failure
    with pytest.raises(strategy.StrategyError, match="status unavailable"):
        broker.fill_limit("BUY")
    assert broker.cancellations == ["TEST-ORDER"] and len(broker.submissions) == 1


def test_restart_daily_entry_limit_cannot_be_bypassed():
    broker = FakeBroker(entries=1)
    with pytest.raises(strategy.StrategyError, match="already entered"):
        broker.fill_limit("BUY")
    assert broker.submissions == []


def test_sell_without_owned_position_is_refused():
    broker = FakeBroker(quantity=0)
    with pytest.raises(strategy.StrategyError, match="exposure"):
        broker.fill_limit("SELL")
    assert broker.submissions == []


def test_holiday_paper_run_submits_nothing():
    broker = FakeBroker()
    report = strategy.paper_run(broker.client, broker, minutes=1)
    assert report["result"] == "exchange_closed" and report["orders_submitted"] == 0
    assert broker.submissions == []
