"""Offline checks for causal rolling levels, frozen support, fills and fees."""

from datetime import datetime
from decimal import Decimal as D

import pandas as pd
import pytest

from scripts import backtest_nifty_sr as bt
from strategies.examples import nifty_sr_paper as strategy

DAY = "2026-11-19"
ZERO_SLIP = bt.Settings(slippage_bps="0")


def stamp(clock):
    return pd.Timestamp(f"{DAY} {clock}", tz=strategy.IST)


def data():
    idx = pd.date_range(stamp("09:15"), stamp("15:00"), freq="5min")
    index = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 65}, index=idx
    )
    index.loc[stamp("10:55") :, ["open", "high", "low", "close"]] = [102.0, 103.0, 101.5, 102.0]
    oi = pd.date_range(stamp("09:15"), stamp("15:05"), freq="5min")
    options = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 65}, index=oi
    )
    return index, options


def signal(clock="11:00", breakout=True, close=102):
    return {"known_at_ist": stamp(clock).isoformat(), "breakout": breakout, "close": close}


def test_levels_exclude_signal_candle_and_need_twenty_prior_bars():
    bars, _ = data()
    frame = strategy.breakout_signals(bars, stamp("11:00"))
    assert frame.iloc[:20].resistance.isna().all()
    assert frame.iloc[-1].resistance == 101
    assert frame.iloc[-1].support == 99
    assert bool(frame.iloc[-1].breakout)
    assert frame.iloc[-1].high == 103


def test_unfinished_candle_cannot_signal():
    bars, _ = data()
    assert not strategy.breakout_signals(bars, stamp("10:59")).breakout.any()
    assert strategy.breakout_signals(bars, stamp("11:00")).breakout.any()


def test_future_candles_cannot_change_known_levels_or_signal():
    bars, _ = data()
    original = strategy.breakout_signals(bars, stamp("11:00"))
    bars.loc[stamp("11:00") :, ["open", "high", "low", "close"]] = 9000
    pd.testing.assert_frame_equal(original, strategy.breakout_signals(bars, stamp("11:00")))


def test_window_does_not_include_previous_session():
    bars, _ = data()
    prior = bars.copy()
    prior.index -= pd.Timedelta(days=1)
    prior.loc[:, ["open", "high", "low", "close"]] = 9000
    result = strategy.breakout_signals(pd.concat([prior, bars]), stamp("11:00"))
    assert len(result) == 21
    assert result.iloc[-1].resistance == 101


def test_high_touch_without_close_above_is_not_entry():
    bars, _ = data()
    bars.loc[stamp("10:55"), "close"] = 101
    assert not strategy.breakout_signals(bars, stamp("11:00")).breakout.any()


def test_downside_break_does_not_buy_a_call():
    bars, _ = data()
    bars.loc[stamp("10:55"), ["open", "high", "low", "close"]] = [98, 99, 97, 98]
    frame = strategy.breakout_signals(bars, stamp("11:00"))
    assert bool(frame.iloc[-1].breakdown)
    assert not bool(frame.iloc[-1].breakout)


@pytest.mark.parametrize("clock", ["10:55", "14:00", "14:30"])
def test_entry_window(clock):
    assert strategy.Policy().decide(signal(clock), 100, stamp(clock).to_pydatetime())[0] == "WAIT"


def test_frozen_support_does_not_follow_new_range():
    state = strategy.Policy()
    state.filled("BUY", 100, 99, 101)
    assert (
        state.decide(signal("11:05", False, 100), 100, stamp("11:05").to_pydatetime())[0] == "WAIT"
    )
    assert state.support == 99
    assert state.decide(signal("11:10", False, 98), 100, stamp("11:10").to_pydatetime()) == (
        "SELL",
        "frozen_support_break",
    )


def test_support_equality_is_not_a_break():
    state = strategy.Policy(quantity=65, entry_price=100, entries=1, support=99, resistance=101)
    assert state.decide(signal(close=99), 100, stamp("11:00").to_pydatetime())[0] == "WAIT"


def test_only_one_entry_after_exit():
    state = strategy.Policy()
    state.filled("BUY", 100, 99, 101)
    state.filled("SELL", 100)
    assert state.decide(signal(), 100, stamp("11:00").to_pydatetime())[0] == "WAIT"


@pytest.mark.parametrize("clock", ["10:50", "11:05"])
def test_stale_and_future_signals_refused(clock):
    assert strategy.Policy().decide(signal(clock), 100, stamp("11:00").to_pydatetime())[0] == "WAIT"


def test_risk_exit_works_on_duplicate_signal():
    state = strategy.Policy(
        quantity=65,
        entry_price=100,
        entries=1,
        support=99,
        resistance=101,
        last_signal=signal()["known_at_ist"],
    )
    assert state.decide(signal(), 85, stamp("11:00").to_pydatetime()) == ("SELL", "premium_stop")


def test_time_exit_priority_over_stop_and_support():
    state = strategy.Policy(quantity=65, entry_price=100, entries=1, support=99, resistance=101)
    assert state.decide(signal("15:05", False, 90), 80, stamp("15:05").to_pydatetime()) == (
        "SELL",
        "time_exit",
    )


def test_next_open_entry_and_after_charge_loss_for_flat_gross():
    bars, options = data()
    options.loc[stamp("10:55"), ["open", "high", "low", "close"]] = 500
    trade = bt.run_day(strategy.breakout_signals(bars, stamp("15:35")), options, DAY, ZERO_SLIP)
    assert trade["entry_time_ist"] == stamp("11:00").isoformat()
    assert trade["entry_price"] == "100.00"
    assert trade["signal_candle_open_ist"] == stamp("10:55").isoformat()
    assert trade["exit_reason"] == "time_exit"
    assert trade["exit_candle_open_ist"] == stamp("15:05").isoformat()
    assert trade["gross_pnl"] == "0.00"
    assert trade["total_charges"] == "62.68"
    assert trade["net_pnl"] == "-62.68"


def test_simultaneous_stop_and_target_is_stop_first():
    bars, options = data()
    options.loc[stamp("11:00"), ["low", "high"]] = [80, 140]
    trade = bt.run_day(strategy.breakout_signals(bars, stamp("15:35")), options, DAY, ZERO_SLIP)
    assert trade["exit_reason"] == "premium_stop"
    assert trade["exit_price"] == "85.00"
    assert trade["same_bar_stop_and_target"] is True
    assert trade["exit_window_end_ist"] == stamp("11:05").isoformat()


def test_gap_stop_executes_at_worse_open():
    bars, options = data()
    options.loc[stamp("11:05"), ["open", "high", "low", "close"]] = 70
    trade = bt.run_day(strategy.breakout_signals(bars, stamp("15:35")), options, DAY, ZERO_SLIP)
    assert trade["exit_reason"] == "premium_stop"
    assert trade["exit_price"] == "70.00"
    assert not trade["intrabar_exit"]


def test_index_support_failure_waits_for_completed_close():
    bars, options = data()
    bars.loc[stamp("11:05"), ["open", "high", "low", "close"]] = [98, 99, 97, 98]
    trade = bt.run_day(strategy.breakout_signals(bars, stamp("15:35")), options, DAY, ZERO_SLIP)
    assert trade["exit_reason"] == "frozen_support_break"
    assert trade["exit_candle_open_ist"] == stamp("11:10").isoformat()


@pytest.mark.parametrize("clock", ["09:15", "11:05", "15:05"])
def test_missing_option_bar_excludes_day_even_outside_trade(clock):
    bars, options = data()
    options = options.drop(stamp(clock))
    report = bt.backtest(bars, options, DAY, DAY, ZERO_SLIP)
    assert report["summary"]["trades"] == 0
    assert report["coverage"][0]["status"] == "excluded_incomplete_option_session"
    assert report["summary"]["win_rate_percent"] is None


def test_zero_volume_excludes_session():
    bars, options = data()
    options.loc[stamp("09:15"), "volume"] = 0
    assert bt.backtest(bars, options, DAY, DAY, ZERO_SLIP)["summary"]["trades"] == 0


def test_no_signal_is_not_counted_as_a_loss():
    bars, options = data()
    bars.loc[:, ["open", "high", "low", "close"]] = 100
    report = bt.backtest(bars, options, DAY, DAY, ZERO_SLIP)
    assert report["coverage"][0]["status"] == "no_entry_signal"
    assert report["summary"]["trades"] == 0


def test_simulation_never_connects_to_network(monkeypatch):
    import socket

    def prohibited(*args, **kwargs):
        raise AssertionError("Offline simulator attempted network access")

    monkeypatch.setattr(socket.socket, "connect", prohibited)
    bars, options = data()
    report = bt.backtest(bars, options, DAY, DAY, ZERO_SLIP)
    assert not report["account_access"]
    assert report["orders_submitted"] == 0


@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0, -1])
def test_invalid_fill_price_rejected(price):
    with pytest.raises(ValueError):
        strategy.Policy().filled("BUY", price, 99, 101)


def test_invalid_entry_levels_rejected():
    with pytest.raises(ValueError):
        strategy.Policy().filled("BUY", 100, 102, 101)
