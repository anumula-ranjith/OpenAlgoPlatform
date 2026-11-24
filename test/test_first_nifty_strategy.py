"""Offline signal tests; never imports the application or submits an order."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "strategies/examples/first_nifty_strategy.py"
SPEC = importlib.util.spec_from_file_location("first_nifty_strategy", SOURCE)
strategy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = strategy
SPEC.loader.exec_module(strategy)


def candles(closes):
    return pd.DataFrame(
        {
            "open": closes,
            "high": np.array(closes) + 1,
            "low": np.array(closes) - 1,
            "close": closes,
        },
        index=pd.date_range(
            "2026-11-23 09:15", periods=len(closes), freq="5min", tz="Asia/Kolkata"
        ),
    )


NOW = pd.Timestamp("2026-11-23 15:30", tz="Asia/Kolkata")


def test_crossover_sequence_contains_both_directions():
    closes = (
        list(np.linspace(200, 100, 40))
        + list(np.linspace(100, 200, 40))
        + list(np.linspace(200, 100, 40))
    )
    frame = strategy.indicators(
        candles(closes), pd.Timestamp("2026-11-24 15:30", tz="Asia/Kolkata")
    )
    assert frame.loc[frame["signal"] != "WAIT", "signal"].tolist() == [
        "BULLISH_CROSS",
        "BEARISH_CROSS",
    ]


def test_unfinished_candle_cannot_change_signal():
    bars = candles(list(np.linspace(200, 100, 39)) + [100000])
    now = bars.index[-1] + pd.Timedelta(minutes=4, seconds=59)
    frame = strategy.indicators(bars, now)
    assert frame.index[-1] == bars.index[-2]
    assert frame.iloc[-1]["signal"] == "WAIT"


def test_candle_is_included_exactly_at_close():
    bars = candles(list(np.linspace(200, 100, 40)))
    frame = strategy.indicators(bars, bars.index[-1] + pd.Timedelta(minutes=5))
    assert len(frame) == len(bars)


def test_duplicate_poll_and_out_of_order_data_do_not_emit_again():
    frame = strategy.indicators(candles(list(np.linspace(200, 100, 40))), NOW)
    tracker = strategy.CandleTracker()
    assert tracker.observe(frame) is not None
    assert tracker.observe(frame) is None
    assert tracker.observe(frame.iloc[:-1]) is None


@pytest.mark.parametrize(
    "response",
    [
        {"status": "success", "data": {"analyze_mode": False}},
        {"status": "success", "data": {"analyze_mode": "true"}},
        {"status": "error"},
        None,
    ],
)
def test_live_and_unverifiable_modes_are_refused(response):
    with pytest.raises(strategy.StrategyError):
        strategy.require_analyzer(response)


def test_no_future_data_leaks_into_past_indicators():
    first = candles(list(np.linspace(100, 200, 40)))
    altered = first.copy()
    altered.loc[altered.index[-5] :, ["open", "close"]] = 10000
    altered.loc[altered.index[-5] :, "high"] = 10001
    altered.loc[altered.index[-5] :, "low"] = 9999
    cutoff = first.index[-6] + pd.Timedelta(minutes=5)
    pd.testing.assert_frame_equal(
        strategy.indicators(first, cutoff), strategy.indicators(altered, cutoff)
    )


def test_naive_timestamps_are_rejected():
    bars = candles(list(np.linspace(100, 200, 40)))
    bars.index = bars.index.tz_localize(None)
    with pytest.raises(strategy.StrategyError, match="timezone"):
        strategy.indicators(bars, NOW)


def test_invalid_prices_are_rejected():
    bars = candles(list(np.linspace(100, 200, 40)))
    bars.loc[bars.index[-1], "close"] = np.nan
    with pytest.raises(strategy.StrategyError, match="invalid prices"):
        strategy.indicators(bars, NOW)


def test_insufficient_warmup_is_rejected():
    with pytest.raises(strategy.StrategyError, match="21 completed"):
        strategy.indicators(candles([100] * 20), NOW)


def test_flat_prices_do_not_invent_crossovers():
    frame = strategy.indicators(candles([100] * 40), NOW)
    assert set(frame["signal"]) == {"WAIT"}


def test_holiday_readiness_does_not_report_an_open_market_or_submit_orders():
    class ReadOnlyClient:
        def analyzerstatus(self):
            return {"status": "success", "data": {"analyze_mode": True}}

        def history(self, **kwargs):
            return candles([100] * 40)

        def timings(self, **kwargs):
            return {"status": "success", "data": []}

    _, report = strategy.evaluate(ReadOnlyClient(), "check", "2026-11-23", "2026-11-23")
    assert report["analyzer_mode"] is True
    assert report["nse_session_open_now"] is False
    assert report["orders_submitted"] == 0
    assert report["test_type"] == "read-only Analyzer readiness check"


def test_mode_is_checked_before_loading_broker_data():
    class LiveClient:
        def analyzerstatus(self):
            return {"status": "success", "data": {"analyze_mode": False}}

        def history(self, **kwargs):
            raise AssertionError("Live-mode check must stop before fetching history")

    with pytest.raises(strategy.StrategyError, match="Analyzer mode is disabled"):
        strategy.evaluate(LiveClient(), "check", "2026-11-23", "2026-11-23")
