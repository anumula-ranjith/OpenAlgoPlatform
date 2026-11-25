"""Offline helper checks for clean-checkout input integrity and net accounting."""

import hashlib
import json
from decimal import Decimal as D

import pandas as pd
import pytest

from scripts import nifty_paper_backtest_utils as utils


def write_inputs(folder):
    bars = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 65},
        index=pd.date_range("2026-11-19 09:15", periods=2, freq="5min", tz=utils.IST),
    )
    manifest = {"files": {}}
    for name, symbol, exchange in (
        ("underlying_5m.csv", "NIFTY", "NSE_INDEX"),
        ("option_5m.csv", utils.SYMBOL, "NFO"),
    ):
        path = folder / name
        bars.to_csv(path)
        manifest["files"][name] = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": "5m",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (folder / "download_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_verified_inputs_load_without_broker_or_account(tmp_path):
    write_inputs(tmp_path)
    index, option, manifest = utils.read_inputs(tmp_path, 5)
    assert len(index) == len(option) == 2
    assert index.index.tz is not None
    assert manifest["files"]["option_5m.csv"]["symbol"] == utils.SYMBOL


@pytest.mark.parametrize("kind", ["hash", "symbol", "interval"])
def test_tampered_history_is_rejected(tmp_path, kind):
    manifest = write_inputs(tmp_path)
    metadata = manifest["files"]["option_5m.csv"]
    metadata[{"hash": "sha256", "symbol": "symbol", "interval": "interval"}[kind]] = "wrong"
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        utils.read_inputs(tmp_path, 5)


def test_fill_slippage_and_tick_rounding_are_adverse():
    assert utils.fill("100.01", "BUY", utils.Settings()) == D("100.15")
    assert utils.fill("100.01", "SELL", utils.Settings()) == D("99.90")


def test_flat_round_trip_is_net_loss_after_charges():
    fees = utils.charges(D(100), "BUY", utils.Settings()) + utils.charges(
        D(100), "SELL", utils.Settings()
    )
    assert fees.total == D("62.68")
    result = utils.summary(
        [{"gross_pnl": "0.00", "total_charges": str(fees.total), "net_pnl": "-62.68"}]
    )
    assert result["win_rate_percent"] == 0
    assert result["losses_after_charges"] == 1
    assert result["net_pnl"] == "-62.68"


def test_summary_uses_net_winners_and_closed_trade_drawdown():
    result = utils.summary(
        [
            {"gross_pnl": "50", "total_charges": "60", "net_pnl": "-10"},
            {"gross_pnl": "100", "total_charges": "20", "net_pnl": "80"},
            {"gross_pnl": "-20", "total_charges": "5", "net_pnl": "-25"},
        ]
    )
    assert result["wins_after_charges"] == 1
    assert result["win_rate_percent"] == 33.33
    assert result["net_pnl"] == "45.00"
    assert result["closed_trade_max_drawdown"] == "25.00"


def test_empty_sample_has_no_win_rate():
    assert utils.summary([])["win_rate_percent"] is None


@pytest.mark.parametrize("kind", ["duplicate", "naive", "infinite", "bad_high", "negative_volume"])
def test_invalid_market_data_fails_closed(tmp_path, kind):
    write_inputs(tmp_path)
    index, _, _ = utils.read_inputs(tmp_path, 5)
    if kind == "duplicate":
        index = pd.concat([index, index.iloc[:1]])
    elif kind == "naive":
        index.index = index.index.tz_localize(None)
    elif kind == "infinite":
        index["open"] = index["open"].astype(float)
        index.iloc[0, index.columns.get_loc("open")] = float("inf")
    elif kind == "bad_high":
        index.iloc[0, index.columns.get_loc("high")] = 90
    else:
        index.iloc[0, index.columns.get_loc("volume")] = -1
    with pytest.raises(ValueError):
        utils.validate_bars(index, 5)
