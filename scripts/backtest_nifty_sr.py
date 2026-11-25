"""Offline rolling support/resistance call backtest. No account or order APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.charges import STATUTORY_SCHEDULE_AS_OF, paisa
from scripts.nifty_paper_backtest_utils import (
    Settings,
    charges,
    fill,
    read_inputs,
    summary,
    validate_bars,
)
from strategies.examples.nifty_sr_paper import (
    IST,
    LOOKBACK,
    QUANTITY,
    STOP,
    SYMBOL,
    TARGET,
    Policy,
    breakout_signals,
)


def risk_touch(bar, entry):
    stop, target = entry * (1 - D(str(STOP))), entry * (1 + D(str(TARGET)))
    stopped, targeted = D(str(bar.low)) <= stop, D(str(bar.high)) >= target
    if stopped:
        return "premium_stop", stop, targeted
    if targeted:
        return "premium_target", target, False
    return None, None, False


def run_day(signals, options, day, settings):
    state, entry = Policy(), None
    step = pd.Timedelta(minutes=settings.option_minutes)
    start, end = pd.Timestamp(f"{day} 11:00", tz=IST), pd.Timestamp(f"{day} 15:05", tz=IST)
    for stamp in pd.date_range(start, end, freq=step):
        bar = options.loc[stamp]
        known = signals.loc[signals.index + pd.Timedelta(minutes=5) <= stamp]
        row = known.iloc[-1]
        known_at = known.index[-1] + pd.Timedelta(minutes=5)
        action, reason = state.decide(
            {
                "known_at_ist": known_at.isoformat(),
                "close": float(row.close),
                "breakout": bool(row.breakout),
            },
            float(bar.open),
            stamp.to_pydatetime(),
        )
        if action == "BUY":
            buy = fill(bar.open, "BUY", settings)
            state.filled("BUY", float(buy), float(row.support), float(row.resistance))
            entry = {
                "date": str(day),
                "symbol": SYMBOL,
                "quantity": QUANTITY,
                "entry_time_ist": stamp.isoformat(),
                "signal_known_at_ist": known_at.isoformat(),
                "signal_candle_open_ist": known.index[-1].isoformat(),
                "entry_price": str(buy.quantize(D("0.01"))),
                "frozen_support": float(row.support),
                "frozen_resistance": float(row.resistance),
            }
        intrabar, ambiguous, raw_exit = False, False, bar.open
        if state.quantity and action != "SELL":
            reason, raw_exit, ambiguous = risk_touch(bar, D(entry["entry_price"]))
            if reason:
                action, intrabar = "SELL", True
        if action == "SELL":
            sell = fill(raw_exit, "SELL", settings)
            fees = charges(D(entry["entry_price"]), "BUY", settings) + charges(
                sell, "SELL", settings
            )
            gross = paisa((sell - D(entry["entry_price"])) * QUANTITY)
            state.filled("SELL", float(sell))
            return {
                **entry,
                "exit_candle_open_ist": stamp.isoformat(),
                "exit_window_end_ist": (stamp + step if intrabar else stamp).isoformat(),
                "intrabar_exit": intrabar,
                "same_bar_stop_and_target": ambiguous,
                "exit_price": str(sell.quantize(D("0.01"))),
                "exit_reason": reason,
                "gross_pnl": str(gross),
                "charges": {k: str(v) for k, v in fees.as_columns().items()},
                "total_charges": str(fees.total),
                "net_pnl": str(paisa(gross - fees.total)),
                "modeled_only": True,
            }
    if state.quantity:
        raise ValueError("Unexpected modeled exposure after the time cutoff.")
    return None


def backtest(index, options, start, end, settings=Settings()):
    if settings.option_minutes not in (1, 5):
        raise ValueError("Option interval must be one or five minutes.")
    for value in (settings.slippage_bps, settings.brokerage_per_order):
        if not D(value).is_finite() or D(value) < 0:
            raise ValueError("Costs must be finite and nonnegative.")
    if D(settings.slippage_bps) > 100:
        raise ValueError("Slippage must be at most 100 bps per side.")
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    if (
        not date(2026, 4, 1)
        <= start
        <= end
        < min(date(2026, 12, 1), pd.Timestamp.now(tz=IST).date())
    ):
        raise ValueError(
            "Use ordered completed dates within this fee period and before contract expiry."
        )
    index, options = validate_bars(index, 5), validate_bars(options, settings.option_minutes)
    trades, coverage = [], []
    for day in sorted(set(index.index.date)):
        if not start <= day <= end:
            continue
        required_index = pd.date_range(f"{day} 09:15", f"{day} 15:00", freq="5min", tz=IST)
        required_option = pd.date_range(
            f"{day} 09:15", f"{day} 15:05", freq=f"{settings.option_minutes}min", tz=IST
        )
        audit = {"date": str(day)}
        if len(required_index.difference(index.index)):
            audit["status"] = "excluded_nonstandard_or_incomplete_index_session"
        else:
            missing = required_option.difference(options.index)
            zero_volume = int(
                (options.loc[required_option.intersection(options.index), "volume"] <= 0).sum()
            )
            audit.update(
                {"missing_option_bars": len(missing), "zero_volume_option_bars": zero_volume}
            )
            if len(missing) or zero_volume:
                audit["status"] = "excluded_incomplete_option_session"
            else:
                signals = breakout_signals(index, pd.Timestamp(f"{day} 15:35", tz=IST))
                trade = run_day(signals, options, day, settings)
                audit["status"] = "completed_trade" if trade else "no_entry_signal"
                if trade:
                    trades.append(trade)
                    audit["net_pnl"] = trade["net_pnl"]
        coverage.append(audit)
    return {
        "strategy": "NIFTY rolling support/resistance breakout (long calls only)",
        "test_type": "offline_historical_paper_simulation",
        "orders_submitted": 0,
        "account_access": False,
        "symbol": SYMBOL,
        "quantity": QUANTITY,
        "lookback_completed_bars": LOOKBACK,
        "period": {"start": str(start), "end": str(end)},
        "settings": asdict(settings),
        "charges_schedule_as_of": STATUTORY_SCHEDULE_AS_OF,
        "summary": summary(trades),
        "coverage": coverage,
        "trades": trades,
        "assumptions": [
            "Fixed 20-bar rolling same-session high/low, shifted one candle to exclude the signal bar.",
            "Buy one 65-unit fixed call after a completed NIFTY close strictly above resistance, 11:00 through before 14:00.",
            "One entry maximum per day; no puts, shorts, automatic strike selection or rollover.",
            "Exit on a completed close below entry's frozen support, 15% premium stop, 30% premium target, or 15:05.",
            "Open-time exit priority: time, premium stop, premium target, support break; gaps fill at open.",
            "Intrabar stop wins when both levels touch; exact intrabar execution time is unknown.",
            "Adverse slippage per side and adverse 0.05 tick rounding; premium levels use modeled entry fill.",
            "Eligibility is full-session positive-volume history before inspecting any signal or result.",
            "Missing sessions are unscored, never zero-P&L substitutions; no forward fill.",
            "Fees are the existing pure Analyzer charge engine with per-fill rounding, not actual broker contract notes.",
            "OHLC does not model spread, latency, order queues or quote polling between candles.",
            "Rules fixed before first run; small single-contract development sample, not independent out-of-sample evidence.",
        ],
    }


def write_report(report, output, chart=False):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [
        {k: v for k, v in t.items() if k != "charges"}
        | {f"fee_{k}": v for k, v in t["charges"].items()}
        for t in report["trades"]
    ]
    pd.DataFrame(rows).to_csv(output / "trades.csv", index=False)
    pd.DataFrame(report["coverage"]).to_csv(output / "coverage.csv", index=False)
    if chart and rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = np.array([float(t["net_pnl"]) for t in rows])
        labels = [t["date"][5:] for t in rows]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), layout="constrained")
        fig.suptitle(
            "NIFTY support/resistance breakout | Historical paper simulation",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].bar(labels, values, color=["#15803d" if n > 0 else "#b91c1c" for n in values])
        axes[0].set_ylabel("Net per trade (INR)")
        axes[1].plot(
            ["Start", *labels], np.r_[0, values.cumsum()], "o-", color="#1e3a8a", linewidth=2
        )
        axes[1].set_ylabel("Cumulative net P&L (INR)")
        axes[1].set_xlabel("Trade date, November 2026 (excluded sessions omitted)")
        for ax in axes:
            ax.axhline(0, color="#64748b", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        fig.savefig(output / "backtest.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "tmp/nifty_backtest/fresh_download")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp/nifty_sr/five_minute")
    parser.add_argument("--start", default="2026-11-03")
    parser.add_argument("--end", default="2026-11-23")
    parser.add_argument("--option-minutes", type=int, choices=(1, 5), default=5)
    parser.add_argument("--slippage-bps", default="10")
    parser.add_argument("--brokerage-per-order", default="20")
    parser.add_argument("--chart", action="store_true")
    args = parser.parse_args(argv)
    index, options, manifest = read_inputs(args.data_dir, args.option_minutes)
    report = backtest(
        index,
        options,
        args.start,
        args.end,
        Settings(args.option_minutes, args.slippage_bps, args.brokerage_per_order),
    )
    report["input_manifest"] = manifest
    report["strategy_source_sha256"] = hashlib.sha256(
        (ROOT / "strategies/examples/nifty_sr_paper.py").read_bytes()
    ).hexdigest()
    write_report(report, args.output_dir, args.chart)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "coverage": report["coverage"],
                "trades": report["trades"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"BACKTEST STOPPED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
