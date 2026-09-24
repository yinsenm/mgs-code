#!/usr/bin/env python3
"""Run the paper's top-30 MGS+Div and MGS+RVOL comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mgs"))

from mgs.data import aggregate_sp500_daily_data
from mgs.lib_port_sp500_allocator import SP500PortfolioAllocator
from mgs.lib_port_sp500_backtester import run_sp500_backtest
from mgs.lib_port_strategy_cvx_guarded import PortfolioStrategyCVX as GuardedOptimizer
from mgs.run_sp500_weekly_features import prepare_weekly_sp500_panel


def build_feature_frames(source_csv: Path) -> dict[str, pd.DataFrame]:
    weekly = pd.read_csv(
        source_csv, parse_dates=["date"],
        usecols=["date", "permno", "adjusted_ret", "adjusted_close",
                 "adjusted_volume", "adjusted_close_200sma"],
    )
    monthly = aggregate_sp500_daily_data(
        df_daily=weekly, freq="monthly", asset_col="permno", date_col="date",
        column_methods={
            "adjusted_ret": "compound_return", "adjusted_close": "last",
            "adjusted_volume": "sum", "adjusted_close_200sma": "last",
        },
    ).sort_values(["date", "permno"])
    def panel(column: str) -> pd.DataFrame:
        return monthly.pivot(index="date", columns="permno", values=column).sort_index()
    returns = panel("adjusted_ret")
    volume = panel("adjusted_volume")
    close = panel("adjusted_close")
    sma200 = panel("adjusted_close_200sma")
    rvol = volume.divide(volume.rolling(12, min_periods=12).mean().replace(0.0, np.nan))
    divergence = close.divide(sma200)
    return {
        "ret": returns,
        "rvol": rvol.replace([np.inf, -np.inf], np.nan),
        "div": divergence.replace([np.inf, -np.inf], np.nan) - 1.0,
    }


def make_cov_func(modified_gerber_stat, frames: dict[str, pd.DataFrame], second: str):
    def cov_func(return_window: pd.DataFrame, gamma: float = 1.0, n: float = 1.0, half_life=None):
        inputs = [
            frames[name].reindex(index=return_window.index, columns=return_window.columns).fillna(0.0)
            for name in ("ret", second)
        ]
        return modified_gerber_stat(
            inputs, gamma=gamma, n=n, half_life=half_life,
            signal_weights=(0.5, 0.5), volatility_source=return_window,
        )
    return cov_func


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=ROOT / "data/sp500_1950_2025/sp500_weekly_features.csv")
    parser.add_argument("--prepared-data-path", type=Path, default=ROOT / "data/sp500_1950_2025/sp500_weekly_panel")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/sp500_signals")
    parser.add_argument("--rf-cache-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cov-impl", choices=("cpp", "python"), default="cpp")
    args = parser.parse_args()

    if args.cov_impl == "cpp":
        from mgs.lib_cov_func_cpp import modified_gerber_stat
    else:
        from mgs.lib_cov_func import modified_gerber_stat
    panel = Path(prepare_weekly_sp500_panel(str(args.source_csv), str(args.prepared_data_path)))
    frames = build_feature_frames(args.source_csv)
    start, end, lookback, ntop, tc_bps = "1990-01-31", "2025-12-31", 60, 30, 10
    run_name = (f"{panel.name}_ntop={ntop}_start={start}_end={end}_tc={tc_bps}_"
                f"lbm={lookback}_freq=monthly_cash=1")
    allocations = args.results_root / "allocations" / run_name
    performance = args.results_root / "performance" / f"{run_name}_rp=03-21-03"
    allocations.mkdir(parents=True, exist_ok=True)
    performance.mkdir(parents=True, exist_ok=True)
    (allocations / "config.json").write_text(json.dumps({
        "source_csv": str(args.source_csv), "panel": str(panel),
        "start_date": start, "end_date": end, "lookback_months": lookback,
        "top_n": ntop, "transaction_cost_bps": tc_bps, "allow_cash": True,
        "signals": {"ret_div_eq": ["monthly return", "price / 200-day SMA - 1"],
                    "ret_rvol_eq": ["monthly return", "monthly volume / trailing 12-month volume SMA"]},
        "signal_weights": [0.5, 0.5],
        "n_values": {"ret_div_eq": [1], "ret_rvol_eq": [1]},
        "covariance_implementation": args.cov_impl,
        "optimizer_policy": "guarded July MGS-family refresh",
    }, indent=2))

    for tag, second in (("ret_div_eq", "div"), ("ret_rvol_eq", "rvol")):
        cov_func = make_cov_func(modified_gerber_stat, frames, second)
        for n in (1,):
            name = f"modified_gerber_stat-{tag}-w=50_50-g=1_0-n={n}_0-h=inf"
            allocator = SP500PortfolioAllocator(
                cov_func=cov_func, cov_params={"gamma": 1.0, "n": float(n)}, cov_name=name,
                data_path=str(panel), begin_date=start, end_date=end, ntop=ntop,
                lookback=lookback, freq="monthly", save_path=str(allocations),
                use_cash=True, rf_cache_dir=str(args.rf_cache_dir),
                optimizer_cls=GuardedOptimizer,
            )
            allocator.allocate_weights(verbose=False)
            missing = [date.strftime("%Y%m%d") for date in allocator.dates
                       if not (allocations / name / f"{date:%Y%m%d}.pkl").is_file()]
            if missing:
                raise RuntimeError(f"{name} is missing {len(missing)} allocations: {missing[:10]}")
            run_sp500_backtest(
                cov_name=name, results_folder=str(allocations), performance_prefix=str(performance),
                data_path=str(panel), begin_date=start, end_date=end, tc_cost=tc_bps,
                rp_min=0.03, rp_max=0.21, rp_step=0.03,
                feat="Geometric Return (%)", verbose=False,
            )
            print(f"Completed {name}")
    print(f"Performance files: {performance / f'tc{tc_bps}'}")


if __name__ == "__main__":
    main()
