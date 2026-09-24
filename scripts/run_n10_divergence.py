#!/usr/bin/env python3
"""Run the paper's 50/50 return plus four-week divergence strategy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mgs"))

import mgs.lib_port_multi_assets_allocator as allocator_module
import mgs.lib_portfolio_analyzer as analyzer_module
from mgs.lib_multi_asset_data import (aggregate_multi_asset_prices,
                                      load_multi_asset_price_history, price_history_to_returns)
from mgs.lib_port_multi_assets_allocator import PortfolioAllocator
from mgs.lib_port_multi_assets_backtester import run_backtest


def build_cov_func(modified_gerber_stat, data_path: Path):
    prices = load_multi_asset_price_history(data_path)
    weekly_prices = aggregate_multi_asset_prices(prices, freq="weekly")
    weekly_returns = price_history_to_returns(prices, freq="weekly")
    sma = weekly_prices.rolling(4, min_periods=4).mean()
    divergence = weekly_prices.divide(sma.replace(0.0, np.nan)) - 1.0
    divergence = divergence.replace([np.inf, -np.inf], np.nan)

    def cov_func(return_window: pd.DataFrame, gamma: float = 1.0, n: float = 1.0, half_life=None):
        frames = [
            weekly_returns.reindex(index=return_window.index, columns=return_window.columns).fillna(0.0),
            divergence.reindex(index=return_window.index, columns=return_window.columns).fillna(0.0),
        ]
        return modified_gerber_stat(
            frames, gamma=gamma, n=n, half_life=half_life,
            signal_weights=(0.5, 0.5), volatility_source=return_window,
        )
    return cov_func


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/n10_daily.xlsx")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/n10_divergence")
    parser.add_argument("--rf-cache-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cov-impl", choices=("cpp", "python"), default="cpp")
    args = parser.parse_args()

    if args.cov_impl == "cpp":
        from mgs.lib_cov_func_cpp import modified_gerber_stat
    else:
        from mgs.lib_cov_func import modified_gerber_stat
    allocator_module.rf_cache_path = str(args.rf_cache_dir)
    analyzer_module.rf_cache_path = str(args.rf_cache_dir)
    start, end, lookback, tc_bps = "1990-01-31", "2025-12-31", 36, 10
    run_name = (f"{args.data_path.stem}_weekly_sma_windows=04_start={start}_"
                f"end={end}_tc={tc_bps}_lbm={lookback}_cash")
    allocations = args.results_root / "allocations" / run_name
    performance = args.results_root / "performance" / run_name
    allocations.mkdir(parents=True, exist_ok=True)
    performance.mkdir(parents=True, exist_ok=True)
    (allocations / "config.json").write_text(json.dumps({
        "data_path": str(args.data_path), "start_date": start, "end_date": end,
        "lookback_months": lookback, "signal_frequency": "weekly",
        "rebalance_frequency": "monthly", "transaction_cost_bps": tc_bps,
        "allow_cash": True, "divergence": "weekly close / trailing 4-week SMA - 1",
        "signal_weights": [0.5, 0.5], "gamma": 1, "n": 1,
        "covariance_implementation": args.cov_impl,
    }, indent=2))
    name = "modified_gerber_stat-ret_div04w_eq-w=50_50-g=1_0-n=1_0-h=inf"
    allocator = PortfolioAllocator(
        cov_func=build_cov_func(modified_gerber_stat, args.data_path),
        cov_params={"gamma": 1.0, "n": 1.0}, cov_name=name,
        data_path=str(args.data_path), save_path=str(allocations), lookback=lookback,
        start_date=start, end_date=end, allow_cash=True, freq="weekly", reuse_existing=True,
    )
    allocator.allocate_weights(verbose=False)
    run_backtest(
        cov_name=name, results_folder=str(allocations), performance_prefix=str(performance),
        tc_cost=tc_bps, start_date=start, end_date=end, freq="monthly", reuse_existing=True,
    )
    print(f"Performance files: {performance / f'tc{tc_bps}' / name}")


if __name__ == "__main__":
    main()
