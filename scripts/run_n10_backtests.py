#!/usr/bin/env python3
"""Run the paper's 10-asset return-only covariance comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mgs"))

import mgs.lib_port_multi_assets_allocator as allocator_module
import mgs.lib_portfolio_analyzer as analyzer_module
from mgs.lib_port_multi_assets_allocator import PortfolioAllocator
from mgs.lib_port_multi_assets_backtester import run_backtest
from mgs.paper_methods import get_paper_methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/n10_daily.xlsx")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/n10")
    parser.add_argument("--rf-cache-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cov-impl", choices=("cpp", "python"), default="cpp")
    args = parser.parse_args()

    start, end, lookback, tc_bps = "1990-01-31", "2025-12-31", 36, 10
    run_name = f"{args.data_path.stem}_freq=weekly_start={start}_end={end}_tc={tc_bps}_lbm={lookback}_cash"
    allocations = args.results_root / "allocations" / run_name
    performance = args.results_root / "performance" / run_name
    allocations.mkdir(parents=True, exist_ok=True)
    performance.mkdir(parents=True, exist_ok=True)
    allocator_module.rf_cache_path = str(args.rf_cache_dir)
    analyzer_module.rf_cache_path = str(args.rf_cache_dir)
    methods = get_paper_methods(args.cov_impl, [1])
    (allocations / "config.json").write_text(json.dumps({
        "data_path": str(args.data_path), "start_date": start, "end_date": end,
        "lookback_months": lookback, "signal_frequency": "weekly",
        "rebalance_frequency": "monthly", "transaction_cost_bps": tc_bps,
        "allow_cash": True, "covariance_implementation": args.cov_impl,
        "methods": list(methods),
    }, indent=2))

    for name, spec in methods.items():
        allocator = PortfolioAllocator(
            cov_func=spec["cov_func"], cov_params=spec["cov_params"],
            cov_name=name, data_path=str(args.data_path), save_path=str(allocations),
            lookback=lookback, start_date=start, end_date=end, allow_cash=True,
            freq="weekly", reuse_existing=True,
        )
        allocator.allocate_weights(verbose=False)
        run_backtest(
            cov_name=name, results_folder=str(allocations),
            performance_prefix=str(performance), tc_cost=tc_bps,
            start_date=start, end_date=end, freq="monthly", reuse_existing=True,
        )
        print(f"Completed {name}")
    print(f"Performance files: {performance / f'tc{tc_bps}'}")


if __name__ == "__main__":
    main()
