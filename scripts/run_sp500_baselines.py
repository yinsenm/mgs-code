#!/usr/bin/env python3
"""Run the paper's top-30 S&P 500 return-only covariance comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mgs"))

from mgs.lib_cov_func import ledoit
from mgs.lib_port_sp500_allocator import SP500PortfolioAllocator
from mgs.lib_port_sp500_backtester import run_sp500_backtest
from mgs.lib_port_strategy_cvx import PortfolioStrategyCVX as PaperEraOptimizer
from mgs.lib_port_strategy_cvx_guarded import PortfolioStrategyCVX as GuardedOptimizer
from mgs.run_sp500_weekly_features import prepare_weekly_sp500_panel


def get_methods(cov_impl: str):
    if cov_impl == "cpp":
        from mgs.lib_cov_func_cpp import gerber_cov_stat1, modified_gerber_stat
    else:
        from mgs.lib_cov_func import gerber_cov_stat1, modified_gerber_stat
    methods = {
        "HC": (lambda returns: returns.cov(), {}),
        "SM": (ledoit, {}),
        "GS1-ts=0_5": (gerber_cov_stat1, {"threshold": 0.5, "center_method": "zero"}),
    }
    for n in (1,):
        methods[f"modified_gerber_stat-g=1_0-n={n}_0-h=inf"] = (
            modified_gerber_stat, {"gamma": 1.0, "n": float(n)})
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=ROOT / "data/sp500_1950_2025/sp500_weekly_features.csv")
    parser.add_argument("--prepared-data-path", type=Path, default=ROOT / "data/sp500_1950_2025/sp500_weekly_panel")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/sp500_baselines")
    parser.add_argument("--rf-cache-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cov-impl", choices=("cpp", "python"), default="cpp")
    args = parser.parse_args()

    panel = Path(prepare_weekly_sp500_panel(str(args.source_csv), str(args.prepared_data_path)))
    start, end, lookback, ntop, tc_bps = "1990-01-31", "2025-12-31", 60, 30, 10
    run_name = (f"{panel.name}_ntop={ntop}_start={start}_end={end}_tc={tc_bps}_"
                f"lbm={lookback}_freq=monthly_cash=1")
    allocations = args.results_root / "allocations" / run_name
    performance = args.results_root / "performance" / f"{run_name}_rp=03-21-03"
    allocations.mkdir(parents=True, exist_ok=True)
    performance.mkdir(parents=True, exist_ok=True)
    methods = get_methods(args.cov_impl)
    (allocations / "config.json").write_text(json.dumps({
        "source_csv": str(args.source_csv), "panel": str(panel),
        "start_date": start, "end_date": end, "lookback_months": lookback,
        "top_n": ntop, "signal_frequency": "monthly", "rebalance_frequency": "monthly",
        "transaction_cost_bps": tc_bps, "allow_cash": True,
        "covariance_implementation": args.cov_impl, "methods": list(methods),
        "optimizer_policy": "HC/SM/GS: paper-era; MGS: guarded July refresh",
    }, indent=2))

    for name, (cov_func, cov_params) in methods.items():
        optimizer_cls = GuardedOptimizer if name.startswith("modified_gerber_stat") else PaperEraOptimizer
        allocator = SP500PortfolioAllocator(
            cov_func=cov_func, cov_params=cov_params, cov_name=name,
            data_path=str(panel), begin_date=start, end_date=end, ntop=ntop,
            lookback=lookback, freq="monthly", save_path=str(allocations),
            use_cash=True, rf_cache_dir=str(args.rf_cache_dir),
            optimizer_cls=optimizer_cls,
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
