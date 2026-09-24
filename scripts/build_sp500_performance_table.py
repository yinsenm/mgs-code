#!/usr/bin/env python3
"""Build the main S&P 500 performance table from completed backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_SUFFIX = "ntop=30_start=1990-01-31_end=2025-12-31_tc=10_lbm=60_freq=monthly_cash=1_rp=03-21-03"
RISKS = ("03%", "06%", "09%", "12%", "15%", "18%", "21%")
METRICS = (
    "Geometric Return (%)", "Annualized STD (%)", "Annualized Sharpe Ratio",
    "Cumulative Return (%)", "Maximum Drawdown (%)", "Turnover (%)",
)
METHODS = (
    ("MGS+Div", "signals", "modified_gerber_stat-ret_div_eq-w=50_50-g=1_0-n=1_0-h=inf"),
    ("MGS+RVOL", "signals", "modified_gerber_stat-ret_rvol_eq-w=50_50-g=1_0-n=1_0-h=inf"),
    ("MGS", "baselines", "modified_gerber_stat-g=1_0-n=1_0-h=inf"),
    ("GS", "baselines", "GS1-ts=0_5"),
    ("SM", "baselines", "SM"),
    ("HC", "baselines", "HC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/sp500_baselines")
    parser.add_argument("--signal-root", type=Path, default=ROOT / "results/sp500_signals")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--panel-name", default="sp500_weekly_panel")
    return parser.parse_args()


def make_table(args: argparse.Namespace) -> pd.DataFrame:
    run_name = f"{args.panel_name}_{RUN_SUFFIX}"
    roots = {
        "baselines": args.baseline_root / "performance" / run_name / "tc10",
        "signals": args.signal_root / "performance" / run_name / "tc10",
    }
    frames = {
        label: pd.read_csv(roots[group] / folder / "portfolios_performance.csv", index_col=0)
        for label, group, folder in METHODS
    }
    rows = []
    for risk in RISKS:
        for label, _, _ in METHODS:
            values = [float(frames[label].loc[metric, risk]) for metric in METRICS]
            rows.append((risk, label, *values))
    return pd.DataFrame(
        rows,
        columns=("Risk target", "Method", "Geom. return", "Volatility", "Sharpe",
                 "Cum. return", "Max drawdown", "Turnover"),
    )


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        (r"\caption{Out-of-sample performance for the point-in-time S\&P 500 top-30 study, "
         r"January 1990--December 2025. Portfolios use a trailing 60-month signal window, "
         r"monthly rebalancing, residual cash, and 10 bps transaction costs. "
         r"MGS+Div and MGS+RVOL combine monthly returns with price/200-day-SMA divergence "
         r"and 12-month relative volume, respectively, at equal signal weights. "
         r"All MGS specifications use $\gamma=n=1$ and no time decay. "
         r"Return, volatility, and Sharpe are annualized; turnover is average monthly turnover.}"),
        r"\label{tab:sp500_multisignal_summary}", r"\begin{tabular}{rlrrrrrr}", r"\toprule",
        "Risk target & Method & Geom. return & Volatility & Sharpe & Cum. return & Max drawdown & Turnover \\\\",
        r"\midrule",
    ]
    groups = list(table.groupby("Risk target", sort=False))
    for group_index, (risk, block) in enumerate(groups):
        for row_index, (_, row) in enumerate(block.iterrows()):
            risk_cell = f"\\multirow{{{len(block)}}}{{*}}{{{int(risk[:-1])}\\%}}" if row_index == 0 else ""
            lines.append(
                f"{risk_cell} & {row['Method']} & {row['Geom. return']:.2f} & "
                f"{row['Volatility']:.2f} & {row['Sharpe']:.2f} & {row['Cum. return']:.2f} & "
                f"{row['Max drawdown']:.2f} & {row['Turnover']:.2f} \\\\"
            )
        if group_index < len(groups) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = make_table(args)
    csv_path = args.output_dir / "table_large_cap_multisignal_performance.csv"
    tex_path = args.output_dir / "table_large_cap_multisignal_performance.tex"
    table.to_csv(csv_path, index=False, float_format="%.8f")
    tex_path.write_text(to_latex(table))
    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
