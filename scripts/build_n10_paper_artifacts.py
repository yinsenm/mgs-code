#!/usr/bin/env python3
"""Build the 10-asset main table from the two paper backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RISKS = ["03%", "06%", "09%", "12%", "15%"]
METRICS = (
    "Geometric Return (%)", "Annualized STD (%)", "Annualized Sharpe Ratio",
    "Cumulative Return (%)", "Maximum Drawdown (%)", "Turnover (%)",
)
METHODS = (
    ("MGS+Div", "div", "modified_gerber_stat-ret_div04w_eq-w=50_50-g=1_0-n=1_0-h=inf"),
    ("MGS", "baseline", "modified_gerber_stat-g=1_0-n=1_0-h=inf"),
    ("GS", "baseline", "GS1-ts=0_5"),
    ("SM", "baseline", "SM"),
    ("HC", "baseline", "HC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/n10")
    parser.add_argument("--div-root", type=Path, default=ROOT / "results/n10_divergence")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--data-stem", default="n10_daily")
    parser.add_argument("--start-date", default="1990-01-31")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--tc-bps", type=int, default=10)
    parser.add_argument("--lookback-months", type=int, default=36)
    parser.add_argument("--div-windows-tag", default="04", help="Output-folder tag from the divergence run")
    return parser.parse_args()


def method_roots(args: argparse.Namespace) -> dict[str, Path]:
    suffix = f"start={args.start_date}_end={args.end_date}_tc={args.tc_bps}_lbm={args.lookback_months}_cash"
    return {
        "baseline": args.baseline_root / "performance" / f"{args.data_stem}_freq=weekly_{suffix}" / f"tc{args.tc_bps}",
        "div": args.div_root / "performance" / f"{args.data_stem}_weekly_sma_windows={args.div_windows_tag}_{suffix}" / f"tc{args.tc_bps}",
    }


def make_table(args: argparse.Namespace) -> pd.DataFrame:
    roots = method_roots(args)
    frames = {
        label: pd.read_csv(roots[group] / folder / "portfolios_performance.csv", index_col=0)
        for label, group, folder in METHODS
    }
    rows = []
    for risk in RISKS:
        for label, _, _ in METHODS:
            values = [float(frames[label].loc[metric, risk]) for metric in METRICS]
            values[-1] /= 12.0  # The multi-asset analyzer stores annual turnover.
            rows.append((risk, label, *values))
    return pd.DataFrame(rows, columns=("Risk target", "Method", "Geom. return", "Volatility", "Sharpe", "Cum. return", "Max drawdown", "Turnover"))


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        (r"\caption{Out-of-sample performance for the 10-asset study, January 1990--December 2025. "
         r"Covariance inputs use weekly observations and a trailing window of up to 36 months; "
         r"portfolios rebalance monthly, allow residual cash, and pay 10 bps transaction costs. "
         r"MGS+Div combines return and four-week price/SMA divergence evidence equally, with $\gamma=n=1$. "
         r"Return, volatility, and Sharpe are annualized; turnover is average monthly turnover.}"),
        r"\label{tab:n10-weekly-multisignal-summary}", r"\begin{tabular}{rlrrrrrr}", r"\toprule",
        r"Risk target & Method & Geom. return & Volatility & Sharpe & Cum. return & Max drawdown & Turnover \\",
        r"\midrule",
    ]
    for risk_index, (risk, block) in enumerate(table.groupby("Risk target", sort=False)):
        for row_index, (_, row) in enumerate(block.iterrows()):
            risk_cell = f"\\multirow{{{len(block)}}}{{*}}{{{int(risk[:-1])}\\%}}" if row_index == 0 else ""
            lines.append(
                f"{risk_cell} & {row['Method']} & {row['Geom. return']:.2f} & "
                f"{row['Volatility']:.2f} & {row['Sharpe']:.2f} & {row['Cum. return']:.2f} & "
                f"{row['Max drawdown']:.2f} & {row['Turnover']:.2f} \\\\"
            )
        if risk_index != len(RISKS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = make_table(args)
    csv_path = args.output_dir / "table_multi_asset_performance.csv"
    tex_path = args.output_dir / "table_multi_asset_performance.tex"
    table.to_csv(csv_path, index=False, float_format="%.8f")
    tex_path.write_text(to_latex(table))
    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
