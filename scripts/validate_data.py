#!/usr/bin/env python3
"""Read-only checks for the unpublished data inputs required by the paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mgs"))
from mgs.lib_multi_asset_data import load_multi_asset_price_history


N10_ASSETS = {"SPX", "RTY", "MXEA", "MXEF", "LBUSTRUU", "LF98TRUU",
              "FNERTR", "SPGSCI", "XAU", "NDX"}
STOCK_COLUMNS = {"date", "permno", "adjusted_ret", "adjusted_close",
                 "adjusted_volume", "adjusted_close_200sma", "sp500_market_cap_rank"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--study", choices=("n10", "sp500", "all"), default="all")
    args = parser.parse_args()
    data_dir = args.data_dir

    if args.study in ("n10", "all"):
        prices = load_multi_asset_price_history(data_dir / "n10_daily.xlsx")
        if set(prices.columns) != N10_ASSETS:
            raise ValueError(f"10-asset columns differ; got {list(prices.columns)}")
        if prices.index.min() > pd.Timestamp("1987-12-31") or prices.index.max() < pd.Timestamp("2025-12-31"):
            raise ValueError("10-asset panel must cover 1987-12-31 through 2025-12-31")
        print(f"10 assets: {len(prices)} dates, {prices.index.min().date()} to {prices.index.max().date()}")

    if args.study in ("sp500", "all"):
        source = data_dir / "sp500_1950_2025/sp500_weekly_features.csv"
        header = set(pd.read_csv(source, nrows=0).columns)
        if not STOCK_COLUMNS.issubset(header):
            raise ValueError(f"Stock file is missing: {sorted(STOCK_COLUMNS - header)}")
        stock_dates = pd.read_csv(source, usecols=["date"], parse_dates=["date"])["date"]
        if stock_dates.min() > pd.Timestamp("1985-01-01") or stock_dates.max() < pd.Timestamp("2025-12-01"):
            raise ValueError("Stock data need pre-1990 lookback history and coverage through December 2025")
        print(f"Stock observations: {len(stock_dates):,}, {stock_dates.min().date()} to {stock_dates.max().date()}")

    rf = data_dir / "tb3ms_19900101_20251231.csv"
    if rf.exists():
        tbills = pd.read_csv(rf, parse_dates=["date"])
        if not {"date", "TB3MS"}.issubset(tbills.columns):
            raise ValueError("TB3MS cache needs date and TB3MS columns")
        if tbills["date"].min() > pd.Timestamp("1990-01-31") or tbills["date"].max() < pd.Timestamp("2025-12-31"):
            raise ValueError("TB3MS cache must cover January 1990 through December 2025")
        print(f"TB3MS cache: {len(tbills)} months")
    else:
        print("TB3MS cache missing: backtests will download and cache it from FRED")


if __name__ == "__main__":
    main()
