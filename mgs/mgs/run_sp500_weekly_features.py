"""Prepare the weekly stock source into the panel consumed by the paper backtest."""

import json
import os

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ("date", "permno", "adjusted_ret", "sp500_market_cap_rank")


def _compound_return(values: pd.Series) -> float:
    observed = values.dropna()
    return np.nan if observed.empty else (1.0 + observed).prod() - 1.0


def _last_observation(values: pd.Series) -> float:
    return np.nan if values.empty else values.iloc[-1]


def prepare_weekly_sp500_panel(source_csv: str, output_dir: str) -> str:
    source_csv = os.path.abspath(source_csv)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isfile(source_csv):
        raise FileNotFoundError(f"Weekly S&P 500 source file not found: {source_csv}")

    os.makedirs(output_dir, exist_ok=True)
    day_rets_path = os.path.join(output_dir, "day_rets.csv")
    mon_rets_path = os.path.join(output_dir, "mon_rets.csv")
    metadata_path = os.path.join(output_dir, "sp500_weekly_panel_metadata.json")
    if all(os.path.isfile(path) for path in (day_rets_path, mon_rets_path, metadata_path)):
        try:
            with open(metadata_path) as handle:
                metadata = json.load(handle)
            if (os.path.abspath(metadata.get("source_csv", "")) == source_csv
                    and min(os.path.getmtime(day_rets_path), os.path.getmtime(mon_rets_path))
                    >= os.path.getmtime(source_csv)):
                return output_dir
        except (OSError, ValueError):
            pass

    weekly = pd.read_csv(source_csv, usecols=list(SOURCE_COLUMNS), parse_dates=["date"])
    weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    weekly.sort_values(["date", "permno"], inplace=True)
    day_rets = (weekly.rename(columns={"adjusted_ret": "ret"})[["date", "permno", "ret"]]
                .dropna(subset=["ret"]).reset_index(drop=True))
    monthly = weekly.assign(month_end=weekly["date"].dt.to_period("M").dt.to_timestamp("M"))
    mon_rets = (monthly.groupby(["month_end", "permno"], observed=True, sort=True)
                .agg(ret=("adjusted_ret", _compound_return),
                     market_cap_rank=("sp500_market_cap_rank", _last_observation))
                .reset_index().rename(columns={"month_end": "date"}))
    mon_rets["market_cap"] = np.where(
        mon_rets["market_cap_rank"].notna() & (mon_rets["market_cap_rank"] > 0),
        1.0 / mon_rets["market_cap_rank"], np.nan)
    mon_rets = mon_rets[["date", "permno", "ret", "market_cap", "market_cap_rank"]]
    day_rets.to_csv(day_rets_path, index=False)
    mon_rets.to_csv(mon_rets_path, index=False)
    with open(metadata_path, "w") as handle:
        json.dump({
            "source_csv": source_csv,
            "source_frequency": "weekly",
            "n_rows_weekly": int(len(day_rets)),
            "n_rows_monthly": int(len(mon_rets)),
            "n_assets": int(weekly["permno"].nunique()),
            "min_date": str(weekly["date"].min().date()),
            "max_date": str(weekly["date"].max().date()),
        }, handle, indent=2)
    return output_dir
