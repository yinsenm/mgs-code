from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

SP500_SIGNAL_FREQS = ("daily", "weekly", "monthly")
ANNUALIZATION_FACTORS = {"daily": 252, "weekly": 52, "monthly": 12}
DEFAULT_AGGREGATION_METHODS = {
    "ret": "compound_return",
    "adj_ret": "compound_return",
    "adjusted_return": "compound_return",
    "adj_price": "last",
    "adjusted_price": "last",
    "price": "last",
    "market_cap": "last",
    "volume": "sum",
}


def normalize_signal_frequency(freq: str) -> str:
    freq_key = freq.strip().lower()
    aliases = {
        "d": "daily",
        "day": "daily",
        "daily": "daily",
        "w": "weekly",
        "week": "weekly",
        "weekly": "weekly",
        "m": "monthly",
        "month": "monthly",
        "monthly": "monthly",
    }
    try:
        return aliases[freq_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported signal frequency {freq!r}. Choose from {SP500_SIGNAL_FREQS}.") from exc


def annualization_factor(freq: str) -> int:
    return ANNUALIZATION_FACTORS[normalize_signal_frequency(freq)]


def trailing_window_start(end_date, lookback_months: int) -> pd.Timestamp:
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive.")
    return pd.Timestamp(end_date).normalize() - pd.DateOffset(months=lookback_months)


def _compound_return(values: pd.Series) -> float:
    non_missing = values.dropna()
    if non_missing.empty:
        return np.nan
    return (1.0 + non_missing).prod() - 1.0


def _period_buckets(dates: pd.Series, freq: str) -> pd.Series:
    freq = normalize_signal_frequency(freq)
    normalized = pd.to_datetime(dates).dt.normalize()
    if freq == "daily":
        return normalized
    if freq == "weekly":
        # The historical file contains Saturday sessions through 1952-05-24,
        # so we anchor weekly bars to Saturday instead of Friday.
        return normalized.dt.to_period("W-SAT")
    return normalized.dt.to_period("M")


def aggregate_sp500_daily_data(
    df_daily: pd.DataFrame,
    freq: str,
    *,
    asset_col: str = "permno",
    date_col: str = "date",
    column_methods: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    freq = normalize_signal_frequency(freq)
    df_daily = df_daily.copy()
    df_daily[date_col] = pd.to_datetime(df_daily[date_col]).dt.normalize()
    df_daily.sort_values([date_col, asset_col], inplace=True)

    if freq == "daily":
        return df_daily.reset_index(drop=True)

    value_columns = [column for column in df_daily.columns if column not in {date_col, asset_col}]
    if not value_columns:
        raise ValueError("No value columns found to aggregate.")

    rules = dict(DEFAULT_AGGREGATION_METHODS)
    if column_methods is not None:
        rules.update(column_methods)

    unknown_columns = [column for column in value_columns if column not in rules]
    if unknown_columns:
        raise ValueError(
            "Missing aggregation rules for columns "
            f"{unknown_columns}. Pass column_methods explicitly for those fields."
        )

    df_grouped = df_daily.assign(_period_bucket=_period_buckets(df_daily[date_col], freq))
    period_labels = df_grouped.groupby("_period_bucket", sort=True)[date_col].max()
    grouped = df_grouped.groupby(
        ["_period_bucket", asset_col],
        observed=True,
        sort=True,
    )

    aggregated_columns = {}
    for column in value_columns:
        method = rules[column]
        if method == "compound_return":
            aggregated_columns[column] = grouped[column].apply(_compound_return)
        elif method == "last":
            aggregated_columns[column] = grouped[column].last()
        elif method == "sum":
            aggregated_columns[column] = grouped[column].agg(lambda values: values.sum(min_count=1))
        else:
            raise ValueError(f"Unsupported aggregation method {method!r} for column {column!r}.")

    aggregated = pd.concat(aggregated_columns, axis=1).reset_index()
    aggregated[date_col] = aggregated["_period_bucket"].map(period_labels)
    aggregated.drop(columns="_period_bucket", inplace=True)
    aggregated.sort_values([date_col, asset_col], inplace=True)
    return aggregated.reset_index(drop=True)


def build_signal_panel(
    df_daily: pd.DataFrame,
    end_date,
    lookback_months: int,
    freq: str,
    *,
    asset_col: str = "permno",
    date_col: str = "date",
    return_col: str = "ret",
) -> pd.DataFrame:
    signal_returns = aggregate_sp500_daily_data(
        df_daily=df_daily,
        freq=freq,
        asset_col=asset_col,
        date_col=date_col,
        column_methods={return_col: "compound_return"},
    )
    end_date = pd.Timestamp(end_date).normalize()
    window_start = trailing_window_start(end_date=end_date, lookback_months=lookback_months)
    df_window = signal_returns[
        (signal_returns[date_col] > window_start) & (signal_returns[date_col] <= end_date)
    ]
    if df_window.empty:
        return pd.DataFrame()
    return df_window.pivot(index=date_col, columns=asset_col, values=return_col).sort_index()
