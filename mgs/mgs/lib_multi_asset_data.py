from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MULTI_ASSET_SIGNAL_FREQS = ("weekly", "monthly")
ANNUALIZATION_FACTORS = {"daily": 252, "weekly": 52, "monthly": 12}


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
        raise ValueError(
            f"Unsupported signal frequency {freq!r}. Choose from {tuple(aliases.values())}."
        ) from exc


def annualization_factor(freq: str) -> int:
    return ANNUALIZATION_FACTORS[normalize_signal_frequency(freq)]


def trailing_window_start(end_date, lookback_months: int) -> pd.Timestamp:
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive.")
    return pd.Timestamp(end_date).normalize() - pd.DateOffset(months=lookback_months)


def _normalize_excel_ticker(value) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("Encountered an empty ticker cell in the Excel header.")
    return text.split()[0]


def load_multi_asset_price_history(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path).expanduser().resolve()
    suffix = data_path.suffix.lower()

    if suffix == ".csv":
        df_prices = pd.read_csv(data_path, parse_dates=["Date"])
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(data_path, header=None)
        if raw.shape[0] < 7 or raw.shape[1] < 2:
            raise ValueError(f"Excel file {data_path} does not match the expected Bloomberg-style layout.")

        tickers = [_normalize_excel_ticker(value) for value in raw.iloc[3, 1:].tolist()]
        df_prices = raw.iloc[6:, : len(tickers) + 1].copy()
        df_prices.columns = ["Date", *tickers]
    else:
        raise ValueError(f"Unsupported data file type {data_path.suffix!r}. Use CSV or Excel.")

    if "Date" not in df_prices.columns:
        raise ValueError(f"Input file {data_path} must contain a Date column.")

    df_prices["Date"] = pd.to_datetime(df_prices["Date"])
    for column in df_prices.columns[1:]:
        df_prices[column] = pd.to_numeric(df_prices[column], errors="coerce")

    df_prices = df_prices.dropna(subset=["Date"]).sort_values("Date")
    df_prices = df_prices.groupby("Date", as_index=False).last()
    df_prices = df_prices.set_index("Date").sort_index()
    df_prices.index = pd.to_datetime(df_prices.index).normalize()
    return df_prices.dropna(how="all")


def infer_observation_spacing(df_prices: pd.DataFrame) -> str:
    dates = pd.Index(df_prices.index).sort_values().unique()
    if len(dates) < 3:
        return "unknown"

    deltas = dates.to_series().diff().dropna().dt.days
    median_gap = float(deltas.median())
    if median_gap <= 3.0:
        return "daily"
    if median_gap <= 10.0:
        return "weekly"
    return "monthly"


def aggregate_multi_asset_prices(df_prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq = normalize_signal_frequency(freq)
    if freq == "daily":
        return df_prices.copy()

    index = pd.to_datetime(df_prices.index).normalize()
    if freq == "weekly":
        period_buckets = index.to_series(index=index).dt.to_period("W-FRI")
    elif freq == "monthly":
        period_buckets = index.to_series(index=index).dt.to_period("M")
    else:
        raise ValueError(f"Unsupported multi-asset frequency {freq!r}.")

    last_dates = period_buckets.groupby(period_buckets).apply(lambda values: values.index.max())
    aggregated = df_prices.assign(_period_bucket=period_buckets.values).groupby("_period_bucket", sort=True).last()
    aggregated.index = pd.to_datetime(last_dates.loc[aggregated.index].to_numpy())
    aggregated.index.name = "Date"
    return aggregated.sort_index()


def price_history_to_returns(df_prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    aggregated_prices = aggregate_multi_asset_prices(df_prices=df_prices, freq=freq)
    return aggregated_prices.pct_change(fill_method=None).dropna(how="all")


def build_multi_asset_divergence_signal(
    df_prices: pd.DataFrame,
    freq: str,
    sma_window: int = 200,
) -> pd.DataFrame:
    if sma_window <= 0:
        raise ValueError("sma_window must be positive.")

    daily_sma = df_prices.rolling(window=sma_window, min_periods=sma_window).mean()
    aggregated_prices = aggregate_multi_asset_prices(df_prices=df_prices, freq=freq)
    aggregated_sma = aggregate_multi_asset_prices(df_prices=daily_sma, freq=freq)
    divergence = aggregated_prices.divide(aggregated_sma.replace(0.0, np.nan)) - 1.0
    divergence = divergence.replace([np.inf, -np.inf], np.nan)
    divergence.index.name = "Date"
    return divergence.sort_index()


def build_multi_asset_return_window(
    df_prices: pd.DataFrame,
    end_date,
    lookback_months: int,
    freq: str,
) -> pd.DataFrame:
    df_returns = price_history_to_returns(df_prices=df_prices, freq=freq)
    end_date = pd.Timestamp(end_date).normalize()
    window_start = trailing_window_start(end_date=end_date, lookback_months=lookback_months)
    return df_returns[(df_returns.index > window_start) & (df_returns.index <= end_date)]
