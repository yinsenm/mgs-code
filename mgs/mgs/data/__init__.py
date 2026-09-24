"""Data helpers for package-local research datasets."""

from .sp500 import (
    SP500_SIGNAL_FREQS,
    aggregate_sp500_daily_data,
    annualization_factor,
    build_signal_panel,
    normalize_signal_frequency,
    trailing_window_start,
)
from .multi_asset import (
    MULTI_ASSET_SIGNAL_FREQS,
    aggregate_multi_asset_prices,
    build_multi_asset_return_window,
    infer_observation_spacing,
    load_multi_asset_price_history,
    price_history_to_returns,
)

__all__ = [
    "MULTI_ASSET_SIGNAL_FREQS",
    "SP500_SIGNAL_FREQS",
    "aggregate_multi_asset_prices",
    "aggregate_sp500_daily_data",
    "annualization_factor",
    "build_multi_asset_return_window",
    "build_signal_panel",
    "infer_observation_spacing",
    "load_multi_asset_price_history",
    "normalize_signal_frequency",
    "price_history_to_returns",
    "trailing_window_start",
]
