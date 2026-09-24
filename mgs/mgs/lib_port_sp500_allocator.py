import os
import pickle
from glob import glob
from math import ceil, floor, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm

from .data import (
    aggregate_sp500_daily_data,
    annualization_factor,
    normalize_signal_frequency,
    trailing_window_start,
)
from .lib_cov_func import correlation_from_covariance, is_psd_def
from .lib_port_strategy_cvx_guarded import PortfolioStrategyCVX
from .nearest_correlation import nearcorr

DEFAULT_SP500_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sp500_top500"
DEFAULT_RF_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def compute_portfolio_return_and_volatility(
    weights,
    return_vector,
    covariance_matrix,
    annualize_factor=252,
    cash_weight=0.0,
    cash_return=0.0,
):
    weights = np.array(weights)
    return_vector = np.array(return_vector)
    covariance_matrix = np.array(covariance_matrix)

    portfolio_return = (np.dot(weights, return_vector) + cash_weight * cash_return) * annualize_factor
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights)) * annualize_factor
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_return.item(), portfolio_volatility.item()


class SP500PortfolioAllocator:
    _returns_cache = {}
    _signal_cache = {}

    def __init__(
        self,
        cov_func=None,
        cov_name=None,
        cov_params=None,
        data_path=DEFAULT_SP500_DATA_PATH,
        begin_date="2015-01-31",
        end_date="2024-12-31",
        ntop=50,
        lookback=48,
        freq="daily",
        save_path="../results/sp500_example/allocations",
        use_cash=False,
        rf_cache_dir=DEFAULT_RF_CACHE_DIR,
        optimizer_cls=None,
    ):
        self.rf_benchmark = "TB3MS"
        self.data_path = os.path.abspath(data_path)
        self.lookback_months = lookback
        self.freq = normalize_signal_frequency(freq)
        self.annual_factor = annualization_factor(self.freq)
        self.ntop = ntop
        self.use_cash = use_cash
        self.rf_cache_dir = os.path.abspath(rf_cache_dir)
        self.optimizer_cls = optimizer_cls or PortfolioStrategyCVX

        returns_cache_key = self.data_path
        if returns_cache_key not in self._returns_cache:
            df_mons_rets = pd.read_csv(
                os.path.join(self.data_path, "mon_rets.csv"),
                parse_dates=["date"],
                index_col="date",
            )
            df_days_rets = pd.read_csv(
                os.path.join(self.data_path, "day_rets.csv"),
                parse_dates=["date"],
                index_col="date",
            ).sort_index()
            self._returns_cache[returns_cache_key] = (df_mons_rets, df_days_rets)
        self.df_mons_rets, self.df_days_rets = self._returns_cache[returns_cache_key]

        signal_cache_key = (self.data_path, self.freq)
        if signal_cache_key not in self._signal_cache:
            self._signal_cache[signal_cache_key] = (
                aggregate_sp500_daily_data(
                    df_daily=self.df_days_rets.reset_index(),
                    freq=self.freq,
                    column_methods={"ret": "compound_return"},
                )
                .set_index("date")
                .sort_index()
            )
        self.df_signal_rets = self._signal_cache[signal_cache_key]

        self.dates = sorted(self.df_mons_rets.loc[begin_date:end_date].index.drop_duplicates())
        if not self.dates:
            raise ValueError(
                f"No monthly S&P 500 observations found between {begin_date} and {end_date} "
                f"in {self.data_path}."
            )

        self.begin_date = self.dates[0]
        self.end_date = self.dates[-1]
        self.df_rf = self.get_df_rf(
            begin_date=f"{self.begin_date.year}0101",
            end_date=f"{self.end_date.year}1231",
        )

        if cov_params is None:
            cov_params = {}
        if cov_func is None:
            self.cov_name = "HC"
            self.cov_func = lambda x: x.cov()
            self.cov_params = {}
        else:
            self.cov_name = cov_name or cov_params.get("cov_name", cov_func.__name__)
            self.cov_func = cov_func
            self.cov_params = cov_params

        self.save_path = os.path.abspath(save_path)
        self.output_path = os.path.join(self.save_path, self.cov_name)
        os.makedirs(self.output_path, exist_ok=True)

    def _solve_weights_with_fallback(
        self,
        strategy_name,
        covariance_matrix,
        expected_returns=None,
        target_returns=None,
        target_volatilities=None,
    ):
        solver_candidates = ("GUROBI", "ECOS", "SCS")
        last_exc = None

        for solver in solver_candidates:
            strat = self.optimizer_cls(
                strategy=strategy_name,
                min_weight=0,
                max_weight=1,
                txn_cost=0,
                use_cash=self.use_cash,
                solver=solver,
            )
            try:
                return strat.get_weights(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    target_returns=target_returns,
                    target_volatilities=target_volatilities,
                )
            except Exception as exc:
                last_exc = exc

        raise last_exc

    def allocate_weights(self, verbose=True):
        for idx, date in tqdm(
            enumerate(self.dates),
            disable=not verbose,
            desc=f"{self.cov_name} from {self.begin_date.date()} to {self.end_date.date()}",
            total=len(self.dates),
        ):
            date_str = date.strftime("%Y%m%d")
            output_file = os.path.join(self.output_path, f"{date_str}.pkl")
            if os.path.exists(output_file):
                continue

            date_tp1 = self.dates[idx + 1] if idx != len(self.dates) - 1 else None

            window_start = trailing_window_start(end_date=date, lookback_months=self.lookback_months)
            df_signal_rets = self.df_signal_rets[
                (self.df_signal_rets.index > window_start) & (self.df_signal_rets.index <= date)
            ].pivot(columns="permno", values="ret")
            if df_signal_rets.empty or len(df_signal_rets) < 2:
                continue

            signal_periods = len(df_signal_rets)
            count_nas = df_signal_rets.isna().sum()
            investable_universe = count_nas.index[(count_nas / signal_periods) < 0.9]

            mon_ret = self.df_mons_rets.loc[date]
            asset_universe = (
                mon_ret[mon_ret["permno"].isin(investable_universe)]
                .sort_values("market_cap", ascending=False)
                .head(self.ntop)["permno"]
                .tolist()
            )
            if not asset_universe:
                continue

            df_signal_rets = df_signal_rets.reindex(columns=asset_universe).fillna(0.0)
            if df_signal_rets.empty:
                continue

            dict_permno2ret_t = (
                self.df_mons_rets[self.df_mons_rets["permno"].isin(asset_universe)]
                .loc[date]
                .set_index("permno")["ret"]
                .to_dict()
            )

            if date_tp1 is not None:
                dict_permno2ret_tp1 = (
                    self.df_mons_rets[self.df_mons_rets["permno"].isin(asset_universe)]
                    .loc[date_tp1]
                    .set_index("permno")["ret"]
                    .to_dict()
                )
            else:
                dict_permno2ret_tp1 = {}

            rf = self.df_rf.loc[date].item()
            asset_returns = df_signal_rets.mean() * self.annual_factor
            asset_vols = df_signal_rets.std() * sqrt(self.annual_factor)
            asset_covs = self.cov_func(df_signal_rets, **self.cov_params) * self.annual_factor

            is_psd = is_psd_def(asset_covs)
            if not is_psd:
                non_psd_asset_covs = asset_covs.copy()
                try:
                    cor_mat = correlation_from_covariance(asset_covs).values
                    cor_mat_corrected = nearcorr(cor_mat, max_iterations=5000)
                    sd = np.diag(np.sqrt(np.diag(asset_covs.values)))
                    asset_covs = pd.DataFrame(
                        sd @ cor_mat_corrected @ sd,
                        index=asset_universe,
                        columns=asset_universe,
                    )
                    eigenvalues, _ = np.linalg.eigh(asset_covs)
                    min_eigenvalue = np.min(eigenvalues)
                    if min_eigenvalue < 0:
                        asset_covs -= min_eigenvalue * np.eye(len(asset_covs))
                except Exception as exc:
                    print(f"Failed to repair covariance matrix on {date_str}: {exc}")
                    continue
            else:
                non_psd_asset_covs = None

            df_asset_stats = pd.DataFrame(
                {
                    "asset_ret": asset_returns,
                    "asset_vol": asset_vols,
                    "permno": asset_universe,
                    "date": date_str,
                }
            )

            strategies = []
            try:
                mvp_weights = self._solve_weights_with_fallback(
                    strategy_name="mvp",
                    covariance_matrix=asset_covs,
                )
            except Exception as exc:
                print(f"An error occurred for mvp on {date_str}: {exc}")
                continue

            mvp_star = pd.Series(mvp_weights)[asset_universe]
            mvp_cash_weight = max(0.0, 1.0 - mvp_star.sum())
            mvp_ret, mvp_vol = compute_portfolio_return_and_volatility(
                mvp_star,
                asset_returns,
                asset_covs,
                annualize_factor=1,
                cash_weight=mvp_cash_weight,
                cash_return=rf,
            )
            strategies.append(
                {
                    "Strategy": "MVP",
                    "Expected Return": mvp_ret,
                    "Annual Volatility": mvp_vol,
                    "Cash Weight": mvp_cash_weight,
                    "weights": mvp_star.to_dict(),
                }
            )

            try:
                mrp_weights = self._solve_weights_with_fallback(
                    strategy_name="mvo_tgt_ret",
                    covariance_matrix=asset_covs,
                    expected_returns=asset_returns,
                    target_returns=float(asset_returns.max()),
                )
            except Exception as exc:
                print(f"An error occurred for mrp on {date_str}: {exc}")
                continue

            mrp_star = pd.Series(mrp_weights)[asset_universe]
            mrp_cash_weight = max(0.0, 1.0 - mrp_star.sum())
            mrp_ret, mrp_vol = compute_portfolio_return_and_volatility(
                mrp_star,
                asset_returns,
                asset_covs,
                annualize_factor=1,
                cash_weight=mrp_cash_weight,
                cash_return=rf,
            )

            min_vol = max(ceil(mvp_vol * 100) / 100, 0)
            max_vol = floor(mrp_vol * 100) / 100
            tgt_vols = [round(tgt_vol, 2) for tgt_vol in np.arange(min_vol, max_vol + 0.001, 0.01) if tgt_vol > 0]

            if tgt_vols:
                try:
                    dict_mvos = self._solve_weights_with_fallback(
                        strategy_name="mvo_tgt_vols",
                        covariance_matrix=asset_covs,
                        expected_returns=asset_returns,
                        target_volatilities=tgt_vols,
                    )
                except Exception as exc:
                    print(f"An error occurred for mvo on {date_str}: {exc}")
                    continue

                for tgt_vol, weights in dict_mvos.items():
                    weights_arr = pd.Series(weights)[asset_universe].values
                    cash_weight = max(0.0, 1.0 - pd.Series(weights)[asset_universe].sum())
                    mvo_ret, mvo_vol = compute_portfolio_return_and_volatility(
                        weights_arr,
                        asset_returns,
                        asset_covs,
                        annualize_factor=1,
                        cash_weight=cash_weight,
                        cash_return=rf,
                    )
                    strategies.append(
                        {
                            "Strategy": f"{tgt_vol:.0%}",
                            "Expected Return": mvo_ret,
                            "Annual Volatility": mvo_vol,
                            "Cash Weight": cash_weight,
                            "weights": weights,
                        }
                    )

            strategies.append(
                {
                    "Strategy": "MRP",
                    "Expected Return": mrp_ret,
                    "Annual Volatility": mrp_vol,
                    "Cash Weight": mrp_cash_weight,
                    "weights": mrp_star.to_dict(),
                }
            )

            with open(output_file, "wb") as file_handle:
                pickle.dump(
                    {
                        "rf": rf * 12,
                        "rf_monthly": rf,
                        "assets_stats": df_asset_stats,
                        "strategies": strategies,
                        "rm_t": {permno: dict_permno2ret_t.get(permno, 0.0) for permno in asset_universe},
                        "rm_tp1": {permno: dict_permno2ret_tp1.get(permno, 0.0) for permno in asset_universe},
                        "is_psd": is_psd,
                        "cov_name": self.cov_name,
                        "cov_params": self.cov_params,
                        "date": date_str,
                        "lookback_months": self.lookback_months,
                        "signal_freq": self.freq,
                        "use_cash": self.use_cash,
                        "signal_periods": signal_periods,
                        "cov": asset_covs,
                        "non_psd_cov": non_psd_asset_covs,
                    },
                    file_handle,
                )

    def get_df_rf(self, begin_date, end_date):
        os.makedirs(self.rf_cache_dir, exist_ok=True)
        exact_path = os.path.join(
            self.rf_cache_dir,
            f"{self.rf_benchmark.lower()}_{begin_date}_{end_date}.csv",
        )

        if os.path.isfile(exact_path):
            df_rf = pd.read_csv(exact_path, parse_dates=["date"], index_col="date")
        else:
            cached_path = self._find_covering_rf_cache(begin_date, end_date)
            if cached_path is not None:
                df_rf = pd.read_csv(cached_path, parse_dates=["date"], index_col="date")
            else:
                df_rf = web.DataReader(self.rf_benchmark, "fred", begin_date, end_date).resample("ME", label="right").last()
                df_rf.index.name = "date"
                df_rf.to_csv(exact_path)

        df_rf = df_rf.loc[self.begin_date:self.end_date]
        return (1 + (df_rf / 100)) ** (1 / 12.0) - 1.0

    def _find_covering_rf_cache(self, begin_date, end_date):
        pattern = os.path.join(self.rf_cache_dir, f"{self.rf_benchmark.lower()}_*.csv")
        candidate_paths = []
        for path in glob(pattern):
            name = os.path.splitext(os.path.basename(path))[0]
            parts = name.split("_")
            if len(parts) < 3:
                continue
            file_begin = parts[-2]
            file_end = parts[-1]
            if file_begin <= begin_date and file_end >= end_date:
                span = int(file_end) - int(file_begin)
                candidate_paths.append((span, path))

        if not candidate_paths:
            return None

        candidate_paths.sort(key=lambda item: item[0])
        return candidate_paths[0][1]
