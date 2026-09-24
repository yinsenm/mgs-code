import os.path
from .nearest_correlation import nearcorr
import pandas as pd
import numpy as np
import pickle
from math import sqrt, ceil, floor
from tqdm import tqdm
from .lib_port_strategy_cvx import PortfolioStrategyCVX
from .lib_cov_func import check_symmetric, is_psd_def, correlation_from_covariance
from .lib_cov_func import ledoit, gerber_cov_stat1
from .lib_multi_asset_data import (
    annualization_factor,
    infer_observation_spacing,
    load_multi_asset_price_history,
    normalize_signal_frequency,
    price_history_to_returns,
    trailing_window_start,
)
import pandas_datareader.data as web  # download 3m t-bill from FRED as risk-free rate
rf_cache_path = "../data"


def compute_portfolio_return_and_volatility(weights, return_vector, covariance_matrix, annualize_factor=252):
    """
    Compute the expected portfolio return and volatility.

    Parameters:
    weights (np.ndarray): Array of asset weights in the portfolio.
    return_vector (np.ndarray): Array of expected returns for each asset.
    covariance_matrix (np.ndarray): Covariance matrix of asset returns.
    annualize_factor (int): Number of periods per year (default is 252 for daily returns).


    Returns:
    tuple: Expected portfolio return and portfolio volatility.
    """
    # Ensure weights, return_vector, and covariance_matrix are numpy arrays
    weights = np.array(weights)
    return_vector = np.array(return_vector)
    covariance_matrix = np.array(covariance_matrix)

    # Calculate expected portfolio return
    portfolio_return = np.dot(weights, return_vector) * annualize_factor

    # Calculate portfolio volatility (standard deviation)
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights)) * annualize_factor
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_return.item(), portfolio_volatility.item()


class PortfolioAllocator:
    def __init__(
            self,
            cov_func = None,
            cov_name: str = None,
            cov_params: dict | None = None,
            data_path:str = "../data/n9_1988_2024.csv",
            save_path: str = "../results",
            lookback: int = 24,
            start_date: str = None,
            end_date: str = None,
            skip_dates: list[str] | tuple[str, ...] | None = None,
            allow_cash: bool = False,
            freq: str = "monthly",
            reuse_existing: bool = False,
    ):
        self.rf_benchmark = "TB3MS"
        self.save_path = f"{save_path}"
        self.freq = normalize_signal_frequency(freq)
        self.signal_freq = self.freq
        self.signal_annual_factor = annualization_factor(self.signal_freq)
        self.rebalance_freq = "monthly"
        self.rebalance_annual_factor = 12
        self.lookback = lookback
        self.lookback_months = lookback
        self.allow_cash = allow_cash
        self.reuse_existing = reuse_existing
        self.prices = load_multi_asset_price_history(data_path)
        self.source_spacing = infer_observation_spacing(self.prices)
        if self.signal_freq == "weekly" and self.source_spacing == "monthly":
            raise ValueError(
                "Weekly signal calculation requires a daily or weekly source series. "
                f"Received sparse monthly observations in {data_path}."
            )

        self.rets = price_history_to_returns(self.prices, freq=self.rebalance_freq)
        self.signal_rets = price_history_to_returns(self.prices, freq=self.signal_freq)
        self.tickers = self.rets.columns

        self.dates = self.rets.index
        if start_date is not None:
            self.dates = self.dates[self.dates >= start_date]
        if end_date is not None:
            self.dates = self.dates[self.dates <= end_date]
        if skip_dates:
            skip_timestamps = {pd.Timestamp(value) for value in skip_dates}
            self.dates = self.dates[~self.dates.isin(skip_timestamps)]
        if len(self.dates) == 0:
            raise ValueError(
                f"No {self.freq} observations found between {start_date} and {end_date} "
                f"for {data_path}."
            )
        self.begin_date, self.end_date = self.dates[0], self.dates[-1]
        self.df_rf = self.align_monthly_rf_to_dates(
            self.get_df_rf(f"{self.begin_date.year}0101", f"{self.end_date.year}1231"),
            self.dates,
        )

        self.is_psd = None
        if cov_func is None:
            self.cov_name = "HC"
            self.cov_func = lambda x: x.cov()
            self.cov_params = {}
        else:
            resolved_cov_params = {} if cov_params is None else dict(cov_params)
            self.cov_name = resolved_cov_params.get("cov_name", cov_func.__name__) if cov_name is None else cov_name
            self.cov_func = cov_func
            self.cov_params = resolved_cov_params
        os.makedirs(f"{self.save_path}/{self.cov_name}", exist_ok=True)

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
            strat = PortfolioStrategyCVX(
                strategy=strategy_name,
                min_weight=0,
                max_weight=1,
                txn_cost=0,
                use_cash=self.allow_cash,
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
                enumerate(self.dates), disable=not verbose,
                desc=f"{self.cov_name} from {self.dates[0]} to {self.dates[-1]}",
                total=len(self.dates)
        ):
            date_str = date.strftime("%Y%m%d")
            if self.reuse_existing and os.path.exists(f"{self.save_path}/{self.cov_name}/{date_str}.pkl"):
                continue

            window_start = trailing_window_start(end_date=date, lookback_months=self.lookback_months)
            df_rets = self.signal_rets[
                (self.signal_rets.index > window_start) & (self.signal_rets.index <= date)
            ].dropna(how="all")
            if len(df_rets) < 2:
                continue

            tickers = df_rets.columns
            dict_ret_t = self.rets.loc[date].fillna(0.0).to_dict()

            if idx != len(self.dates) - 1:
                date_tp1 = self.dates[idx + 1]
                dict_ret_tp1 = self.rets.loc[date_tp1].fillna(0.0).to_dict()
            else:
                date_tp1 = None
                dict_ret_tp1 = {}

            # get rf
            rf = float(self.df_rf.loc[date].item())

            # compute return, vol and correlation
            asset_returns = df_rets.mean() * self.signal_annual_factor
            asset_vols = df_rets.std() * sqrt(self.signal_annual_factor)
            asset_covs = self.cov_func(df_rets, **self.cov_params) * self.signal_annual_factor

            # check asset_cov is psd or not
            self.is_psd = is_psd_def(asset_covs)

            if not self.is_psd:
                # correct non-psd matrix using nearcorr method
                non_psd_asset_covs = asset_covs.copy()
                # correct the correlation matrix
                cor_mat = correlation_from_covariance(asset_covs).values
                # find the nearest correlation matrix that is psd
                try:
                    cor_mat_corrected = nearcorr(cor_mat, max_iterations=5000)
                except Exception as e:
                    print(f"Error in nearcorr on {date}: {e}")
                    continue
                sd = np.diag(np.sqrt(np.diag(asset_covs.values)))
                asset_covs = pd.DataFrame(
                    sd @ cor_mat_corrected @ sd, index=tickers, columns=tickers
                )
                w, U = np.linalg.eigh(asset_covs)
                # https://github.com/robertmartin8/PyPortfolioOpt/issues/82
                mw = np.min(w)
                if mw < 0:
                    asset_covs -= mw * np.eye(len(asset_covs))

            # print(f"test psd {date} ...{self.cov_params}")
            # print(is_psd_def(cor_mat_corrected))
            else:
                non_psd_asset_covs = None

            df_asset_stats = pd.DataFrame({
                "asset_ret": asset_returns,
                "asset_vol": asset_vols,
                "ticker": tickers,
                "date": date
            })
            strategies = []
            # run MVP (minimal variance portfolio)
            try:
                weights = self._solve_weights_with_fallback(
                    strategy_name="mvp",
                    covariance_matrix=asset_covs,
                )
            except Exception as e:
                print(f"An error occurred for mvp on {date}:", e)
                continue

            mvp_star = pd.Series(weights)[tickers]

            # compute port mvp ret and vol
            mvp_ret, mvp_vol = compute_portfolio_return_and_volatility(mvp_star, asset_returns, asset_covs, 1)

            strategies.append({
                "Strategy": "MVP", "Expected Return": mvp_ret, "Annual Volatility": mvp_vol, "weights": mvp_star.to_dict(),
            })

            # find the weights that corresponds to the max_ret
            max_ret = asset_returns.max().item()
            mrp_star = pd.Series(
                self._solve_weights_with_fallback(
                    strategy_name="mvo_tgt_ret",
                    expected_returns=asset_returns,
                    covariance_matrix=asset_covs,
                    target_returns=max_ret,
                )
            )[tickers]

            # compute port mrp ret and vol
            mrp_ret, mrp_vol = compute_portfolio_return_and_volatility(mrp_star, asset_returns, asset_covs, 1)
            min_vol, max_vol = ceil(mvp_vol * 100) / 100, floor(mrp_vol * 100) / 100

            step = 0.01
            tgt_vols = np.arange(min_vol, max_vol + 1e-9, step)
            tgt_vols = np.round(tgt_vols, 2)
            tgt_vols = tgt_vols[tgt_vols <= max_vol + 1e-12]

            try:
                dict_mvos = self._solve_weights_with_fallback(
                    strategy_name="mvo_tgt_vols",
                    expected_returns=asset_returns,
                    covariance_matrix=asset_covs,
                    target_volatilities=tgt_vols,
                )
            except Exception as e:
                print(f"An error occurred for mvo on {date}:", e)
                continue

            for tgt_vol, weights in dict_mvos.items():
                weights_arr = pd.Series(weights)[self.tickers].values
                mvo_ret, mvo_vol = compute_portfolio_return_and_volatility(weights_arr, asset_returns, asset_covs, 1)
                strategies.append({
                    "Strategy": f"{tgt_vol:.0%}",
                    "Expected Return": mvo_ret,
                    "Annual Volatility": mvo_vol,
                    "weights": weights,
                })

            # append results of MRP
            strategies.append({
                "Strategy": "MRP", "Expected Return": mrp_ret, "Annual Volatility": mrp_vol, "weights": mrp_star.to_dict(),
            })



            # save results
            with open(f"{self.save_path}/{self.cov_name}/{date_str}.pkl", "wb") as file:
                res = {
                    "rf": (1.0 + rf) ** self.rebalance_annual_factor - 1.0,
                    "rf_periodic": rf,
                    "assets_stats": df_asset_stats,
                    "strategies": strategies,
                    "rm_t": dict_ret_t,
                    "rm_tp1":dict_ret_tp1 if dict_ret_tp1 is not None else {},
                    "cov": asset_covs,
                    "non_psd_cov": non_psd_asset_covs,
                    "is_psd": self.is_psd,
                    "cov_name": self.cov_name,
                    "cov_params": self.cov_params,
                    "use_cash": self.allow_cash,
                    "freq": self.signal_freq,
                    "signal_freq": self.signal_freq,
                    "rebalance_freq": self.rebalance_freq,
                    "lookback_months": self.lookback_months,
                    "annual_factor": self.signal_annual_factor,
                    "date": date
                }
                if date_tp1 is not None:
                    res["date_tp1"] = date_tp1
                res["rf_monthly"] = rf
                pickle.dump(res, file)

            # delete variables
            # del df_daily_rets, df_monthly_rets_t, df_monthly_rets_tp1, res, asset_covs, non_psd_asset_covs, strategies, port_strat, port_strats
            # gc.collect()

    def get_df_rf(self, begin_date: str, end_date: str) -> pd.DataFrame:
        rf_cache_file = ("%s/%s_%s_%s.csv" % (
            rf_cache_path, self.rf_benchmark.lower(),begin_date, end_date
        ))
        os.makedirs(rf_cache_path, exist_ok=True)
        if os.path.isfile(rf_cache_file):
            df_rf = pd.read_csv(rf_cache_file, parse_dates=["date"], index_col="date")
        else:
            cache_prefix = f"{self.rf_benchmark.lower()}_"
            begin_month = begin_date[:6]
            end_month = end_date[:6]
            cache_candidates = []

            for file_name in os.listdir(rf_cache_path):
                if not file_name.startswith(cache_prefix) or not file_name.endswith(".csv"):
                    continue

                stem = os.path.splitext(file_name)[0]
                parts = stem.split("_")
                if len(parts) != 3:
                    continue

                _, cache_begin, cache_end = parts
                if cache_begin[:6] <= begin_month and cache_end[:6] >= end_month:
                    span = int(cache_end) - int(cache_begin)
                    cache_candidates.append((span, os.path.join(rf_cache_path, file_name)))

            if cache_candidates:
                _, fallback_cache = min(cache_candidates, key=lambda item: item[0])
                df_rf = pd.read_csv(fallback_cache, parse_dates=["date"], index_col="date")
            else:
                df_rf = web.DataReader(self.rf_benchmark, "fred", begin_date, end_date).\
                    resample("ME", label="right").last()
                df_rf.index.name = "date"
                df_rf.to_csv(rf_cache_file)

        df_rf = df_rf.loc[begin_date:end_date]
        df_rf = (1 + (df_rf / 100)) ** (1 / self.rebalance_annual_factor) - 1.0
        return df_rf

    @staticmethod
    def align_monthly_rf_to_dates(df_rf: pd.DataFrame, target_dates) -> pd.DataFrame:
        rf_by_period = df_rf.copy()
        rf_by_period["_period"] = rf_by_period.index.to_period("M")
        rf_by_period = rf_by_period.drop_duplicates("_period", keep="last").set_index("_period")

        target_index = pd.DatetimeIndex(pd.to_datetime(target_dates))
        target_periods = target_index.to_period("M")
        aligned = rf_by_period.reindex(target_periods)
        aligned.index = target_index
        return aligned



if __name__ == "__main__":
    port_alloc = PortfolioAllocator(cov_func=ledoit, cov_params={}, cov_name="SM")
    port_alloc.allocate_weights(verbose=True)
