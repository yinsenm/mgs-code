import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
from math import sqrt
from .lib_multi_asset_data import annualization_factor, normalize_signal_frequency
import pandas_datareader.data as web  # download 3m t-bill from FRED as risk-free rate
rf_cache_path = "../data"  # path to save for risk free rate data

class portfolio_analyzer:
    def __init__(self, freq: str = "monthly"):
        self.rf_benchmark = "TB3MS"
        self.freq = normalize_signal_frequency(freq)
        self.annual_factor = annualization_factor(self.freq)
        self.var_prefix = self.freq.capitalize() if self.freq != "monthly" else "Monthly"

    def get_df_rf(self, begin_date: str, end_date: str, target_index=None) -> pd.DataFrame:
        rf_cache_file = ("%s/%s_%s_%s.csv" % (
            rf_cache_path, self.rf_benchmark.lower(), begin_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        ))
        if os.path.isfile(rf_cache_file):
            df_rf = pd.read_csv(rf_cache_file, parse_dates=["date"], index_col="date")
        else:
            os.makedirs(rf_cache_path, exist_ok=True)
            cache_prefix = f"{self.rf_benchmark.lower()}_"
            begin_month = begin_date.strftime("%Y%m")
            end_month = end_date.strftime("%Y%m")
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
        df_rf = (1 + (df_rf / 100)) ** (1 / self.annual_factor) - 1.0
        if target_index is not None:
            target_index = pd.DatetimeIndex(pd.to_datetime(target_index)).sort_values()
            if self.freq == "monthly":
                rf_by_period = df_rf.copy()
                rf_by_period["_period"] = rf_by_period.index.to_period("M")
                rf_by_period = rf_by_period.drop_duplicates("_period", keep="last").set_index("_period")
                target_periods = target_index.to_period("M")
                df_rf = rf_by_period.reindex(target_periods)
                df_rf.index = target_index
            else:
                df_rf = df_rf.reindex(target_index, method="ffill")
        return df_rf

    @staticmethod
    def get_metrics(prcs, rets, excess_rets, annual_factor: int = 12, var_prefix: str = "Monthly"):
        metrics = dict()
        metrics["Sharpe Ratio"] = excess_rets.mean() / rets.std()
        metrics["Annualized Sharpe Ratio"] = metrics["Sharpe Ratio"] * sqrt(annual_factor)
        metrics["Skewness"] = rets.skew()
        metrics["Kurtosis"] = stats.kurtosis(rets.to_list(), fisher=False)
        metrics["Adjusted Sharpe Ratio"] = metrics["Sharpe Ratio"] * (
                1.0 + (metrics["Skewness"] / 6.0) * metrics["Sharpe Ratio"] -
                ((metrics["Kurtosis"] - 3) / 24.) * metrics["Sharpe Ratio"] ** 2
        )

        metrics["Annualized STD (%)"] = sqrt(annual_factor) * rets.std() * 100.
        metrics["Annualized Kurtosis"] = stats.kurtosis((annual_factor * rets).to_list(), fisher=False)
        metrics["Annualized Skewness"] = (annual_factor * rets).skew()
        metrics["Cumulative Return (%)"] = ((1. + rets).prod() - 1.0) * 100.
        metrics["Annual Return (%)"] = (1.0 + rets).groupby(rets.index.year).prod() - 1.0

        n = len(metrics["Annual Return (%)"])
        metrics["Arithmetic Return (%)"] = metrics["Annual Return (%)"].mean() * 100.
        metrics["Geometric Return (%)"] = (((metrics["Annual Return (%)"] + 1.).prod()) ** (1.0 / n) - 1.0) * 100.

        # compute VaR
        metrics[f"{var_prefix} 95% VaR (%)"] = rets.quantile(0.05) * 100.
        metrics[f"Alt {var_prefix} 95% VaR (%)"] = norm.ppf(0.05, rets.mean(), rets.std()) * 100.

        # compute maximum drawdown
        rolling_max = prcs.expanding().max()
        drawdown = prcs / rolling_max - 1.0
        metrics["Maximum Drawdown (%)"] = drawdown.min() * 100.
        metrics["MDD / VOL"] = metrics["Maximum Drawdown (%)"] / metrics["Annualized STD (%)"]

        dd = np.sqrt(np.sum(np.minimum(excess_rets, 0) ** 2) / len(excess_rets))
        metrics["Sortino Ratio"] = excess_rets.mean() / (dd + 1e-8)
        metrics["Annualized Sortino Ratio"] = sqrt(annual_factor) * metrics["Sortino Ratio"]

        metrics["Calmar Ratio"] = metrics["Geometric Return (%)"] / (abs(metrics["Maximum Drawdown (%)"]) + 1e-8)
        metrics["Terminal Dollar Value"] = prcs.iloc[-1]
        return metrics

    def get_portfolio_metrics(self, df_ports, used_metrics=None) -> pd.DataFrame:
        begin_date, end_date = df_ports["date"].min(), df_ports["date"].max()
        df_port_values = df_ports.copy()
        df_port_values["date"] = pd.to_datetime(df_port_values["date"])
        df_port_values = df_port_values.sort_values("date").set_index("date")
        if self.freq == "monthly":
            df_port_values = df_port_values.resample("ME", label="right").last()

        df_rf = self.get_df_rf(
            begin_date=begin_date,
            end_date=end_date,
            target_index=df_port_values.index,
        )

        if used_metrics is None:
            used_metrics = [
                "Arithmetic Return (%)",
                "Geometric Return (%)",
                "Annualized STD (%)",
                "Cumulative Return (%)",
                "Maximum Drawdown (%)",
                "MDD / VOL",
                f"{self.var_prefix} 95% VaR (%)",
                "Sharpe Ratio",
                "Adjusted Sharpe Ratio",
                "Annualized Sharpe Ratio",
                "Annualized Sortino Ratio",
                "Calmar Ratio",
                "Terminal Dollar Value",
            ]

        port_metrics = {}
        for port_name in df_port_values.columns:
            prcs = df_port_values[port_name]
            rets = prcs.pct_change(fill_method=None).dropna()
            excess_rets = rets - df_rf.loc[rets.index, self.rf_benchmark]
            port_metrics[port_name] = self.get_metrics(
                prcs=prcs,
                rets=rets,
                excess_rets=excess_rets,
                annual_factor=self.annual_factor,
                var_prefix=self.var_prefix,
            )

        df_port_performance = (
            pd.DataFrame(port_metrics).loc[used_metrics].
                infer_objects(copy=False).fillna(0.)
        )
        df_port_performance.rename_axis("metrics", inplace=True)
        return df_port_performance
