import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from .lib_portfolio_analyzer import portfolio_analyzer
from .lib_trans_cost import TransCost


def dot(dict1, dict2):
    return np.nansum([dict1.get(key, 0.0) * dict2.get(key, 0.0) for key in set(dict1) | set(dict2)]).item()


def risky_turnover(new_weights, old_weights):
    assets = set(new_weights) | set(old_weights)
    return sum(abs(new_weights.get(asset, 0.0) - old_weights.get(asset, 0.0)) for asset in assets)


def run_sp500_backtest(
    cov_name,
    results_folder,
    performance_prefix,
    data_path,
    begin_date,
    end_date,
    tc_cost=10,
    rp_min=0.03,
    rp_max=0.15,
    rp_step=0.01,
    feat="Annualized Sharpe Ratio",
    verbose=False,
):
    tc = TransCost(c=tc_cost)
    performance_path = os.path.join(performance_prefix, f"tc{tc_cost}", cov_name)
    if os.path.exists(os.path.join(performance_path, "portfolios_performance.csv")):
        df_performs = pd.read_csv(
            os.path.join(performance_path, "portfolios_performance.csv"),
            index_col="metrics",
        )
        return df_performs.loc[feat].sum().item()

    os.makedirs(performance_path, exist_ok=True)
    rps = np.arange(rp_min, rp_max + rp_step, rp_step)
    res = []
    dict_prev_weights = {}

    df_mons_rets = pd.read_csv(
        os.path.join(data_path, "mon_rets.csv"),
        parse_dates=["date"],
        index_col="date",
    ).loc[begin_date:end_date]
    dates = df_mons_rets.index.drop_duplicates()

    prev_dict_effs = None
    for idx, date in tqdm(
        enumerate(dates),
        desc=f"Backtesting {cov_name}",
        total=len(dates),
        disable=not verbose,
    ):
        date_str = date.strftime("%Y%m%d")
        result_file = os.path.join(results_folder, cov_name, f"{date_str}.pkl")
        try:
            with open(result_file, "rb") as file_handle:
                dict_effs = pickle.load(file_handle)
        except Exception as exc:
            print(f"Error loading {result_file}: {exc}")
            dict_effs = prev_dict_effs

        if dict_effs is None:
            continue
        if idx == len(dates) - 1:
            continue

        rf = dict_effs["rf"]
        rf_monthly = float(dict_effs.get("rf_monthly", rf / 12.0))
        use_cash = bool(dict_effs.get("use_cash", False))
        dict_mvp = dict_effs["strategies"][0]
        dict_mrp = dict_effs["strategies"][-1]
        mvp_vol = dict_mvp["Annual Volatility"]
        mrp_vol = dict_mrp["Annual Volatility"]
        mvp_weights = dict_mvp["weights"]
        mrp_weights = dict_mrp["weights"]
        df_effs = pd.DataFrame(dict_effs["strategies"]).set_index("Strategy")
        assets = dict_effs["assets_stats"].index

        rets_t = df_mons_rets.loc[date].set_index("permno")["ret"].to_dict()
        rets_tp1 = df_mons_rets.loc[dates[idx + 1]].set_index("permno")["ret"].to_dict()

        rm_t = {asset: rets_t.get(asset, 0.0) for asset in assets}
        rm_tp1 = {asset: rets_tp1.get(asset, 0.0) for asset in assets}

        for rp in rps:
            if rp <= mvp_vol:
                w_t = mvp_weights
            elif rp > mrp_vol:
                w_t = mrp_weights
            else:
                try:
                    w_t = df_effs.loc[f"{rp * 100:.0f}%", "weights"]
                except Exception as exc:
                    print(f"Missing target-return weights for {date_str} at {rp:.2f}: {exc}")
                    continue

            cash_weight_t = max(0.0, 1.0 - sum(w_t.values())) if use_cash else 0.0
            rpm_t = dot(rm_t, w_t) + cash_weight_t * rf_monthly
            rpm_tp1_wo_cost = dot(rm_tp1, w_t) + cash_weight_t * rf_monthly
            w_tm1 = dict_prev_weights.get(rp, {})
            cash_weight_tm1 = max(0.0, 1.0 - sum(w_tm1.values())) if use_cash else 0.0
            prev_portfolio_return = dot(rm_t, w_tm1) + cash_weight_tm1 * rf_monthly
            gross_portfolio = 1.0 + prev_portfolio_return
            if gross_portfolio <= 0:
                w_tplus_normalized = {}
            else:
                w_tplus_normalized = {
                    k: (v * (1 + rm_t.get(k, 0.0))) / gross_portfolio
                    for k, v in w_tm1.items()
                }

            if use_cash:
                turnover = risky_turnover(new_weights=w_t, old_weights=w_tplus_normalized)
                cost = (tc_cost / 10000.0) * turnover
            else:
                cost = tc.get_cost(new_weights=w_t, old_weights=w_tplus_normalized)
                turnover = sum(
                    abs(w_t.get(asset, 0.0) - w_tplus_normalized.get(asset, 0.0))
                    for asset in set(w_t) | set(w_tplus_normalized)
                )
            rpm_tp1_wt_cost = (1 + rpm_tp1_wo_cost) * (1 - cost) - 1
            hhi = (pd.Series(w_t) ** 2).sum().item()
            dict_prev_weights[rp] = w_t

            res.append(
                {
                    "date": dates[idx + 1],
                    "tgt_ret_str": f"{rp * 100:.0f}%",
                    "tgt_ret": round(rp.item(), 2),
                    "rf": rf,
                    "rpm_tp1": rpm_tp1_wt_cost,
                    "rpm_tp1_wo_cost": rpm_tp1_wo_cost,
                    "rpm_t": rpm_t,
                    "wgt_t": w_t,
                    "cost": cost,
                    "turnover": turnover,
                    "hhi": hhi,
                }
            )

        prev_dict_effs = dict_effs

    return process_results(res=res, result_path=performance_path, feat=feat)


def process_results(res, result_path, feat="Annualized Sharpe Ratio"):
    if not res:
        raise ValueError(f"No backtest observations were generated for {result_path}.")

    df_weights = pd.DataFrame.from_dict(
        {(pd.Timestamp(r["date"]), r["tgt_ret_str"]): r["wgt_t"] for r in res},
        orient="index",
    ).fillna(0)
    df_weights.index.set_names(["date", "tgt_ret"], inplace=True)
    df_weights.sort_index(ascending=[True, True], inplace=True)

    df_port_turn_hhi = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(r["date"]),
                "tgt_ret_str": f"{r['tgt_ret'] * 100:02.0f}%",
                "turnover": r["turnover"],
                "hhi": r["hhi"],
                "cost": r["cost"],
            }
            for r in res
        ]
    )
    df_metrics_turnover = df_port_turn_hhi.groupby("tgt_ret_str")["turnover"].mean()
    df_metrics_hhi = 1 / df_port_turn_hhi.groupby("tgt_ret_str")["hhi"].mean()

    df_res_pivot = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(r["date"]),
                "tgt_ret_str": f"{r['tgt_ret'] * 100:02.0f}%",
                "rpm_tp1": r["rpm_tp1"],
            }
            for r in res
        ]
    ).pivot(index="date", columns="tgt_ret_str", values="rpm_tp1").fillna(0.0)
    first_date = pd.Timestamp(res[0]["date"])
    baseline_date = first_date - pd.offsets.MonthEnd(1)
    if baseline_date not in df_res_pivot.index:
        df_res_pivot.loc[baseline_date] = 0.0
    df_res_pivot.sort_index(ascending=True, inplace=True)

    df_port_values = (100000 * (1 + df_res_pivot).cumprod()).reset_index()
    df_port_values.columns.name = None

    pe = portfolio_analyzer()
    df_performs = pe.get_portfolio_metrics(df_ports=df_port_values)
    df_performs.loc["Turnover (%)", df_metrics_turnover.index] = df_metrics_turnover * 100.0
    df_performs.loc["HHI", df_metrics_hhi.index] = df_metrics_hhi

    df_performs.to_csv(os.path.join(result_path, "portfolios_performance.csv"), float_format="%.4f")
    df_port_values.to_csv(os.path.join(result_path, "portfolios_value.csv"), float_format="%.2f", index=False)
    return df_performs.loc[feat].sum().item()
