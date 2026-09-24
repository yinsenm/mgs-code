import pickle
from glob import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .lib_multi_asset_data import annualization_factor, normalize_signal_frequency
from .lib_portfolio_analyzer import portfolio_analyzer
from .lib_trans_cost import TransCost

def dot(dict1, dict2):
    # Get values using .get() to handle missing keys
    return np.nansum([dict1.get(key, 0) * dict2.get(key, 0) for key in set(dict1) | set(dict2)]).item()


def risky_turnover(new_weights, old_weights):
    assets = set(new_weights) | set(old_weights)
    return sum(abs(new_weights.get(asset, 0.0) - old_weights.get(asset, 0.0)) for asset in assets)


def run_backtest(
    cov_name,
    results_folder,
    performance_prefix,
    tc_cost=10,
    start_date=None,
    end_date=None,
    feat="Annualized Sharpe Ratio",
    freq="monthly",
    reuse_existing=False,
):
    rets_files = sorted(glob(f"{results_folder}/{cov_name}/*.pkl"))
    dates = [file.split("/")[-1].split(".")[0] for file in rets_files]
    if start_date:
        dates = [d for d in dates if d >= start_date.replace("-", "")]
    if end_date:
        dates = [d for d in dates if d <= end_date.replace("-", "")]
    tc = TransCost(c=tc_cost)  # define transaction cost
    performance_path = f"{performance_prefix}/tc{tc_cost}/{cov_name}"
    performance_file = f"{performance_path}/portfolios_performance.csv"
    if reuse_existing and os.path.exists(performance_file):
        df_performs = pd.read_csv(
            performance_file, index_col=["metrics"]
        )
        return df_performs.loc['Annualized Sharpe Ratio'].sum().item()


    os.makedirs(performance_path, exist_ok=True)

    rp_min, rp_max, rp_step = 0.03, 0.15, 0.01
    rps = np.arange(rp_min, rp_max + rp_step, rp_step)
    res = []
    dict_prev_weights = {}
    realized_freq = normalize_signal_frequency(freq)
    for date in tqdm(dates, desc=f"backtest {cov_name} ... "):
        with open(f"{results_folder}/{cov_name}/{date}.pkl", "rb") as f:
            dict_rets = pickle.load(f)

        dict_effs = dict_rets
        realized_freq = normalize_signal_frequency(dict_rets.get("rebalance_freq", realized_freq))

        rf = dict_rets["rf"]
        rf_periodic = float(dict_rets.get("rf_periodic", dict_rets.get("rf_monthly", rf / 12.0)))
        use_cash = bool(dict_rets.get("use_cash", False))
        dict_mvp = dict_effs["strategies"][0]
        dict_mrp = dict_effs["strategies"][-1]

        mvp_vol, mrp_vol = dict_mvp["Annual Volatility"], dict_mrp["Annual Volatility"]
        mvp_weights, mrp_weights = dict_mvp["weights"], dict_mrp["weights"]
        df_effs = pd.DataFrame(dict_effs["strategies"]).set_index("Strategy")

        decision_date = pd.Timestamp(dict_rets["date"])
        date_tp1 = dict_rets.get("date_tp1")
        if date_tp1 is None:
            continue
        date_tp1 = pd.Timestamp(date_tp1)
        if end_date is not None and date_tp1 > pd.Timestamp(end_date):
            continue
        rm_t = dict_rets["rm_t"]
        rm_tp1 = dict_rets["rm_tp1"]
        if len(rm_tp1) == 0:
            continue

        for rp in rps:
            if rp <= mvp_vol:
                w_t = mvp_weights
            elif rp > mrp_vol:
                w_t = mrp_weights
            else:
                try:
                    w_t = df_effs.loc["%.0f%%" % (rp * 100), "weights"]
                except Exception as e:
                    print(f"error {date} target weight missing - {rp * 100} {e}")
                    continue

            cash_weight_t = max(0.0, 1.0 - sum(w_t.values())) if use_cash else 0.0
            rpm_tp1_wo_cost = dot(rm_tp1, w_t) + cash_weight_t * rf_periodic
            w_tm1 = dict_prev_weights.get(rp, {})  # get previous weights
            cash_weight_tm1 = max(0.0, 1.0 - sum(w_tm1.values())) if use_cash else 0.0
            prev_portfolio_return = dot(rm_t, w_tm1) + cash_weight_tm1 * rf_periodic
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
                union_assets = list(w_t.keys() | w_tplus_normalized.keys())
                turnover = sum([
                    abs(w_t.get(asset, 0) - w_tplus_normalized.get(asset, 0)) for asset in union_assets
                ])
            rpm_tp1_wt_cost = (1 + rpm_tp1_wo_cost) * (1 - cost) - 1

            # compute diversification metrics
            hhi = (pd.Series(w_t) ** 2).sum().item()
            dict_prev_weights[rp] = w_t
            res.append({
                "date": date_tp1,
                "decision_date": decision_date,
                "tgt_ret_str": "%.0f%%" % (rp * 100),
                "tgt_ret": round(rp.item(), 2),
                "rf": rf,
                "rpm_tp1": rpm_tp1_wt_cost,
                "rpm_tp1_wo_cost": rpm_tp1_wo_cost,
                "rpm_t": prev_portfolio_return,
                "wgt_t": w_t,
                "cost": cost,  # drag on return
                "turnover": turnover,
                "hhi": hhi,
            })

    if not res:
        raise ValueError(f"No backtest observations available for {cov_name} in {results_folder}.")

    df_weights = pd.DataFrame.from_dict(
        {(pd.Timestamp(r["decision_date"]), r["tgt_ret_str"]): r["wgt_t"] for r in res}
    , orient='index').fillna(0)
    df_weights.index.set_names(['date', 'tgt_ret'], inplace=True)
    df_weights.sort_index(ascending=[True, True], inplace=True)

    df_port_turn_hhi = pd.DataFrame([{
        "date": pd.Timestamp(r["decision_date"]),
        "tgt_ret_str": f"{r['tgt_ret'] * 100:02.0f}%",
        "turnover": r["turnover"],
        "hhi": r["hhi"],
        "cost": r["cost"]
    }  for r in res ])
    df_metrics_turnover = df_port_turn_hhi.groupby(["tgt_ret_str"])['turnover'].mean()
    df_metrics_hhi = 1 / df_port_turn_hhi.groupby(["tgt_ret_str"])['hhi'].mean()

    df_res_pivot = pd.DataFrame([{
        "date": pd.Timestamp(r["date"]),
        "tgt_ret_str": f"{r['tgt_ret'] * 100:02.0f}%",
        "rpm_tp1": r["rpm_tp1"],
    } for r in res]).pivot(index="date", columns="tgt_ret_str", values="rpm_tp1").fillna(0.)
    if start_date is not None:
        seed_date = pd.Timestamp(start_date)
    else:
        first_date = df_res_pivot.index.min()
        if realized_freq == "weekly":
            seed_date = first_date - pd.offsets.Week(1)
        else:
            seed_date = first_date - pd.offsets.MonthEnd(1)
    if seed_date not in df_res_pivot.index:
        df_res_pivot.loc[seed_date] = 0.
    df_res_pivot.sort_index(ascending=True, inplace=True)

    df_port_values = (100000 * (1 + df_res_pivot).cumprod()).reset_index()
    df_port_values.columns.name = None
    pe = portfolio_analyzer(freq=realized_freq)
    df_performs = pe.get_portfolio_metrics(df_ports=df_port_values)
    df_performs.loc["Turnover (%)", df_metrics_turnover.index] = (
        df_metrics_turnover * annualization_factor(realized_freq) * 100.0
    )
    df_performs.loc["HHI", df_metrics_hhi.index] = df_metrics_hhi

    df_performs.to_csv(f"{performance_path}/portfolios_performance.csv", float_format="%.4f")
    df_port_values.to_csv(f"{performance_path}/portfolios_value.csv", float_format="%.2f", index=False)

    # return sum of sharps
    return df_performs.loc[feat].sum().item()


if __name__ == "__main__":
    run_backtest(cov_name="HC", results_folder="../optuna/prcs", performance_prefix="../perform_optuna", tc_cost=10)
