# Modified Gerber Statistic — main paper examples

This repository reproduces the two **main performance examples** in the *Modified Gerber Statistic* SSRN manuscript (SSRN 7190778): the 10-asset multi-asset study and the point-in-time S&P 500 top-30 study. It retains the paper's `n=1`, `gamma=1`, no-decay specifications and the HC, SM, and GS comparators. Appendix sensitivity sweeps, paired significance tests, and the polar illustration are intentionally excluded.

There are two end-to-end entry points, both run from the repository root (or by absolute path):

```bash
./run_multi_asset.sh
./run_sp500.sh
```

Each validates its inputs, builds the C++ covariance extension if needed, runs the relevant Python backtests, and writes the main performance table as CSV and LaTeX. These commands can take several minutes. Existing allocation and performance files are reused on reruns. Neither script uploads data or commits results.

## 1. Prepare the data

Market data are **not included**. Put licensed files under `data/` as documented in [data/README.md](data/README.md), [data/n10.md](data/n10.md), [data/sp500.md](data/sp500.md), and [data/risk_free.md](data/risk_free.md):

```text
data/n10_daily.xlsx
data/sp500_1950_2025/sp500_weekly_features.csv
data/tb3ms_19900101_20251231.csv
```

The Treasury-bill cache is optional: the Python code can fetch TB3MS from FRED, but a frozen local file is preferable for reproduction. The S&P 500 file must contain *point-in-time* constituents and ranks, not a list of today's constituents. Input files, prepared panels, and results are ignored by Git.

If the inputs live elsewhere, set `DATA_DIR` to the directory containing `n10_daily.xlsx` and the `sp500_1950_2025/` subdirectory. Each entry point checks only the input for its own study, plus the risk-free cache when present:

```bash
DATA_DIR=/absolute/path/to/data ./run_multi_asset.sh
DATA_DIR=/absolute/path/to/data ./run_sp500.sh
```

## 2. Install the environment

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './mgs[cpp]'
```

The paper's C++ MGS/GS estimator requires CMake, a C++17 compiler, and Eigen3 headers. Install Eigen3 with your system package manager; if its headers are in a nonstandard location, set `EIGEN3_INCLUDE_DIR` to the directory containing `Eigen/`. The bash entry points compile the extension on first use. The allocator tries GUROBI, then ECOS, then SCS. A licensed Gurobi installation gives the closest match to the original run; solver and library versions can change the last displayed digit. To use the slower pure-Python covariance estimator instead, run either entry point with `COV_IMPL=python`.

The scripts use `python3` by default. Set `PYTHON_BIN` when using another interpreter:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" ./run_multi_asset.sh
```

## 3. What each entry point produces

`run_multi_asset.sh` runs the 10-asset HC, SM, GS, and return-only MGS baselines, followed by the equal-weight return-plus-four-week-divergence MGS overlay. It uses weekly covariance signals, a trailing 36-calendar-month window, monthly rebalancing from January 1990 through December 2025, residual cash, and 10 bps transaction costs. Its main table is [results/tables/table_multi_asset_performance.csv](results/tables/table_multi_asset_performance.csv), with a matching `.tex` file.

`run_sp500.sh` prepares a local monthly panel from the weekly point-in-time stock file, then runs HC, SM, GS, return-only MGS, MGS+Div, and MGS+RVOL. It selects up to 30 stocks by contemporaneous market-cap rank, uses a trailing 60-calendar-month signal window and the same dates, cash treatment, and transaction cost. It tests annualized volatility targets from 3% through 21% in 3-point steps. Its main table is [results/tables/table_large_cap_multisignal_performance.csv](results/tables/table_large_cap_multisignal_performance.csv), with a matching `.tex` file. `PREPARED_DATA_PATH` can override the location of the generated monthly stock panel.

Intermediate allocations and portfolio series are saved under `results/n10/`, `results/n10_divergence/`, `results/sp500_baselines/`, and `results/sp500_signals/`. All `results/` content remains local.

The submitted stock table combines historical optimizer runs: HC, SM, and GS use the paper-era optimizer, while MGS-family rows use its guarded revision. The runners select these versions explicitly. The guarded version allows a `2e-6` absolute variance residual (or 0.1% of target variance, whichever is larger) so minor cross-platform solver differences do not silently omit a rebalance month. The stock runners fail if any expected allocation file is missing.

## 4. Compare with the manuscript

At the 9% target, the manuscript reports annualized geometric returns of 11.26% for multi-asset MGS, 11.04% for GS, and 11.25% for MGS+Div; the stock example reports 9.15% for MGS, 8.42% for GS, 9.76% for MGS+Div, and 9.26% for MGS+RVOL. With the original local input files, a complete run produced 432 allocations per method and 431 monthly return observations. The largest geometric-return differences from the submitted main tables were 0.006 and 0.008 percentage points for the multi-asset and stock examples, respectively. This is a close numeric reproduction, not byte-for-byte equality; independently sourced data or different solvers may differ more.
