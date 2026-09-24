# Local inputs (not committed)

No market data, prepared panels, risk-free-rate files, or generated results belong in this repository. Put your licensed input files here, using the paths below. The `.gitignore` excludes the input files and the derived `sp500_weekly_panel/` directory while keeping these Markdown instructions trackable.

| Input | Used for | Preparation guide |
| --- | --- | --- |
| `n10_daily.xlsx` | 10-asset backtests and four-week divergence | [10-asset panel](n10.md) |
| `sp500_1950_2025/sp500_weekly_features.csv` | Point-in-time top-30 stock backtests | [S&P 500 weekly features](sp500.md) |
| `tb3ms_19900101_20251231.csv` | Cash return and Sharpe ratios | [Risk-free rate](risk_free.md) |

The scripts can fetch TB3MS from FRED when an appropriate local cache is absent, but the other source files must be supplied by the researcher. An independently sourced file may yield different numeric results because of vendor revisions, total-return conventions, security identifiers, or point-in-time universe definitions.

Check the inputs before a full run:

```bash
python scripts/validate_data.py --study all
```

Run that command from the repository root after installation. It checks input columns and date coverage, plus the named 10-asset securities; it does not validate the data vendor's point-in-time construction or upload data.
