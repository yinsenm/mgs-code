# Three-month Treasury bill series

The code uses FRED series `TB3MS` as a monthly, annualized percentage yield to compute cash returns and Sharpe ratios. For an offline or frozen reproduction, place `tb3ms_19900101_20251231.csv` here with columns:

```text
date,TB3MS
1990-01-31,<annualized-percent-yield>
...
2025-12-31,<annualized-percent-yield>
```

Use month-end dates and the rate in **percent**, not decimal units; for example, a 5% yield is recorded as `5.0`. Cover every month from January 1990 through December 2025. The analyzers convert an annual yield to a monthly return as `(1 + TB3MS/100)^(1/12) - 1`. If the file is absent, the Python `pandas-datareader` path queries FRED and caches a local CSV, so resulting numbers can change if FRED revises historical observations. Keeping the frozen local file is preferable for exact comparisons. The risk-free data are not tracked by Git.
