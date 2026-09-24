#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
COV_IMPL="${COV_IMPL:-cpp}"
export PYTHON_BIN

case "$COV_IMPL" in
  cpp|python) ;;
  *) echo "COV_IMPL must be cpp or python" >&2; exit 2 ;;
esac
"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.10+ required; activate the project environment or set PYTHON_BIN") if sys.version_info < (3, 10) else None'

"$PYTHON_BIN" scripts/validate_data.py --study n10 --data-dir "$DATA_DIR"
if [[ "$COV_IMPL" == cpp ]] && ! PYTHONPATH="$REPO_ROOT/mgs" "$PYTHON_BIN" -c 'import mgs.cgerber' >/dev/null 2>&1; then
  ./mgs/cpp/run_compile.sh
fi

"$PYTHON_BIN" scripts/run_n10_backtests.py \
  --data-path "$DATA_DIR/n10_daily.xlsx" --rf-cache-dir "$DATA_DIR" --cov-impl "$COV_IMPL"
"$PYTHON_BIN" scripts/run_n10_divergence.py \
  --data-path "$DATA_DIR/n10_daily.xlsx" --rf-cache-dir "$DATA_DIR" --cov-impl "$COV_IMPL"
"$PYTHON_BIN" scripts/build_n10_paper_artifacts.py

echo "Multi-asset table: $REPO_ROOT/results/tables/table_multi_asset_performance.csv"
