#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" && -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "$PROJECT_ROOT"
if [[ "${INSIGHT_CUP_SKIP_NOMACHINE_CHECK:-0}" != 1 &&
      -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" &&
      -x "$PROJECT_ROOT/tools/systemd/check-nomachine-safe.sh" ]]; then
  "$PROJECT_ROOT/tools/systemd/check-nomachine-safe.sh" || {
    printf 'Refusing to start while NoMachine is configured unsafely.\n' >&2
    printf 'For a deliberate headless/diagnostic run, set INSIGHT_CUP_SKIP_NOMACHINE_CHECK=1.\n' >&2
    exit 78
  }
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"

ui_option_seen=0
args=()
for arg in "$@"; do
  case "$arg" in
    --ui|--no-ui)
      ui_option_seen=1
      args+=("$arg")
      ;;
    *) args+=("$arg") ;;
  esac
done
if (( ! ui_option_seen )); then
  args+=(--no-ui)
fi

exec nice -n "${INSIGHT_CUP_NICE:-10}" ionice -c 3 "$PYTHON_BIN" -m insight_cup "${args[@]}"
