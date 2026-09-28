#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
INSIGHT_CUP_BIN="${INSIGHT_CUP_BIN:-$BUILD_DIR/insight-cup}"

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-4}"

ui_option_seen=0
args=()
for arg in "$@"; do
    case "$arg" in
        --ui|--no-ui)
            ui_option_seen=1
            args+=("$arg")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

if (( ! ui_option_seen )); then
    args+=(--no-ui)
fi

if [[ ! -x "$INSIGHT_CUP_BIN" ]]; then
    "$PROJECT_ROOT/build_cpp.sh"
elif [[ "$INSIGHT_CUP_BIN" == "$BUILD_DIR/insight-cup" ]]; then
    cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-$(nproc)}" >/dev/null
fi

exec "$INSIGHT_CUP_BIN" "${args[@]}"
