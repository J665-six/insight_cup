#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
INSIGHT_CUP_BIN="${INSIGHT_CUP_BIN:-$BUILD_DIR/insight-cup}"
BUILD_JOBS="${BUILD_JOBS:-2}"

cd "$PROJECT_ROOT"

# The affected RK3588 image can crash Xorg when NoMachine keeps its automatic
# encoder/realtime defaults. Refuse to start in a graphical session until the
# one-time root configuration has been applied.
if [[ "${INSIGHT_CUP_SKIP_NOMACHINE_CHECK:-0}" != 1 &&
      -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" &&
      -x "$PROJECT_ROOT/tools/systemd/check-nomachine-safe.sh" ]]; then
  "$PROJECT_ROOT/tools/systemd/check-nomachine-safe.sh" || {
    printf 'Refusing to start while NoMachine is configured unsafely.\n' >&2
    printf 'For a deliberate headless/diagnostic run, set INSIGHT_CUP_SKIP_NOMACHINE_CHECK=1.\n' >&2
    exit 78
  }
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Do not let OpenCV create a large second thread pool while NoMachine/GNOME is
# encoding the desktop. Override explicitly for a headless benchmark.
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
export INSIGHT_CUP_OPENCV_THREADS="${INSIGHT_CUP_OPENCV_THREADS:-1}"

profile=""
profile_seen=0
legacy_runtime_option_seen=0
competition_conflict=""
ui_recording_conflict=""
requested_provider=""
args=()
input_args=("$@")
for ((index = 0; index < ${#input_args[@]}; ++index)); do
    arg="${input_args[index]}"
    case "$arg" in
        --profile)
            if (( profile_seen )); then
                printf '%s\n' 'start.sh: --profile may only be specified once' >&2
                exit 2
            fi
            if (( index + 1 >= ${#input_args[@]} )); then
                printf '%s\n' 'start.sh: --profile requires a value' >&2
                exit 2
            fi
            profile="${input_args[++index]}"
            profile_seen=1
            ;;
        --profile=*)
            if (( profile_seen )); then
                printf '%s\n' 'start.sh: --profile may only be specified once' >&2
                exit 2
            fi
            profile="${arg#--profile=}"
            profile_seen=1
            ;;
        --provider)
            legacy_runtime_option_seen=1
            args+=("$arg")
            if (( index + 1 >= ${#input_args[@]} )); then
                printf '%s\n' 'start.sh: --provider requires a value' >&2
                exit 2
            fi
            requested_provider="${input_args[++index]}"
            args+=("$requested_provider")
            ;;
        --provider=*)
            legacy_runtime_option_seen=1
            requested_provider="${arg#--provider=}"
            args+=("$arg")
            ;;
        --ui|--no-ui)
            legacy_runtime_option_seen=1
            args+=("$arg")
            [[ "$arg" == --ui ]] && competition_conflict="${competition_conflict:-$arg}"
            [[ "$arg" == --no-ui ]] && ui_recording_conflict="${ui_recording_conflict:-$arg}"
            ;;
        --save-video|--no-save-video)
            legacy_runtime_option_seen=1
            args+=("$arg")
            [[ "$arg" == --save-video ]] && competition_conflict="${competition_conflict:-$arg}"
            [[ "$arg" == --no-save-video ]] && ui_recording_conflict="${ui_recording_conflict:-$arg}"
            ;;
        --save-crops|--no-save-crops)
            legacy_runtime_option_seen=1
            args+=("$arg")
            [[ "$arg" == --save-crops ]] && competition_conflict="${competition_conflict:-$arg}"
            ;;
        --open-browser|--shadow-log|--show-raw-label|--start-paused|--verbose)
            legacy_runtime_option_seen=1
            competition_conflict="${competition_conflict:-$arg}"
            args+=("$arg")
            ;;
        --no-open-browser|--no-shadow-log)
            legacy_runtime_option_seen=1
            args+=("$arg")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

if (( ! profile_seen )); then
    if (( legacy_runtime_option_seen )); then
        profile="custom"
    else
        profile="competition"
    fi
fi

case "$profile" in
    competition)
        if [[ -n "$competition_conflict" ]]; then
            printf 'start.sh: --profile competition forbids %s\n' \
                "$competition_conflict" >&2
            exit 2
        fi
        if [[ -n "$requested_provider" && "$requested_provider" != rknn ]]; then
            printf '%s\n' 'start.sh: --profile competition requires --provider rknn' >&2
            exit 2
        fi
        ;;
    ui-recording)
        if [[ -n "$ui_recording_conflict" ]]; then
            printf 'start.sh: --profile ui-recording forbids %s\n' \
                "$ui_recording_conflict" >&2
            exit 2
        fi
        ;;
    custom)
        ui_option_seen=0
        save_video_option_seen=0
        save_crops_option_seen=0
        for arg in "${args[@]}"; do
            case "$arg" in
                --ui|--no-ui) ui_option_seen=1 ;;
                --save-video|--no-save-video) save_video_option_seen=1 ;;
                --save-crops|--no-save-crops) save_crops_option_seen=1 ;;
            esac
        done
        (( ui_option_seen )) || args+=(--no-ui)
        (( save_video_option_seen )) || args+=(--no-save-video)
        (( save_crops_option_seen )) || args+=(--no-save-crops)
        ;;
    *)
        printf 'start.sh: unsupported profile: %s\n' "$profile" >&2
        exit 2
        ;;
esac
args+=(--profile "$profile")

if [[ ! -x "$INSIGHT_CUP_BIN" ]]; then
  "$PROJECT_ROOT/build_cpp.sh"
elif [[ "$INSIGHT_CUP_BIN" == "$BUILD_DIR/insight-cup" ]]; then
    nice -n "${BUILD_NICE:-10}" cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" >/dev/null
fi

if [[ "$profile" == competition ]]; then
    exec "$INSIGHT_CUP_BIN" "${args[@]}"
fi

# UI/custom sessions leave scheduling headroom for the graphical desktop.
exec nice -n "${INSIGHT_CUP_NICE:-10}" ionice -c 3 \
    "$INSIGHT_CUP_BIN" "${args[@]}"
