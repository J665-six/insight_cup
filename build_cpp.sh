#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
# BUILD_JOBS is intentionally conservative because this script is commonly
# run from the NoMachine desktop on the RK3588.
BUILD_JOBS="${BUILD_JOBS:-2}"

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_PREFIX_PATH="/opt/ros/humble${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
nice -n "${BUILD_NICE:-10}" cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

printf 'Built: %s\n' "$BUILD_DIR/insight-cup"
