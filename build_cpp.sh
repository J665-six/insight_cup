#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_PREFIX_PATH="/opt/ros/humble${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

printf 'Built: %s\n' "$BUILD_DIR/insight-cup"
