#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"

if [ -n "${PYTHON_BIN:-}" ]; then
  if ! "$PYTHON_BIN" -m pybind11 --cmakedir >/dev/null 2>&1; then
    echo "Configured PYTHON_BIN=$PYTHON_BIN does not have pybind11 installed." >&2
    exit 1
  fi
else
  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -m pybind11 --cmakedir >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  echo "Could not find a Python interpreter with pybind11 installed. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

if [ -n "${EIGEN3_INCLUDE_DIR:-}" ] && [ -d "$EIGEN3_INCLUDE_DIR/Eigen" ]; then
  :
elif [ -d /opt/homebrew/include/eigen3 ]; then
  EIGEN3_INCLUDE_DIR="/opt/homebrew/include/eigen3"
elif [ -d /usr/local/include/eigen3 ]; then
  EIGEN3_INCLUDE_DIR="/usr/local/include/eigen3"
elif [ -d /usr/include/eigen3 ]; then
  EIGEN3_INCLUDE_DIR="/usr/include/eigen3"
else
  echo "Eigen3 headers not found. Install Eigen or set EIGEN3_INCLUDE_DIR." >&2
  exit 1
fi

PYBIND11_CMAKE_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"
PYTHON_EXECUTABLE="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

rm -rf "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  -DEIGEN3_INCLUDE_DIR="$EIGEN3_INCLUDE_DIR" \
  -DPython_EXECUTABLE="$PYTHON_EXECUTABLE"

cmake --build "$BUILD_DIR" --config Release

MODULE_PATH="$(find "$BUILD_DIR" -maxdepth 1 -type f \( -name 'cgerber*.so' -o -name 'cgerber*.pyd' -o -name 'cgerber*.dylib' \) | head -n 1)"
if [ -z "$MODULE_PATH" ]; then
  echo "Compiled module not found in $BUILD_DIR" >&2
  exit 1
fi

PACKAGE_DIR="$ROOT/../mgs"
mkdir -p "$PACKAGE_DIR"
cp "$MODULE_PATH" "$PACKAGE_DIR"
COPIED_MODULE="$PACKAGE_DIR/$(basename "$MODULE_PATH")"
if [ "$(uname -s)" = "Darwin" ] && command -v codesign >/dev/null 2>&1; then
  # Re-sign after copying so macOS does not reject the locally rebuilt bundle.
  codesign --force --sign - "$COPIED_MODULE"
fi
echo "Copied $(basename "$MODULE_PATH") to $PACKAGE_DIR"
