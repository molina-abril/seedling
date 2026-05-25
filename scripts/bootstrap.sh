#!/usr/bin/env bash

set -euo pipefail

log() {
  printf '[bootstrap] %s\n' "$1"
}

is_macos() {
  [[ "$(uname -s)" == "Darwin" ]]
}

brew_prefix() {
  brew --prefix "$1"
}

llvm20_prefix() {
  brew --prefix llvm@20
}

ensure_cmake() {
  if command -v cmake >/dev/null 2>&1; then
    log "cmake already installed: $(cmake --version | head -n 1)"
    return 0
  fi

  if ! is_macos; then
    log "cmake is missing and automatic installation is only supported on macOS."
    log "Install it manually with your system package manager, then rerun this script."
    exit 1
  fi

  if ! command -v brew >/dev/null 2>&1; then
    log "Homebrew is not available. Install Homebrew first, then rerun this script."
    log "https://brew.sh/"
    exit 1
  fi

  log "cmake not found; installing with Homebrew..."
  brew install cmake
}

ensure_llvm() {
  if command -v llvm-config >/dev/null 2>&1; then
    if [[ "$(llvm-config --version | cut -d. -f1)" == "20" ]]; then
      log "llvm-config already available: $(llvm-config --version)"
      return 0
    fi
    log "Found unsupported LLVM version $(llvm-config --version); llvmlite requires LLVM 20."
  fi

  if ! is_macos; then
    log "LLVM is missing and automatic installation is only supported on macOS."
    log "Install LLVM manually with your system package manager, then rerun this script."
    exit 1
  fi

  if ! command -v brew >/dev/null 2>&1; then
    log "Homebrew is not available. Install Homebrew first, then rerun this script."
    log "https://brew.sh/"
    exit 1
  fi

  if ! brew list --formula llvm@20 >/dev/null 2>&1; then
    log "Installing LLVM 20 with Homebrew..."
    brew install llvm@20
  else
    log "LLVM 20 already installed with Homebrew."
  fi

  local llvm_prefix
  llvm_prefix="$(llvm20_prefix)"
  export PATH="${llvm_prefix}/bin:${PATH}"
  export LLVM_CONFIG="${llvm_prefix}/bin/llvm-config"
  export CMAKE_PREFIX_PATH="${llvm_prefix}:${CMAKE_PREFIX_PATH:-}"
  log "Using LLVM 20 from ${llvm_prefix}"
}

install_python_dependencies() {
  local py="${PYTHON:-$HOME/miniforge3/envs/seedling/bin/python}"
  if [[ ! -x "$py" ]]; then
    log "ERROR: conda env Python not found at: $py"
    log "Create the env with: conda create -n seedling python=3.12 -y"
    log "(or export PYTHON=/path/to/python before running this script)"
    exit 1
  fi
  log "Installing Python dependencies into: $py"
  "$py" -m pip install --upgrade pip setuptools wheel
  "$py" -m pip install -r requirements.txt
}

main() {
  log "Checking system prerequisites..."
  ensure_cmake
  ensure_llvm
  install_python_dependencies
  log "Environment bootstrap complete. Dependencies are installed."
}

main "$@"