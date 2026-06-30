#!/usr/bin/env bash
#
# install.sh — create a Python 3.12 virtual environment and install dependencies.
#
# Usage:
#   ./install.sh            # create .venv and install requirements
#   ./install.sh --dev      # also install dev/test extras (currently same as base)
#   ./install.sh --force    # recreate the venv even if it already exists
#
set -euo pipefail

# --- configuration ----------------------------------------------------------
PYTHON_VERSION="3.12"
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

# Resolve the directory this script lives in, so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- argument parsing --------------------------------------------------------
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --dev)   ;; # reserved; base requirements already include dev tools
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Run './install.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

# --- helpers -----------------------------------------------------------------
info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
error() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; }

# --- locate a Python 3.12 interpreter ----------------------------------------
find_python() {
    # Prefer the explicitly versioned interpreter.
    if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
        echo "python${PYTHON_VERSION}"
        return 0
    fi
    # Fall back to python3 / python only if it reports 3.12.x.
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            local ver
            ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
            if [ "$ver" = "$PYTHON_VERSION" ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

info "Looking for Python ${PYTHON_VERSION}..."
if ! PYTHON_BIN="$(find_python)"; then
    error "Could not find a Python ${PYTHON_VERSION} interpreter on your PATH."
    error "Install it first, e.g.:"
    error "  macOS (Homebrew):  brew install python@${PYTHON_VERSION}"
    error "  pyenv:             pyenv install ${PYTHON_VERSION} && pyenv local ${PYTHON_VERSION}"
    exit 1
fi
PYTHON_PATH="$(command -v "$PYTHON_BIN")"
PYTHON_FULL_VERSION="$("$PYTHON_BIN" --version 2>&1)"
info "Using interpreter: ${PYTHON_PATH} [${PYTHON_FULL_VERSION}]"

# --- check the requirements file exists --------------------------------------
if [ ! -f "$REQUIREMENTS" ]; then
    error "Requirements file '$REQUIREMENTS' not found in $SCRIPT_DIR."
    exit 1
fi

# --- create (or recreate) the virtual environment ----------------------------
if [ -d "$VENV_DIR" ]; then
    if [ "$FORCE" -eq 1 ]; then
        info "Removing existing virtual environment '$VENV_DIR' (--force)..."
        rm -rf "$VENV_DIR"
    else
        warn "Virtual environment '$VENV_DIR' already exists; reusing it."
        warn "Pass --force to recreate it from scratch."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# --- install dependencies ----------------------------------------------------
# Use the venv's interpreter directly rather than relying on activation.
VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
    error "Expected interpreter '$VENV_PY' was not created. The venv may be corrupt."
    error "Try removing '$VENV_DIR' and re-running with --force."
    exit 1
fi

info "Upgrading pip, setuptools, and wheel..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

info "Installing dependencies from $REQUIREMENTS..."
"$VENV_PY" -m pip install -r "$REQUIREMENTS"

# --- set up .env if needed ---------------------------------------------------
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    info "Creating .env from .env.example (remember to fill in your credentials)..."
    cp .env.example .env
fi

# --- done --------------------------------------------------------------------
info "Done! Activate the environment with:"
echo
echo "    source ${VENV_DIR}/bin/activate"
echo
info "Then run the classifier, e.g.:"
echo
echo "    python classify.py \\"
echo "      --input data/example_queries.csv \\"
echo "      --column text \\"
echo "      --output data/example_queries_classified.csv \\"
echo "      --categories resources/categories/example_support_tickets.json"
echo
