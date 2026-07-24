#!/usr/bin/env bash
# One-shot environment setup for the lab.
#
# What it does:
#   1. Creates a Python 3.11 virtual environment and installs env/requirements.txt.
#   2. Checks that Ollama is reachable (does NOT auto-pull models — see note below).
#   3. Writes env/hardware_report.txt so runs are traceable to the machine they ran on.
#
# Usage:
#   bash env/setup.sh              # deps + hardware report, checks Ollama
#   bash env/setup.sh --pull-models  # additionally pull the local LLMs (several GB)
#
# The model pull is opt-in on purpose: the reasoning/coder models total well over 10 GB and
# are only needed from the semantic-audit phase onward, so we don't download them during the
# data-foundation phase.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PULL_MODELS=0
[[ "${1:-}" == "--pull-models" ]] && PULL_MODELS=1

# --- 1. Python environment -------------------------------------------------
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] >= (3, 11), "Python 3.11+ required"'

if [[ ! -d .venv ]]; then
    "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
if [[ -f .venv/Scripts/activate ]]; then
    source .venv/Scripts/activate   # Windows (Git Bash)
else
    source .venv/bin/activate       # POSIX
fi

python -m pip install --upgrade pip
python -m pip install -r env/requirements.txt

# --- 2. Ollama reachability ------------------------------------------------
# Models used later, kept here so the required tags live in one place.
REASONING_MODEL="qwen2.5:7b-instruct-q4_K_M"
CODER_MODEL="qwen2.5-coder:14b-instruct-q4_K_M"

if command -v ollama >/dev/null 2>&1; then
    if ollama list >/dev/null 2>&1; then
        echo "Ollama is running."
        if [[ "$PULL_MODELS" == "1" ]]; then
            ollama pull "$REASONING_MODEL"
            ollama pull "$CODER_MODEL"
        else
            echo "Skipping model pull. Run with --pull-models when you reach the LLM phases."
            echo "Required: $REASONING_MODEL, $CODER_MODEL"
        fi
    else
        echo "WARNING: Ollama is installed but not running. Start it before the LLM phases."
    fi
else
    echo "WARNING: Ollama not found. It is only needed from the semantic-audit phase onward."
fi

# --- 3. Hardware report ----------------------------------------------------
python env/hardware_report.py > env/hardware_report.txt
echo "Wrote env/hardware_report.txt"
echo "Setup complete. Activate with: source .venv/Scripts/activate (Windows) or .venv/bin/activate"
