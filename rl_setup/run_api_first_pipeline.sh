#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-rl_setup/config/api_first.json}"

uv run python rl_setup/cli.py split --config "$CONFIG"
uv run python rl_setup/cli.py episodes --config "$CONFIG"
uv run python rl_setup/cli.py score --config "$CONFIG"
uv run python rl_setup/cli.py preferences --config "$CONFIG"
uv run python rl_setup/cli.py reject --config "$CONFIG"
uv run python rl_setup/cli.py report --config "$CONFIG"
uv run python rl_setup/cli.py export-trl --config "$CONFIG"
uv run python rl_setup/cli.py check --config "$CONFIG"
