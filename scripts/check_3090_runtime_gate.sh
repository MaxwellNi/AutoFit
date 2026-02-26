#!/usr/bin/env bash
# =============================================================================
# 3090 runtime gate wrapper
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/check_4090_runtime_gate.sh" --host-label=3090 "$@"
