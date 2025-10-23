#!/usr/bin/env bash
#
# Convenience wrapper that runs all one-time setup helpers in order:
#   1. Copy Windows assets
#   2. Synchronize API keys
#   3. Validate the Windows <-> WSL bridge
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"${SCRIPT_DIR}/setup_move_windows_components.sh"
"${SCRIPT_DIR}/setup_sync_api_keys.sh"
"${SCRIPT_DIR}/setup_bridge_qc.sh"

echo "All setup routines complete."
