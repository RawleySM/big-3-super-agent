#!/usr/bin/env bash
#
# One-time bridge validation script. Ensures the Windows and WSL
# components can communicate, runs a smoke test, reports the outcome,
# and performs a clean shutdown.
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

WINDOWS_TARGET_ROOT=${WINDOWS_TARGET_ROOT:-/mnt/c/users/rstanhope}
WINDOWS_TARGET_DIR=${WINDOWS_TARGET_DIR:-"${WINDOWS_TARGET_ROOT}/big-3-super-agent-windows"}
WSL_DISTRO=${WSL_DISTRO:-Ubuntu}

if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "[setup_bridge_qc] powershell.exe is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -d "${WINDOWS_TARGET_DIR}" ]]; then
  echo "[setup_bridge_qc] Windows target directory not found: ${WINDOWS_TARGET_DIR}" >&2
  exit 1
fi

WINDOWS_TARGET_DIR_WIN=$(wslpath -w "${WINDOWS_TARGET_DIR}")
START_SCRIPT_WIN="${WINDOWS_TARGET_DIR_WIN}\\scripts\\start_agents.ps1"
SHUTDOWN_BAT_WIN="${WINDOWS_TARGET_DIR_WIN}\\scripts\\Agent_shutdown.bat"

if [[ ! -f "${WINDOWS_TARGET_DIR}/big_3_Windows.py" ]]; then
  echo "[setup_bridge_qc] big_3_Windows.py missing from ${WINDOWS_TARGET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${WINDOWS_TARGET_DIR}/scripts/start_agents.ps1" ]]; then
  echo "[setup_bridge_qc] start_agents.ps1 missing from ${WINDOWS_TARGET_DIR}/scripts" >&2
  exit 1
fi

echo "Running WSL standalone agent smoke test..."
python3 "${REPO_ROOT}/apps/realtime-poc/big_3_WSL.py" --bridge-call list_agents >/tmp/big3_wsl_smoke.json
cat /tmp/big3_wsl_smoke.json

echo "Running Windows bridge smoke test via PowerShell..."
POWERSHELL_CMD=(
  powershell.exe
  -NoProfile
  -ExecutionPolicy
  Bypass
  -File
  "${START_SCRIPT_WIN}"
  -SmokeTest
  -WSLDistro
  "${WSL_DISTRO}"
)
"${POWERSHELL_CMD[@]}"

echo "Smoke test complete. Inspecting results..."
if [[ -f /tmp/big3_wsl_smoke.json ]]; then
  if grep -q '"agents"' /tmp/big3_wsl_smoke.json; then
    echo "WSL agent registry reachable."
  else
    echo "WSL agent registry output unexpected:" >&2
    cat /tmp/big3_wsl_smoke.json >&2
  fi
fi

echo "Invoking shutdown routine..."
cmd.exe /c "\"${SHUTDOWN_BAT_WIN}\"" 2>/dev/null || true

echo "Bridge validation finished."
rm -f /tmp/big3_wsl_smoke.json
