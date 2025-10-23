#!/usr/bin/env bash
#
# Prepare the Windows-side assets for the Big 3 realtime agent stack.
# This script is intended to be executed from the WSL workspace at
#   ~/dev/big-3-super-agent/
# It copies the Windows-specific controller and helper scripts into a
# consolidated directory under the Windows user profile.
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

WINDOWS_TARGET_ROOT=${WINDOWS_TARGET_ROOT:-/mnt/c/users/rstanhope}
WINDOWS_TARGET_DIR=${WINDOWS_TARGET_DIR:-"${WINDOWS_TARGET_ROOT}/big-3-super-agent-windows"}

WINDOWS_FILES=(
  "apps/realtime-poc/big_3_Windows.py"
  "windows/start_agents.ps1"
  "windows/Agent.bat"
  "windows/Agent_shutdown.bat"
)
WINDOWS_DIRS=(
  "apps/realtime-poc/prompts"
)

mkdir -p "${WINDOWS_TARGET_DIR}"

helper_dir="${WINDOWS_TARGET_DIR}/scripts"
mkdir -p "${helper_dir}"

for file in "${WINDOWS_FILES[@]}"; do
  src="${REPO_ROOT}/${file}"
  if [[ ! -f "${src}" ]]; then
    echo "[setup_move_windows_components] Missing source file: ${src}" >&2
    exit 1
  fi

  if [[ "${file}" == apps/realtime-poc/* ]]; then
    dest="${WINDOWS_TARGET_DIR}/$(basename "${file}")"
  else
    dest="${helper_dir}/$(basename "${file}")"
  fi

  install -m 0644 "${src}" "${dest}"
  echo "Copied $(basename "${file}") -> ${dest}"

  if [[ "${file}" == *".ps1" || "${file}" == *".bat" ]]; then
    if command -v unix2dos >/dev/null 2>&1; then
      unix2dos -q "${dest}" || true
    fi
  fi

  if [[ "${file}" == *"big_3_Windows.py" ]]; then
    chmod +x "${dest}"
  fi
done

for dir in "${WINDOWS_DIRS[@]}"; do
  src_dir="${REPO_ROOT}/${dir}"
  if [[ ! -d "${src_dir}" ]]; then
    echo "[setup_move_windows_components] Missing source directory: ${src_dir}" >&2
    exit 1
  fi
  dest_dir="${WINDOWS_TARGET_DIR}/$(basename "${dir}")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${src_dir}/" "${dest_dir}/"
  else
    rm -rf "${dest_dir}"
    mkdir -p "${dest_dir}"
    cp -a "${src_dir}/." "${dest_dir}/"
  fi
  echo "Mirrored ${dir} -> ${dest_dir}"
  chmod -R u+rwX "${dest_dir}"

done

echo "Windows components are available in: ${WINDOWS_TARGET_DIR}"
