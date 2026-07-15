#!/usr/bin/env bash
# VMM/Diffusion_B_VMM_Github TRES+LDM on Data/VMM/all_warps
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
if [[ -d "${SCRIPT_DIR}/../../Data" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [[ -d "${SCRIPT_DIR}/../Data" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data/VMM/all_warps}"
export EXPERIMENT="${EXPERIMENT:-diffusion_b_all_warps}"
export N_EPOCHS="${N_EPOCHS:-210}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
export GPU_NUM="${GPU_NUM:-0}"
exec bash "${SCRIPT_DIR}/run_train.sh"
