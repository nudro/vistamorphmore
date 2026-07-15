#!/usr/bin/env bash
# VMM/Diffusion_B_VMM_Github on Data/all_pairs — phased curriculum smoke + reg
# Phase 1 (epochs 0-49):  DDPM + LPIPS_A + struct-B (ld trains)
# Phase 2 (epochs 50-99): ld frozen (inference-only); STN reg aligns/registers
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
if [[ -d "${SCRIPT_DIR}/../../Data" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [[ -d "${SCRIPT_DIR}/../Data" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Data/all_pairs}"
export EXPERIMENT="${EXPERIMENT:-diffusion_b_all_pairs_p1_50_reg_100}"
export N_EPOCHS="${N_EPOCHS:-100}"
export REG_START_EPOCH="${REG_START_EPOCH:-50}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
# Checkpoints every 10 epochs: phase1 @ 10..50, phase2 @ 50..100
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
export GRAD_CLIP="${GRAD_CLIP:-1.0}"
export GPU_NUM="${GPU_NUM:-0}"
export LAMBDA_LPIPS_A="${LAMBDA_LPIPS_A:-1.0}"
export LAMBDA_STRUCT_B="${LAMBDA_STRUCT_B:-0.2}"
export LAMBDA_LPIPS="${LAMBDA_LPIPS:-1.0}"
exec bash "${SCRIPT_DIR}/run_train.sh"
