#!/usr/bin/env bash
# Train Diffusion_B (latent DDPM + ViT+GNN reg + phase-1 struct-B losses).
# Usage (from this package root, or set DATA_ROOT):
#   bash run_train.sh
#   bash train_allwarps.sh
#   DATA_ROOT=/path/to/tier bash run_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Prefer an enclosing workspace that has Data/; else this package root (standalone clone).
WORKSPACE="${SCRIPT_DIR}"
if [[ -d "${SCRIPT_DIR}/../../Data" ]]; then
  WORKSPACE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [[ -d "${SCRIPT_DIR}/../Data" ]]; then
  WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

DATA_ROOT="${DATA_ROOT:-${WORKSPACE}/Data/VMM/all_warps}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
EXPERIMENT="${EXPERIMENT:-diffusion_b_run}"
GPU_NUM="${GPU_NUM:-0}"

LAMBDA_DDPM="${LAMBDA_DDPM:-1.0}"
LAMBDA_LPIPS_A="${LAMBDA_LPIPS_A:-0.2}"
LAMBDA_STRUCT_B="${LAMBDA_STRUCT_B:-1.0}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-1.0}"
LATENT_HF_RADIUS="${LATENT_HF_RADIUS:-0.25}"
LAMBDA_EPS_LF="${LAMBDA_EPS_LF:-1.0}"
LAMBDA_EPS_HF="${LAMBDA_EPS_HF:-1.0}"
LAMBDA_LATENT_HF="${LAMBDA_LATENT_HF:-0.0}"
LATENT_HF_T_MAX="${LATENT_HF_T_MAX:--1}"
LAMBDA_FFT="${LAMBDA_FFT:-0.25}"
LAMBDA_EO_GRAPH="${LAMBDA_EO_GRAPH:-0.0}"
PHASE="${PHASE:-3}"
N_EPOCHS="${N_EPOCHS:-210}"
EPOCH="${EPOCH:-0}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-1e-4}"
LR_REG="${LR_REG:-}"
DDPM_T="${DDPM_T:-500}"
DDIM_STEPS="${DDIM_STEPS:-32}"
LATENT_CH="${LATENT_CH:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
REG_START_EPOCH="${REG_START_EPOCH:-100}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
VIT_PATCH_SIZE="${VIT_PATCH_SIZE:-32}"
SLIC_BACKEND="${SLIC_BACKEND:-diff}"
SLIC_SEGMENTS="${SLIC_SEGMENTS:-98}"

if [[ ! -d "${DATA_ROOT}/train" ]]; then
  echo "error: expected train split at ${DATA_ROOT}/train" >&2
  echo "  Set DATA_ROOT to a tier folder with train/ and test/." >&2
  exit 1
fi

echo "==> Diffusion_B TRES+LDM"
echo "    DATA_ROOT=${DATA_ROOT}  EXPERIMENT=${EXPERIMENT}"

TRAIN_ARGS=(
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --experiment "${EXPERIMENT}"
  --epoch "${EPOCH}"
  --batch_size "${BATCH_SIZE}"
  --n_epochs "${N_EPOCHS}"
  --lr "${LR}"
  --gpu_num "${GPU_NUM}"
  --lambda_ddpm "${LAMBDA_DDPM}"
  --lambda_lpips_a "${LAMBDA_LPIPS_A}"
  --lambda_struct_b "${LAMBDA_STRUCT_B}"
  --lambda_lpips "${LAMBDA_LPIPS}"
  --latent_hf_radius "${LATENT_HF_RADIUS}"
  --lambda_eps_lf "${LAMBDA_EPS_LF}"
  --lambda_eps_hf "${LAMBDA_EPS_HF}"
  --lambda_latent_hf "${LAMBDA_LATENT_HF}"
  --latent_hf_t_max "${LATENT_HF_T_MAX}"
  --lambda_fft "${LAMBDA_FFT}"
  --lambda_eo_graph "${LAMBDA_EO_GRAPH}"
  --phase "${PHASE}"
  --ddpm_T "${DDPM_T}"
  --ddim_steps "${DDIM_STEPS}"
  --latent_ch "${LATENT_CH}"
  --sample_interval "${SAMPLE_INTERVAL}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --reg_start_epoch "${REG_START_EPOCH}"
  --grad_clip "${GRAD_CLIP}"
  --vit_patch_size "${VIT_PATCH_SIZE}"
  --slic_backend "${SLIC_BACKEND}"
  --slic_segments "${SLIC_SEGMENTS}"
)
if [[ -n "${LR_REG}" ]]; then
  TRAIN_ARGS+=(--lr_reg "${LR_REG}")
fi

python "${SCRIPT_DIR}/train.py" "${TRAIN_ARGS[@]}"

echo "==> Done. Checkpoints: ${OUTPUT_DIR}/${EXPERIMENT}/checkpoints"
echo "    TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tb --reload_interval 5"
