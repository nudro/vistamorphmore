#!/usr/bin/env bash
# Sequential Diffusion_B training: Mild -> Moderate -> Severe -> LITIV
# Does not start itself unless invoked. Override GPU_NUM / BATCH_SIZE via env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Diffusion_B small-tiers series starting"
bash "${SCRIPT_DIR}/train_flir_mild.sh"
bash "${SCRIPT_DIR}/train_flir_moderate.sh"
bash "${SCRIPT_DIR}/train_flir_severe.sh"
bash "${SCRIPT_DIR}/train_litiv.sh"
echo "==> Diffusion_B small-tiers series complete"
