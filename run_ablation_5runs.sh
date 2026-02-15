#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# You can override these without editing the file:
#   FID_DEVICE=cuda FID_NUM_SAMPLES=5000 ./run_ablation_5runs.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
FID_DEVICE="${FID_DEVICE:-cpu}"          # cpu|cuda
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-1000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-30}"
T_STEPS="${T_STEPS:-1000}"
MAX_BATCHES="${MAX_BATCHES:-0}"          # <=0 means full epoch
MAX_SAMPLES="${MAX_SAMPLES:-0}"          # <=0 means full dataset
VIS_EVERY="${VIS_EVERY:-1}"

COMMON=(
  --mode train --dataset cifar10
  --epochs "${EPOCHS}"
  --max-batches "${MAX_BATCHES}" --max-samples "${MAX_SAMPLES}"
  --run-dir runs
  --batch-size 64 --ema-decay 0.9999
  --T "${T_STEPS}" --vis-every "${VIS_EVERY}" --vis-fixed-noise
  --fid-every 0
  --fid-num-samples "${FID_NUM_SAMPLES}"
  --fid-batch-size "${FID_BATCH_SIZE}"
  --fid-use-ema
  --fid-device "${FID_DEVICE}"
)

run_one () {
  local run_name="$1"
  shift
  echo "============================================================"
  echo "RUN: ${run_name}"
  echo "============================================================"

  mkdir -p "runs/${run_name}"
  # Tee logs into the run folder (ddpm.py will create its own unique run dir if name collides).
  "${PYTHON_BIN}" ddpm.py \
    "${COMMON[@]}" \
    --run-name "${run_name}" \
    "$@" 2>&1 | tee "runs/${run_name}/stdout.log"
}

# v0: Anchor
run_one v0_anchor_cosine_attn_blocks2_C128 \
  --lr 1e-4 --schedule-type cosine \
  --unet-chs 128,256,256,256 --unet-time-dim 128 \
  --unet-num-blocks 2 --unet-num-groups 32 --unet-attn-heads 4 --unet-use-attn

# v1: No attention
run_one v1_noattn_blocks2_C128 \
  --lr 1e-4 --schedule-type cosine \
  --unet-chs 128,256,256,256 --unet-time-dim 128 \
  --unet-num-blocks 2 --unet-num-groups 32 --unet-attn-heads 4 --unet-no-attn

# v2: Deeper blocks
run_one v2_attn_blocks3_C128 \
  --lr 1e-4 --schedule-type cosine \
  --unet-chs 128,256,256,256 --unet-time-dim 128 \
  --unet-num-blocks 3 --unet-num-groups 32 --unet-attn-heads 4 --unet-use-attn

# v3: Smaller width C=64, with P1 lr scaling
run_one v3_attn_blocks2_C64_lr1p41e-4 \
  --lr 1.41e-4 --schedule-type cosine \
  --unet-chs 64,128,128,128 --unet-time-dim 128 \
  --unet-num-blocks 2 --unet-num-groups 32 --unet-attn-heads 4 --unet-use-attn

# v4: Linear schedule
run_one v4_linear_attn_blocks2_C128 \
  --lr 1e-4 --schedule-type linear \
  --unet-chs 128,256,256,256 --unet-time-dim 128 \
  --unet-num-blocks 2 --unet-num-groups 32 --unet-attn-heads 4 --unet-use-attn

echo "All runs finished."
echo "Summarize:"
echo "  ${PYTHON_BIN} tools/summarize_sweep.py"
