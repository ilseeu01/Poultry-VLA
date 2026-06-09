#!/usr/bin/env bash
# v3 풀 학습: 200개(v2+thermal_100) no-op 필터 RLDS로 30,000 step LoRA fine-tune.
# - 어댑터 스냅샷 방식(2500스텝마다 ~0.5GB, 15GB merge 생략) → 디스크 안전 + 빠름
# - 가짜 resume 아님: 베이스에서 한 번에 30k.
set -euo pipefail

MAX_STEPS="${1:-30000}"
SAVE_STEPS="${2:-2500}"

BASE="/home/capstone/openvla_ckpts/openvla-7b-finetuned-libero-object"
DATA_ROOT="/home/capstone/tensorflow_datasets"
RUN_ROOT="/home/capstone/openvla_ckpts/runs"
ADAPTER_TMP="/home/capstone/openvla_ckpts/adapter-tmp"
LOG="/home/capstone/openvla_ckpts/logs/train_v3_$(date +%Y%m%d_%H%M%S).log"

source /home/capstone/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd /home/capstone/openvla

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

echo "[INFO] v3 train: max_steps=$MAX_STEPS save_steps=$SAVE_STEPS -> $LOG"
mkdir -p "$RUN_ROOT" "$ADAPTER_TMP"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$BASE" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name blue_chick_thermal \
  --run_root_dir "$RUN_ROOT" \
  --adapter_tmp_dir "$ADAPTER_TMP" \
  --batch_size 16 \
  --learning_rate 5e-4 \
  --lora_rank 32 \
  --lora_dropout 0.0 \
  --image_aug True \
  --shuffle_buffer_size 100000 \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --adapter_snapshots True \
  --run_id_note v3b-smooth1chick \
  2>&1 | tee "$LOG"

echo "[DONE] log: $LOG"
echo "어댑터 스냅샷: ${RUN_ROOT}/*v3-200noop*--{step}_adapter/"
