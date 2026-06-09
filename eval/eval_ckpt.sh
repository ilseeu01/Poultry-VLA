#!/usr/bin/env bash
# 범용 평가: 병합 체크포인트로 양계장 task(only_task_id 0) 평가. 롤아웃 MP4 -> ./rollouts/{DATE}/
# 사용: eval_ckpt.sh <CKPT_DIR> [NUM_TRIALS] [MAX_STEPS] [NOTE]
set -euo pipefail

CKPT="${1:?Usage: eval_ckpt.sh <CKPT_DIR> [NUM_TRIALS] [MAX_STEPS] [NOTE]}"
NUM_TRIALS="${2:-10}"
MAX_STEPS="${3:-700}"
NOTE="${4:-eval}"

EVAL_LOG_DIR="/home/capstone/openvla_ckpts/eval_logs"
mkdir -p "$EVAL_LOG_DIR"

source /home/capstone/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd /home/capstone/openvla

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TF_CPP_MIN_LOG_LEVEL=3

LOG="${EVAL_LOG_DIR}/${NOTE}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] eval ckpt=$CKPT trials=$NUM_TRIALS max_steps=$MAX_STEPS -> $LOG"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name libero_object \
  --only_task_id 0 \
  --num_trials_per_task "$NUM_TRIALS" \
  --max_steps_override "$MAX_STEPS" \
  --unnorm_key blue_chick_thermal \
  --apply_thermal_fx True \
  --center_crop True \
  --run_id_note "$NOTE" \
  --use_wandb False \
  --seed 7 \
  2>&1 | tee "$LOG"

echo "[RESULT] $(grep -E 'Current total success rate' "$LOG" | tail -1)"
