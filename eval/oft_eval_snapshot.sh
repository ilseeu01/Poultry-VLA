#!/bin/bash
# Evaluate one OFT Stage-2a snapshot on the blue_chick task.
# Usage: oft_eval_snapshot.sh <step> [num_trials] [max_steps] [--keep-merged]
# e.g.:  oft_eval_snapshot.sh 15000 10 400
#
# Merges the LoRA snapshot in-place (base + lora_adapter -> merged safetensors,
# ~15GB added to the snapshot dir), then runs the blue_chick eval with the
# train-matched observation pipeline (BLUE_CHICK_THERMAL=1: [::-1] + thermal@256).
# By default deletes the merged safetensors afterward to reclaim disk (keeps
# lora_adapter so it can be re-merged). Pass --keep-merged to keep them.
set -e
STEP="${1:?usage: oft_eval_snapshot.sh <step> [trials] [maxsteps] [--keep-merged]}"
TRIALS="${2:-10}"
MAXS="${3:-400}"
KEEP="${4:-}"

OFT=/home/capstone/openvla-oft
RUNDIR=/home/capstone/openvla_ckpts/runs_oft
BASE=/home/capstone/oft_base/openvla-7b-finetuned-libero-object
OFTPY=/home/capstone/miniconda3/envs/oft/bin
SNAP=$(ls -d "$RUNDIR"/*stage2a*--"${STEP}"_chkpt 2>/dev/null | head -1)
[ -z "$SNAP" ] && { echo "snapshot for step $STEP not found in $RUNDIR"; exit 1; }
echo "snapshot: $SNAP"

# --- disk guard ---
FREE_G=$(df -BG --output=avail /home/capstone | tail -1 | tr -dc '0-9')
if [ "${FREE_G:-0}" -lt 25 ]; then echo "ABORT: only ${FREE_G}G free (<25G); merge needs ~15G"; exit 2; fi

# 1) merge LoRA in-place if not already merged
if ! ls "$SNAP"/model-*.safetensors >/dev/null 2>&1; then
  echo "=== merging LoRA (base + adapter -> $SNAP) ==="
  cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint "$BASE" --lora_finetuned_checkpoint_dir "$SNAP"
else
  echo "merged weights already present, skipping merge"
fi

# 2) eval (train-matched obs: thermal + vertical-flip-only; multiview + proprio)
echo "=== eval: task 0 (blue_chick), $TRIALS trials, max_steps $MAXS ==="
cd "$OFT" && BLUE_CHICK_THERMAL=1 PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 \
  experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$SNAP" \
  --task_suite_name libero_object --only_task_id 0 \
  --num_images_in_input 2 --use_proprio True --center_crop True \
  --unnorm_key blue_chick_thermal \
  --num_trials_per_task "$TRIALS" --max_steps_override "$MAXS" \
  --env_img_res 256

# 3) reclaim disk unless --keep-merged
if [ "$KEEP" != "--keep-merged" ]; then
  echo "=== removing merged safetensors (keeping lora_adapter) ==="
  rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
fi
echo "done: step $STEP"
