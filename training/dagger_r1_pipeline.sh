#!/bin/bash
# DAgger round 1 연쇄 파이프라인: on-policy 수집 → RLDS 빌드 → OFT 재학습 → solo 평가.
# 사용: nohup bash dagger_r1_pipeline.sh > /home/capstone/openvla_ckpts/runs_oft/dagger_r1_pipeline.log 2>&1 &
set -e
OFT=/home/capstone/openvla-oft
LIBDIR=$OFT/experiments/robot/libero
OFTPY=/home/capstone/miniconda3/envs/oft/bin
RUNDIR=/home/capstone/openvla_ckpts/runs_oft
CKPT2A="$RUNDIR/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt"
DEMOS=/home/capstone/openvla/demos/blue_chick_dagger_r1
ENVV="BLUE_CHICK_THERMAL=1 WANDB_MODE=offline PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"

echo "[pipeline] $(date) stage 1: DAgger collection (beta=0.35, 32x 1-chick + 16x 2-chick)"
cd "$LIBDIR"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$CKPT2A" --beta 0.35 \
    --n-dead 1 --seeds 11000-11079 --n-good 32 --out-dir "$DEMOS"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$CKPT2A" --beta 0.35 \
    --n-dead 2 --seeds 11100-11159 --n-good 16 --out-dir "$DEMOS"
echo "[pipeline] collected: $(ls "$DEMOS" | wc -l) episodes"

FREE_G=$(df -BG --output=avail /home/capstone | tail -1 | tr -dc '0-9')
[ "${FREE_G:-0}" -lt 33 ] && { echo "[pipeline] ABORT: ${FREE_G}G free (<33G: RLDS ~14G + merge 15G)"; exit 2; }

echo "[pipeline] $(date) stage 2: RLDS build (blue_chick_dagger = v3 demos + corrections)"
cd /home/capstone/rlds_builders/blue_chick_dagger
"$OFTPY/tfds" build --data_dir /home/capstone/tensorflow_datasets

echo "[pipeline] $(date) stage 3: OFT finetune from 2a-30000 (10k steps, b8)"
cd "$OFT"
WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path "$CKPT2A" \
    --data_root_dir /home/capstone/tensorflow_datasets \
    --dataset_name blue_chick_dagger \
    --run_root_dir "$RUNDIR" \
    --num_images_in_input 2 --use_proprio True \
    --batch_size 8 --learning_rate 5e-4 --max_steps 10000 --save_freq 2500 \
    --merge_lora_during_training False --image_aug True --use_lora True --lora_rank 32 \
    --run_id_note dagger-r1

SNAP=$(ls -d "$RUNDIR"/*dagger-r1*--10000_chkpt 2>/dev/null | head -1)
[ -z "$SNAP" ] && { echo "[pipeline] ABORT: 10000_chkpt not found"; exit 3; }
echo "[pipeline] $(date) stage 4: merge final snapshot $SNAP"
FREE_G=$(df -BG --output=avail /home/capstone | tail -1 | tr -dc '0-9')
[ "${FREE_G:-0}" -lt 18 ] && { echo "[pipeline] ABORT: ${FREE_G}G free (<18G) before merge"; exit 2; }
cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint "$CKPT2A" --lora_finetuned_checkpoint_dir "$SNAP"

echo "[pipeline] $(date) stage 5: VLA-solo grasp eval (결정적 지표) + hybrid eval"
cd "$LIBDIR"
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" \
    --unnorm-key blue_chick_dagger --handoff-xy 0 --n-dead 1 \
    --seeds 12000-12017 --n-good 8 \
    --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r1_solo_1chick.jsonl" \
    --save-video-dir /home/capstone/openvla/rollouts/2026_06_12_dagger || true
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" \
    --unnorm-key blue_chick_dagger --handoff-xy 0.05 --handoff-consec 3 --n-dead 1 \
    --seeds 12100-12117 --n-good 8 \
    --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r1_hybrid_1chick.jsonl" || true

echo "[pipeline] $(date) stage 6: reclaim disk (merged safetensors 삭제, lora_adapter 보존)"
rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
echo "[pipeline] $(date) DONE"
