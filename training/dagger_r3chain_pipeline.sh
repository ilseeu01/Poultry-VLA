#!/bin/bash
# r3 chained fix: r2에서 warm-start로 집계(3.0.0) 10k 재학습 → solo 평가.
# canonical-cold(r3agg)가 no-op 붕괴 → 검증된 체인 레시피로 복귀. 재수집/재빌드 없음.
set -e
OFT=/home/capstone/openvla-oft; LIBDIR=$OFT/experiments/robot/libero
OFTPY=/home/capstone/miniconda3/envs/oft/bin; RUNDIR=/home/capstone/openvla_ckpts/runs_oft
CKPT2A="$RUNDIR/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt"
R1=$(ls -d "$RUNDIR"/*dagger-r1*--10000_chkpt|head -1)
R2=$(ls -d "$RUNDIR"/*dagger-r2*--10000_chkpt|head -1)
ENVV="BLUE_CHICK_THERMAL=1 WANDB_MODE=offline PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
gpu_wait(){ while true; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$U" -lt "$1" ] && break; echo "[pipeline] GPU busy ${U}MiB"; sleep 120; done; }

echo "[pipeline] $(date) r3chain stage 0: rebuild merge chain (r1←2a, r2←r1)"
ls "$R1"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$CKPT2A" --lora_finetuned_checkpoint_dir "$R1")
ls "$R2"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R1" --lora_finetuned_checkpoint_dir "$R2")
rm -f "$R1"/model-*.safetensors "$R1"/model.safetensors.index.json   # r2 now standalone-merged

echo "[pipeline] $(date) r3chain stage 1: train from r2 (warm-start), aggregate 3.0.0, 10k"
gpu_wait 10000; cd "$OFT"
WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$RUNDIR/r2base" --data_root_dir /home/capstone/tensorflow_datasets --dataset_name blue_chick_dagger \
  --run_root_dir "$RUNDIR" --num_images_in_input 2 --use_proprio True \
  --batch_size 8 --learning_rate 5e-4 --max_steps 10000 --save_freq 5000 \
  --merge_lora_during_training False --image_aug True --use_lora True --lora_rank 32 --run_id_note dagger-r3chain

SNAP=$(ls -d "$RUNDIR"/*dagger-r3chain*--10000_chkpt|head -1); [ -z "$SNAP" ] && exit 3
echo "[pipeline] $(date) r3chain stage 2: merge (base=r2) + solo eval (seeds 12000s, 검증범위)"
cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R2" --lora_finetuned_checkpoint_dir "$SNAP"
rm -f "$R2"/model-*.safetensors "$R2"/model.safetensors.index.json
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" --unnorm-key blue_chick_dagger --handoff-xy 0 --n-dead 1 --seeds 12000-12011 --n-good 8 --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r3chain_solo_1chick.jsonl" --save-video-dir /home/capstone/openvla/rollouts/2026_06_13_dagger_r3chain || true
rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
echo "[pipeline] $(date) r3chain DONE"
