#!/bin/bash
# r5 lock-in: firm+thick 칙(train=eval 일치)으로 fresh DAgger 수집 → 체인(r4 warm) 재학습 → eval.
# 현 dead_chick.xml = firm-hold(마찰6/관성4e-4) + thick(캡슐0.018). 수집·학습·평가 모두 동일 물리.
set -e
OFT=/home/capstone/openvla-oft; LIBDIR=$OFT/experiments/robot/libero
OFTPY=/home/capstone/miniconda3/envs/oft/bin; RUNDIR=/home/capstone/openvla_ckpts/runs_oft
CKPT2A="$RUNDIR/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt"
R1=$(ls -d "$RUNDIR"/*dagger-r1*--10000_chkpt|head -1); R2=$(ls -d "$RUNDIR"/*dagger-r2*--10000_chkpt|head -1)
R4=$(ls -d "$RUNDIR"/*dagger-r4rel*--8000_chkpt|head -1)
DEMOS=/home/capstone/openvla/demos/blue_chick_dagger_r5
ENVV="BLUE_CHICK_THERMAL=1 WANDB_MODE=offline PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
gpu_wait(){ while true; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$U" -lt "$1" ] && break; echo "[pipeline] GPU busy ${U}MiB"; sleep 120; done; }

echo "[pipeline] $(date) r5 stage 0: merge chain (r4<-r2<-r1<-2a)"
ls "$R1"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$CKPT2A" --lora_finetuned_checkpoint_dir "$R1")
ls "$R2"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R1" --lora_finetuned_checkpoint_dir "$R2")
rm -f "$R1"/model-*.safetensors "$R1"/model.safetensors.index.json
ls "$R4"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R2" --lora_finetuned_checkpoint_dir "$R4")
# r4 standalone merged (벽 base로 r5 학습·이후 병합에 사용) — r2 정리
rm -f "$R2"/model-*.safetensors "$R2"/model.safetensors.index.json

echo "[pipeline] $(date) r5 stage 1: collect on firm+thick (r4, expert-on-holding, only-grasped, 1ch48+2ch16)"
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R4" --unnorm-key blue_chick_dagger \
  --beta 0.3 --n-dead 1 --expert-on-holding --only-grasped --seeds 18000-18099 --n-good 36 --out-dir "$DEMOS"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R4" --unnorm-key blue_chick_dagger \
  --beta 0.3 --n-dead 2 --expert-on-holding --only-grasped --seeds 18100-18159 --n-good 12 --out-dir "$DEMOS"
echo "[pipeline] r5 collected: $(ls "$DEMOS"/*.h5 2>/dev/null|wc -l)"

echo "[pipeline] $(date) r5 stage 2: RLDS 5.0.0 build"
rm -rf /home/capstone/tensorflow_datasets/blue_chick_dagger/4.0.0
cd /home/capstone/rlds_builders/blue_chick_dagger && "$OFTPY/tfds" build --data_dir /home/capstone/tensorflow_datasets

echo "[pipeline] $(date) r5 stage 3: train from r4 (warm), 10k"
gpu_wait 10000; cd "$OFT"
WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$RUNDIR/r4base" --data_root_dir /home/capstone/tensorflow_datasets --dataset_name blue_chick_dagger \
  --run_root_dir "$RUNDIR" --num_images_in_input 2 --use_proprio True \
  --batch_size 8 --learning_rate 5e-4 --max_steps 10000 --save_freq 5000 \
  --merge_lora_during_training False --image_aug True --use_lora True --lora_rank 32 --run_id_note dagger-r5firm

SNAP=$(ls -d "$RUNDIR"/*dagger-r5firm*--10000_chkpt|head -1); [ -z "$SNAP" ] && exit 3
echo "[pipeline] $(date) r5 stage 4: merge(base=r4) + eval firm+thick 18seed"
cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R4" --lora_finetuned_checkpoint_dir "$SNAP"
rm -f "$R4"/model-*.safetensors "$R4"/model.safetensors.index.json
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" --unnorm-key blue_chick_dagger \
  --handoff-xy 0 --n-dead 1 --seeds 12000-12025 --n-good 18 \
  --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r5firm_solo.jsonl" \
  --save-video-dir /home/capstone/openvla/rollouts/2026_06_14_r5 || true
rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
echo "[pipeline] $(date) r5 DONE"
