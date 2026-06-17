#!/bin/bash
# r4 targeted: release-descend 신호 농축. r2로 expert-on-holding 수집(파지성공만) →
# 4x 복제(단일 정규화 유지) → v3+r1+r2+r4(×4) 4.0.0 → r2 warm-start 8k → solo 평가.
set -e
OFT=/home/capstone/openvla-oft; LIBDIR=$OFT/experiments/robot/libero
OFTPY=/home/capstone/miniconda3/envs/oft/bin; RUNDIR=/home/capstone/openvla_ckpts/runs_oft
CKPT2A="$RUNDIR/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt"
R1=$(ls -d "$RUNDIR"/*dagger-r1*--10000_chkpt|head -1)
R2=$(ls -d "$RUNDIR"/*dagger-r2*--10000_chkpt|head -1)
DEMOS=/home/capstone/openvla/demos/blue_chick_dagger_r4
ENVV="BLUE_CHICK_THERMAL=1 WANDB_MODE=offline PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
gpu_wait(){ while true; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$U" -lt "$1" ] && break; echo "[pipeline] GPU busy ${U}MiB"; sleep 120; done; }

echo "[pipeline] $(date) r4 stage 0: rebuild merge chain (r1<-2a, r2<-r1)"
ls "$R1"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$CKPT2A" --lora_finetuned_checkpoint_dir "$R1")
ls "$R2"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R1" --lora_finetuned_checkpoint_dir "$R2")

echo "[pipeline] $(date) r4 stage 1: collect release-focused (r2, expert-on-holding, only-grasped)"
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R2" --unnorm-key blue_chick_dagger \
  --beta 0.2 --n-dead 1 --expert-on-holding --only-grasped \
  --seeds 17000-17099 --n-good 40 --out-dir "$DEMOS"
rm -f "$R1"/model-*.safetensors "$R1"/model.safetensors.index.json
echo "[pipeline] r4 collected: $(ls "$DEMOS"/*.h5 2>/dev/null|wc -l)"

echo "[pipeline] $(date) r4 stage 1b: 4x 복제 (단일 정규화 내 oversample)"
cd "$DEMOS"
for f in dagger_demo_seed*.h5; do
  case "$f" in *_x*.h5) continue;; esac
  for k in 2 3 4; do ln -sf "$f" "${f%.h5}_x${k}.h5"; done
done
echo "[pipeline] r4 after replicate: $(ls "$DEMOS"/*.h5|wc -l) (orig*4 기대)"

echo "[pipeline] $(date) r4 stage 2: RLDS 4.0.0 build (v3+r1+r2+r4x4)"
rm -rf /home/capstone/tensorflow_datasets/blue_chick_dagger/3.0.0
cd /home/capstone/rlds_builders/blue_chick_dagger && "$OFTPY/tfds" build --data_dir /home/capstone/tensorflow_datasets

echo "[pipeline] $(date) r4 stage 3: train from r2 (warm), 8k"
gpu_wait 10000; cd "$OFT"
WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$RUNDIR/r2base" --data_root_dir /home/capstone/tensorflow_datasets --dataset_name blue_chick_dagger \
  --run_root_dir "$RUNDIR" --num_images_in_input 2 --use_proprio True \
  --batch_size 8 --learning_rate 5e-4 --max_steps 8000 --save_freq 4000 \
  --merge_lora_during_training False --image_aug True --use_lora True --lora_rank 32 --run_id_note dagger-r4rel

SNAP=$(ls -d "$RUNDIR"/*dagger-r4rel*--8000_chkpt|head -1); [ -z "$SNAP" ] && exit 3
echo "[pipeline] $(date) r4 stage 4: merge(base=r2) + solo eval 12000s"
cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R2" --lora_finetuned_checkpoint_dir "$SNAP"
rm -f "$R2"/model-*.safetensors "$R2"/model.safetensors.index.json
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" --unnorm-key blue_chick_dagger --handoff-xy 0 --n-dead 1 --seeds 12000-12013 --n-good 10 --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r4rel_solo_1chick.jsonl" --save-video-dir /home/capstone/openvla/rollouts/2026_06_13_dagger_r4 || true
rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
echo "[pipeline] $(date) r4 DONE"
