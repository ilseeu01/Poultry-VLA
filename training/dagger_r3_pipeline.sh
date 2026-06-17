#!/bin/bash
set -e
OFT=/home/capstone/openvla-oft; LIBDIR=$OFT/experiments/robot/libero
OFTPY=/home/capstone/miniconda3/envs/oft/bin; RUNDIR=/home/capstone/openvla_ckpts/runs_oft
CKPT2A="$RUNDIR/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt"
R1=$(ls -d "$RUNDIR"/*dagger-r1*--10000_chkpt|head -1); R2=$(ls -d "$RUNDIR"/*dagger-r2*--10000_chkpt|head -1)
DEMOS=/home/capstone/openvla/demos/blue_chick_dagger_r3
ENVV="BLUE_CHICK_THERMAL=1 WANDB_MODE=offline PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
gpu_wait(){ while true; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$U" -lt "$1" ] && break; echo "[pipeline] GPU busy ${U}MiB"; sleep 120; done; }
echo "[pipeline] $(date) r3 stage 0: RLDS 2.0.0 삭제(재생성가능) + r2 병합 체인"
rm -rf /home/capstone/tensorflow_datasets/blue_chick_dagger/2.0.0
ls "$R1"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$CKPT2A" --lora_finetuned_checkpoint_dir "$R1")
ls "$R2"/model-*.safetensors >/dev/null 2>&1 || (cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R1" --lora_finetuned_checkpoint_dir "$R2")
rm -f "$R1"/model-*.safetensors "$R1"/model.safetensors.index.json
echo "[pipeline] $(date) r3 stage 1: collect with r2 model (beta 0.25)"
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R2" --unnorm-key blue_chick_dagger --beta 0.25 --n-dead 1 --seeds 15000-15079 --n-good 32 --out-dir "$DEMOS"
env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R2" --unnorm-key blue_chick_dagger --beta 0.25 --n-dead 2 --seeds 15100-15159 --n-good 16 --out-dir "$DEMOS"
echo "[pipeline] r3 collected: $(ls "$DEMOS"|wc -l)"
rm -f "$R2"/model-*.safetensors "$R2"/model.safetensors.index.json
echo "[pipeline] $(date) r3 stage 2: RLDS 3.0.0 build (v3+r1+r2+r3 집계)"
cd /home/capstone/rlds_builders/blue_chick_dagger && "$OFTPY/tfds" build --data_dir /home/capstone/tensorflow_datasets
echo "[pipeline] $(date) r3 stage 3: canonical DAgger — base(2a)에서 집계로 15k 학습"
gpu_wait 10000; cd "$OFT"
WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path "$CKPT2A" --data_root_dir /home/capstone/tensorflow_datasets --dataset_name blue_chick_dagger --run_root_dir "$RUNDIR" --num_images_in_input 2 --use_proprio True --batch_size 8 --learning_rate 5e-4 --max_steps 15000 --save_freq 5000 --merge_lora_during_training False --image_aug True --use_lora True --lora_rank 32 --run_id_note dagger-r3agg
SNAP=$(ls -d "$RUNDIR"/*dagger-r3agg*--15000_chkpt|head -1); [ -z "$SNAP" ] && exit 3
FREE=$(df -BG --output=avail /home/capstone|tail -1|tr -dc '0-9'); [ "$FREE" -lt 17 ] && { echo "[pipeline] ABORT ${FREE}G"; exit 2; }
echo "[pipeline] $(date) r3 stage 4: merge + solo eval"
cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$CKPT2A" --lora_finetuned_checkpoint_dir "$SNAP"
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" --unnorm-key blue_chick_dagger --handoff-xy 0 --n-dead 1 --seeds 16000-16017 --n-good 8 --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r3_solo_1chick.jsonl" --save-video-dir /home/capstone/openvla/rollouts/2026_06_13_dagger_r3 || true
rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
echo "[pipeline] $(date) r3 DONE"
