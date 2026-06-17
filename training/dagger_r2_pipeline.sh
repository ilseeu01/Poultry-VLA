#!/bin/bash
set -e
OFT=/home/capstone/openvla-oft; LIBDIR=$OFT/experiments/robot/libero
OFTPY=/home/capstone/miniconda3/envs/oft/bin; RUNDIR=/home/capstone/openvla_ckpts/runs_oft
R1="/home/capstone/openvla_ckpts/runs_oft/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt+blue_chick_dagger+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--dagger-r1--10000_chkpt"
DEMOS=/home/capstone/openvla/demos/blue_chick_dagger_r2
ENVV="BLUE_CHICK_THERMAL=1 WANDB_MODE=offline PYTHONPATH=/home/capstone/LIBERO MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
gpu_wait(){ while true; do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$U" -lt "$1" ] && break; echo "[pipeline] GPU busy ${U}MiB, wait 120s"; sleep 120; done; }
echo "[pipeline] $(date) r2 stage 0: re-merge r1-10000"
ls "$R1"/model-*.safetensors >/dev/null 2>&1 || { cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$RUNDIR/openvla-7b-finetuned-libero-object+blue_chick_thermal+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--stage2a-wrist-proprio--30000_chkpt" --lora_finetuned_checkpoint_dir "$R1"; }
N=$(ls "$DEMOS" 2>/dev/null|wc -l); [ "$N" -ge 45 ] && SKIPCOL=1 || SKIPCOL=0
echo "[pipeline] $(date) r2 stage 1: collect with r1 model (beta 0.25)"
gpu_wait 50000; cd "$LIBDIR"
[ "$SKIPCOL" = 0 ] && env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R1" --unnorm-key blue_chick_dagger --beta 0.25 --n-dead 1 --seeds 13000-13079 --n-good 32 --out-dir "$DEMOS"
[ "$SKIPCOL" = 0 ] && env $ENVV "$OFTPY/python" dagger_collect.py --ckpt "$R1" --unnorm-key blue_chick_dagger --beta 0.25 --n-dead 2 --seeds 13100-13159 --n-good 16 --out-dir "$DEMOS"
echo "[pipeline] r2 collected: $(ls "$DEMOS"|wc -l)"
echo "[pipeline] $(date) r2 stage 2: RLDS 2.0.0 build"
cd /home/capstone/rlds_builders/blue_chick_dagger && "$OFTPY/tfds" build --data_dir /home/capstone/tensorflow_datasets
rm -rf /home/capstone/tensorflow_datasets/blue_chick_dagger/1.0.0
echo "[pipeline] $(date) r2 stage 3: train 10k from r1"
gpu_wait 10000; cd "$OFT"
WANDB_MODE=offline "$OFTPY/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path "$RUNDIR/r1base" --data_root_dir /home/capstone/tensorflow_datasets --dataset_name blue_chick_dagger --run_root_dir "$RUNDIR" --num_images_in_input 2 --use_proprio True --batch_size 8 --learning_rate 5e-4 --max_steps 10000 --save_freq 2500 --merge_lora_during_training False --image_aug True --use_lora True --lora_rank 32 --run_id_note dagger-r2
SNAP=$(ls -d "$RUNDIR"/*dagger-r2*--10000_chkpt|head -1); [ -z "$SNAP" ] && exit 3
FREE=$(df -BG --output=avail /home/capstone|tail -1|tr -dc '0-9'); [ "$FREE" -lt 17 ] && { echo "[pipeline] ABORT ${FREE}G"; exit 2; }
echo "[pipeline] $(date) r2 stage 4: merge r2 (base=r1)"
cd "$OFT" && "$OFTPY/python" vla-scripts/merge_lora_weights_and_save.py --base_checkpoint "$R1" --lora_finetuned_checkpoint_dir "$SNAP"
rm -f "$R1"/model-*.safetensors "$R1"/model.safetensors.index.json
echo "[pipeline] $(date) r2 stage 5: solo eval"
gpu_wait 50000; cd "$LIBDIR"
env $ENVV "$OFTPY/python" hybrid_eval.py --policy vla --ckpt "$SNAP" --unnorm-key blue_chick_dagger --handoff-xy 0 --n-dead 1 --seeds 14000-14017 --n-good 8 --grasp-log "$RUNDIR/eval_wd_sweep/dagger_r2_solo_1chick.jsonl" --save-video-dir /home/capstone/openvla/rollouts/2026_06_12_dagger_r2 || true
rm -f "$SNAP"/model-*.safetensors "$SNAP"/model.safetensors.index.json
echo "[pipeline] $(date) r2 DONE"
