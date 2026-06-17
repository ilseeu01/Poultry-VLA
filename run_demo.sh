#!/usr/bin/env bash
# =====================================================================
# Poultry-VLA — 빠른 데모: 양계장 씬 1개 생성 후 스크립트 컨트롤러로
# 파란(죽은) 병아리 pick-and-place 를 실행하고 IR 열화상 mp4 를 저장한다.
#
# 사용:  bash run_demo.sh [SEED] [OUT_DIR]
#   예:  bash run_demo.sh 2000 demo_out
# =====================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_DIR="${LIBERO_DIR:-$REPO_DIR/third_party/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
# LIBERO editable 설치 매핑이 비어 import 안 되는 경우가 있어 PYTHONPATH 명시.
export PYTHONPATH="$LIBERO_DIR:${PYTHONPATH:-}"

SEED="${1:-2000}"
OUT_DIR="${2:-$REPO_DIR/demo_out}"
BDDL="_demo_seed${SEED}.bddl"
mkdir -p "$OUT_DIR"

echo "==> 씬 생성 (seed=$SEED): 살아있는 병아리 7 + 죽은 병아리 1"
python "$LIBERO_DIR/scripts/generate_chicken_farm_bddl.py" \
    --n-chicks 7 --n-hens 0 --n-dead 1 \
    --n-straw 50 --n-feed 15 --n-manure 4 --n-straw-visual 1000 \
    --out "$BDDL" --seed "$SEED"

echo "==> 스크립트 컨트롤러 실행 → IR 열화상 mp4"
python "$REPO_DIR/data_pipeline/scripted_blue_chick.py" \
    --bddl "$BDDL" --seed "$SEED" \
    --max-steps 1100 --resolution 256 --fps 30 \
    --thermal --moving-chicks \
    --out-dir "$OUT_DIR"

echo "==> 완료. mp4: $OUT_DIR/"
ls -t "$OUT_DIR"/*.mp4 2>/dev/null | head -1
