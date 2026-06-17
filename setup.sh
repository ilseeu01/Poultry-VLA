#!/usr/bin/env bash
# =====================================================================
# Poultry-VLA — 스크립트 시뮬레이션 클론-실행 셋업
#
#   클론 직후 이 스크립트 한 번이면 양계장 시뮬레이션(데모 수집/렌더)이
#   돌아간다. LIBERO 를 핀된 커밋으로 설치하고, 양계장 객체/씬/BDDL/문제
#   정의를 주입하고, robosuite 시각 버퍼를 패치한다.
#
#   검증 환경: Python 3.10, Linux, conda/venv. 렌더 백엔드는 osmesa.
#   GPU 불필요(스크립트 컨트롤러는 ground-truth grasp 기반).
#
# 사용:
#   bash setup.sh                 # third_party/LIBERO 에 설치
#   LIBERO_DIR=/path/LIBERO bash setup.sh   # 위치 지정
# =====================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_DIR="${LIBERO_DIR:-$REPO_DIR/third_party/LIBERO}"

echo "==> Poultry-VLA setup"
echo "    repo       : $REPO_DIR"
echo "    LIBERO_DIR : $LIBERO_DIR (commit $LIBERO_COMMIT)"

# --- 1) 파이썬 의존성 ---------------------------------------------------
echo "==> [1/5] pip 의존성 설치"
pip install -r "$REPO_DIR/requirements.txt"

# --- 2) LIBERO 핀 커밋 클론 + 설치 -------------------------------------
echo "==> [2/5] LIBERO 설치"
if [ ! -d "$LIBERO_DIR/.git" ]; then
  mkdir -p "$(dirname "$LIBERO_DIR")"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
git -C "$LIBERO_DIR" fetch --quiet origin
git -C "$LIBERO_DIR" checkout --quiet "$LIBERO_COMMIT"
pip install -e "$LIBERO_DIR" --no-deps

# --- 3) 양계장 에셋/코드 주입 ------------------------------------------
echo "==> [3/5] 양계장 객체·씬·BDDL 주입"
B="$REPO_DIR/libero_chickenfarm"
L="$LIBERO_DIR/libero/libero"

cp "$B/objects/hope_objects.py"               "$L/envs/objects/hope_objects.py"
cp "$B/problems/libero_floor_manipulation.py" "$L/envs/problems/libero_floor_manipulation.py"
mkdir -p "$L/assets/scenes"
cp "$B/assets/scenes/libero_floor_poultry_style.xml" "$L/assets/scenes/"
# 객체 에셋 (상대 심링크 보존: -a)
cp -a "$B/assets/stable_hope_objects/."   "$L/assets/stable_hope_objects/"
cp -a "$B/assets/stable_scanned_objects/." "$L/assets/stable_scanned_objects/"
# BDDL
cp "$B/bddl/"*.bddl "$L/bddl_files/libero_object/"
# 씬 생성기는 <LIBERO>/scripts/ 에 있어야 기본 출력 경로가 맞음
mkdir -p "$LIBERO_DIR/scripts"
cp "$REPO_DIR/data_pipeline/generate_chicken_farm_bddl.py" "$LIBERO_DIR/scripts/"

# --- 4) robosuite 시각 버퍼 패치 (maxgeom 1000 -> 5000) ----------------
echo "==> [4/5] robosuite maxgeom 패치"
python "$REPO_DIR/patches/apply_robosuite_patch.py"

# --- 5) 안내 -----------------------------------------------------------
echo "==> [5/5] 완료"
cat <<EOF

✅ 셋업 완료.

다음 환경변수로 실행하세요 (LIBERO_DIR 는 auto-detect 되지만 명시 권장):

    export LIBERO_DIR="$LIBERO_DIR"
    export MUJOCO_GL=osmesa
    export PYTHONPATH="\$LIBERO_DIR:\$PYTHONPATH"   # editable 매핑 비는 경우 대비

빠른 데모 (단일 에피소드 mp4 렌더):

    bash run_demo.sh

데모 100개 병렬 수집:

    python data_pipeline/collect_blue_chick_demos.py --n-demos 100 \\
        --out-dir demos/blue_chick --workers 6

EOF
