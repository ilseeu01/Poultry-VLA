"""파란 병아리 픽업 task의 demo 자동 수집기.

매 seed마다:
1. BDDL 재생성 (chick 배치 랜덤)
2. scripted_blue_chick 실행 (trajectory 저장)
3. success한 episode만 채택

사용 예:
    python data_pipeline/collect_blue_chick_demos.py \
        --n-demos 50 --out-dir demos/blue_chick_v1

LIBERO 위치는 환경변수 LIBERO_DIR 로 지정 가능. 미지정 시 설치된 libero
패키지 위치에서 자동 탐지한다 (setup.sh 가 generate_chicken_farm_bddl.py 를
<LIBERO_DIR>/scripts/ 로 복사해 둠).
"""
import argparse
import os
import subprocess
import sys
import time


def _detect_libero_dir():
    """LIBERO 루트 경로 탐지: env LIBERO_DIR > 설치된 libero 패키지 위치 > 기본값."""
    if os.environ.get("LIBERO_DIR"):
        return os.environ["LIBERO_DIR"]
    try:
        import libero
        # <LIBERO_DIR>/libero/__init__.py -> 위로 두 단계
        return os.path.dirname(os.path.dirname(os.path.abspath(libero.__file__)))
    except Exception:
        return "/home/capstone/LIBERO"


LIBERO_DIR = _detect_libero_dir()
GENERATOR = os.path.join(LIBERO_DIR, "scripts", "generate_chicken_farm_bddl.py")
CONTROLLER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripted_blue_chick.py")
BDDL_BASENAME = "pick_up_the_blue_chick_and_place_it_in_the_basket.bddl"

# BDDL 생성 파라미터: 살아있는 병아리 7 + 죽은 병아리 3 = 10마리.
# 살아있는 병아리는 reach 제약 없이 floor 어디든 배치 (컨트롤러가 안 건드림).
GEN_ARGS = [
    "--n-chicks", "7",
    "--n-hens", "0",
    "--n-dead", "1",
    "--n-straw", "50",
    "--n-feed", "15",
    "--n-manure", "4",
    "--n-straw-visual", "1000",
]


def regenerate_bddl(seed, bddl_name):
    """주어진 seed로 BDDL 재생성. --no-arena: 병렬 워커가 공용 arena XML을
    동시에 덮어쓰지 않도록 (arena는 수집 전 1회 미리 생성해 둠)."""
    cmd = ["python", GENERATOR] + GEN_ARGS + [
        "--out", bddl_name, "--no-arena", "--seed", str(seed)]
    result = subprocess.run(cmd, cwd=LIBERO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    return True, result.stdout


def run_demo(seed, out_path, bddl_name, max_steps=1100, resolution=256):
    """scripted_blue_chick 실행, HDF5에 trajectory 저장.
    Returns (success, stdout_tail)."""
    cmd = [
        "python", CONTROLLER,
        "--bddl", bddl_name,
        "--max-steps", str(max_steps),
        "--resolution", str(resolution),
        "--no-save-mp4",
        "--save-trajectory", out_path,
        "--seed", str(seed),
        "--thermal",          # HDF5 이미지에 IR 열화상 후처리 적용
        "--moving-chicks",    # 살아있는 병아리 배회
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout_tail = "\n".join(result.stdout.split("\n")[-10:])
    # success 여부는 출력 마지막 "(success=True/False" 로 판단
    success = "(success=True" in result.stdout
    return success, stdout_tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=10, help="수집할 success demo 수")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="최대 시도 횟수 (0 = n-demos*3)")
    ap.add_argument("--out-dir", default="demos/blue_chick_v1")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1100)
    ap.add_argument("--resolution", type=int, default=256, help="이미지 해상도 (작을수록 빠름)")
    ap.add_argument("--worker-tag", default="",
                    help="병렬 수집 시 워커 식별자 (BDDL/로그/출력 파일명 prefix)")
    args = ap.parse_args()

    if args.max_attempts == 0:
        args.max_attempts = args.n_demos * 3

    # 워커별 고유 BDDL 파일 — 병렬 시 서로 안 덮어쓰게
    bddl_name = f"_collect_{args.worker_tag or 'single'}.bddl"

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"collection_log_{args.worker_tag or 'single'}.txt")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Target: {args.n_demos} success demos, max {args.max_attempts} attempts")
    log(f"Output dir: {args.out_dir}")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    n_success = 0
    seed = args.start_seed
    attempts = 0
    start_time = time.time()

    while n_success < args.n_demos and attempts < args.max_attempts:
        attempts += 1
        attempt_start = time.time()
        log(f"\n[attempt {attempts} | seed {seed}] regenerating BDDL...")
        ok, msg = regenerate_bddl(seed, bddl_name)
        if not ok:
            log(f"  BDDL gen failed: {msg.strip()[-200:]}")
            seed += 1
            continue

        out_path = os.path.join(
            args.out_dir, f"{args.worker_tag}demo_{n_success:03d}_seed{seed}.h5")
        log(f"  running controller → {out_path}")
        success, tail = run_demo(seed, out_path, bddl_name, max_steps=args.max_steps,
                                  resolution=args.resolution)
        dur = time.time() - attempt_start

        if success:
            n_success += 1
            log(f"  ✓ SUCCESS [{n_success}/{args.n_demos}] in {dur:.1f}s")
        else:
            log(f"  ✗ fail in {dur:.1f}s — last output:\n{tail}")
            # 실패한 HDF5는 디스크 공간 차지가 크므로(~120MB) 즉시 삭제.
            if os.path.exists(out_path):
                os.remove(out_path)

        seed += 1

    total = time.time() - start_time
    log(f"\n=== Done ===")
    log(f"Collected {n_success}/{args.n_demos} demos in {attempts} attempts ({total/60:.1f} min)")
    log(f"Success rate: {n_success/attempts*100:.1f}%")
    log_f.close()


if __name__ == "__main__":
    main()
