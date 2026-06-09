"""데모 품질 점검: (1) 통계표 CSV, (2) 전체 montage 영상.
사용: python review_demos.py <demo_dir> <out_dir>
"""
import sys, os, glob, csv
import numpy as np
import h5py
import imageio.v2 as iio
from PIL import Image

DEMO_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/capstone/openvla/demos/blue_chick_thermal_v2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/capstone/openvla/demo_review"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(DEMO_DIR, "*.h5")))
print(f"{len(files)} demos in {DEMO_DIR}")

TILE = 96          # montage 타일 한 변 픽셀
T = 90             # montage 공통 타임라인 프레임 수 (각 데모를 이 길이로 리샘플)
rows = []
montage_frames = [None] * len(files)  # 각 데모의 (T, TILE, TILE, 3)

for i, fp in enumerate(files):
    with h5py.File(fp, "r") as f:
        a = f.attrs
        act = f["actions"][()]
        av = f["agentview_rgb"]
        n = av.shape[0]
        pos = act[:, :3]
        mag = np.linalg.norm(pos, axis=1)
        noop = float((mag < 0.01).mean())
        sat = float((np.abs(pos) >= 0.499).any(1).mean())
        grip = act[:, 6]
        grip_toggles = int((np.diff(np.sign(grip)) != 0).sum())
        rows.append({
            "idx": i,
            "file": os.path.basename(fp),
            "success": bool(a.get("success", False)),
            "thermal": bool(a.get("thermal", False)),
            "moving_chicks": bool(a.get("moving_chicks", False)),
            "seed": int(a.get("seed", -1)),
            "num_steps": int(n),
            "noop_frac": round(noop, 3),
            "sat_frac": round(sat, 3),
            "grip_toggles": grip_toggles,
        })
        # montage: T개 인덱스로 리샘플 후 다운스케일
        idxs = np.linspace(0, n - 1, T).astype(int)
        tiles = np.empty((T, TILE, TILE, 3), dtype=np.uint8)
        for j, t in enumerate(idxs):
            im = Image.fromarray(av[t]).resize((TILE, TILE), Image.BILINEAR)
            tiles[j] = np.asarray(im)
        montage_frames[i] = tiles
    if (i + 1) % 20 == 0:
        print(f"  processed {i+1}/{len(files)}")

# --- CSV 저장 ---
csv_path = os.path.join(OUT, "demo_stats.csv")
with open(csv_path, "w", newline="") as cf:
    w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("saved", csv_path)

# --- 요약 출력 ---
ns = np.array([r["num_steps"] for r in rows])
succ = sum(r["success"] for r in rows)
print(f"\n=== 요약 ===")
print(f"성공 플래그 True: {succ}/{len(rows)}")
print(f"num_steps: min={ns.min()} median={int(np.median(ns))} max={ns.max()} mean={ns.mean():.0f}")
print(f"noop_frac: mean={np.mean([r['noop_frac'] for r in rows]):.3f}")
print(f"sat_frac : mean={np.mean([r['sat_frac'] for r in rows]):.3f}")
print("이상치(짧은 데모 하위 5):", sorted([(r['num_steps'], r['file']) for r in rows])[:5])
print("이상치(긴 데모 상위 5):", sorted([(r['num_steps'], r['file']) for r in rows])[-5:])

# --- montage 영상 (그리드) ---
N = len(files)
G = int(np.ceil(np.sqrt(N)))   # 정사각 그리드 한 변
W = G * TILE
grid_video = np.zeros((T, W, W, 3), dtype=np.uint8)
for i in range(N):
    r, c = divmod(i, G)
    grid_video[:, r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE] = montage_frames[i]
mont_path = os.path.join(OUT, "montage_all.mp4")
wr = iio.get_writer(mont_path, fps=12, codec="libx264", quality=7)
for t in range(T):
    wr.append_data(grid_video[t])
wr.close()
print("saved montage:", mont_path, f"({G}x{G} grid, {N} demos)")
