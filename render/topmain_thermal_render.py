#!/usr/bin/env python
"""탑뷰(birdview) + 메인뷰(agentview) 동시 렌더 — r4 solo 정책을 2뷰로 캡처,
각 뷰에 thermal_fx 적용 후 가로 타일 mp4. 결과 시연용(탑뷰+메인뷰 IR 영상).

사용 (oft env, GPU):
  BLUE_CHICK_THERMAL=1 MUJOCO_GL=osmesa python topmain_thermal_render.py \
      --ckpt <merged_r4> --seeds 12001,12004,12009 \
      --out-dir /home/capstone/openvla/rollouts/2026_06_17_topmain
옵션:
  --top-clean : 탑뷰는 thermal 미적용(원본 RGB)으로 — 씬/바구니 가독성↑
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("BLUE_CHICK_THERMAL", "1")

import argparse
import sys

import numpy as np
import imageio

sys.path.insert(0, "/home/capstone/LIBERO")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import scripted_blue_chick as C
from thermal_fx import thermal_fx
from hybrid_eval import VLANavPolicy, gen_bddl, _max_dead_chick_z, SETTLE_STEPS

VIEWS = ["birdview", "agentview"]                       # 화면 타일용(탑뷰+메인뷰)
# ⚠️ agentview를 첫 번째로 둬야 함 — osmesa에서 첫 카메라가 agentview가 아니면
# 모델 입력 agentview 렌더가 깨져 정책 실패(birdview-first일 때 0/4 grasp 확인).
RENDER_CAMS = ["agentview", "robot0_eye_in_hand", "birdview"]  # env 렌더(모델은 agentview+wrist 입력)
VIEW_LABEL = {"birdview": "TOP VIEW", "agentview": "MAIN VIEW"}
LANG = "Pick up the blue chick and place it in the basket"


def _label(img, text):
    """좌상단에 뷰 이름 (PIL)."""
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 9 * len(text) + 8, 16], fill=(0, 0, 0))
        d.text((4, 3), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return img


def tile_views(obs, top_clean=False):
    """탑뷰+메인뷰 → 가로 타일 1프레임."""
    tiles = []
    for v in VIEWS:
        key = v + "_image"
        if key not in obs:
            continue
        raw = np.ascontiguousarray(obs[key][::-1])
        if v == "birdview" and top_clean:
            img = raw                      # 탑뷰 원본 RGB
        else:
            img = thermal_fx(raw, colorbar=True)
        tiles.append(_label(img.astype(np.uint8), VIEW_LABEL[v]))
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    return np.concatenate(tiles, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unnorm-key", default="blue_chick_dagger")
    ap.add_argument("--seeds", default="12001,12004,12009")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--top-clean", action="store_true",
                    help="탑뷰는 thermal 미적용(원본 RGB)")
    ap.add_argument("--post-done-steps", type=int, default=20,
                    help="env done 이후 추가 롤아웃(안착 프레임 확보; 모델 정지)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]

    policy = VLANavPolicy(args.ckpt, args.unnorm_key)
    print(f"[tm] model loaded; seeds={seeds} top_clean={args.top_clean}")

    for seed in seeds:
        name = f"_tm_{seed}.bddl"
        if not gen_bddl(seed, name, 1):
            print(f"[tm] seed {seed} bddl gen fail")
            continue
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_object", name)
        env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_names=RENDER_CAMS,
                camera_heights=args.res, camera_widths=args.res,
                horizon=args.max_steps + 200)
            env.seed(seed)
            obs = env.reset()
            policy.reset()
            rng = np.random.RandomState(seed)
            wanderers = C.setup_wander(env, rng)
            for _ in range(SETTLE_STEPS):
                C.step_wander(env, wanderers, rng)
                obs, done, term = C.safe_step(env, [0, 0, 0, 0, 0, 0, -1])
            dead = sorted(C.find_object_body_names(env, "dead_chick_"))
            z0 = {b: C.get_body_pos(env, b)[2] for b in dead}
            basket = C.get_body_pos(env, C.find_object_body_names(env, "basket_")[0])
            frames = [tile_views(obs, args.top_clean)]
            done = False
            done_at = None
            for t in range(args.max_steps):
                act = policy.act(obs, env, None)
                C.step_wander(env, wanderers, rng)
                obs, done, term = C.safe_step(env, list(act))
                frames.append(tile_views(obs, args.top_clean))
                if term or done:
                    done_at = t
                    break
            for _ in range(args.post_done_steps):
                C.step_wander(env, wanderers, rng)
                obs, d2, term = C.safe_step(env, [0, 0, 0, 0, 0, 0, -1])
                frames.append(tile_views(obs, args.top_clean))
                if term:
                    break
            lift = max((C.get_body_pos(env, b)[2] - z0[b]) for b in dead)
            pend = C.get_body_pos(env, dead[0])
            dend = float(np.hypot(pend[0] - basket[0], pend[1] - basket[1]))
            tag = "success" if done else f"lift{lift*100:.0f}cm"
            out = os.path.join(args.out_dir, f"topmain_seed{seed}_{tag}.mp4")
            w = imageio.get_writer(out, fps=30)
            for f in frames:
                w.append_data(f)
            w.close()
            print(f"[tm] seed {seed}: {len(frames)} frames done={done}@{done_at} "
                  f"lift={lift*100:.1f}cm d_basket={dend:.3f} → {out}")
        except Exception as e:
            print(f"[tm] seed {seed} SKIP: {type(e).__name__}: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    print("[tm] done")


if __name__ == "__main__":
    main()
