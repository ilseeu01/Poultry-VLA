"""그리퍼 추적: v3b 모델을 1마리 학습분포 장면에서 돌리며, 병아리 근처(xy<4cm)일 때
모델의 raw 그리퍼 예측 a[6](0=닫힘/1=열림 컨벤션)을 로깅. 닫기를 예측하는지 확인.
+ 가장 근접한 순간의 thermal 프레임 저장 (그리퍼와 죽은병아리가 시각적으로 합쳐지는지 확인).
"""
import sys, os, subprocess
import numpy as np
import imageio.v2 as iio
from types import SimpleNamespace
sys.path.insert(0, "/home/capstone/openvla")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")
from experiments.robot.openvla_utils import get_vla, get_processor
from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
from experiments.robot.libero.libero_utils import resize_image
from thermal_fx import thermal_fx
import scripted_blue_chick as S
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/capstone/openvla_ckpts/merged-v3b-s15000"
SEEDS = [20000, 20006]   # 앞서 min_xy 0.003~0.022 나온 장면
LIBERO_DIR = "/home/capstone/LIBERO"; GEN = os.path.join(LIBERO_DIR, "scripts", "generate_chicken_farm_bddl.py")
GEN_ARGS = ["--n-chicks","7","--n-hens","0","--n-dead","1","--n-straw","50","--n-feed","15","--n-manure","4","--n-straw-visual","1000"]
BDDL_NAME = "_gtrace.bddl"; BDDL_DIR = os.path.join(get_libero_path("bddl_files"), "libero_object")
LANG = "Pick up the blue chick and place it in the basket"

cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False,
                      model_family="openvla", unnorm_key="blue_chick_thermal", center_crop=True)
vla = get_vla(cfg); processor = get_processor(cfg); rs = get_image_resize_size(cfg)

def bpos(env, name):
    sim = env.env.sim
    for suf in ["", "_main"]:
        try: return np.array(sim.data.body_xpos[sim.model.body_name2id(name + suf)])
        except Exception: continue
    return None

for seed in SEEDS:
    subprocess.run(["python", GEN] + GEN_ARGS + ["--out", BDDL_NAME, "--no-arena", "--seed", str(seed)],
                   cwd=LIBERO_DIR, capture_output=True, text=True)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(BDDL_DIR, BDDL_NAME),
                             camera_heights=256, camera_widths=256, horizon=400)
    env.seed(seed); env.reset(); rng = np.random.RandomState(seed)
    obs = None
    for _ in range(10): obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
    wctx = S.setup_wander(env, rng)
    print(f"\n===== seed{seed} =====")
    print(f"{'t':>3} {'xy_d':>6} {'eef_z':>7} {'chk_z':>7} {'grip_raw(0=닫힘)':>16}")
    min_xy = 1e9; best_frame = None; grip_min = 1.0; near_logged = 0
    for t in range(400):
        S.step_wander(env, wctx, rng)
        thermal_img = thermal_fx(obs["agentview_image"][::-1], colorbar=True)
        img = resize_image(thermal_img, (rs, rs))
        a = np.array(get_action(cfg, vla, {"full_image": img, "state": None}, LANG, processor=processor), dtype=np.float32)
        act = invert_gripper_action(normalize_gripper_action(a.copy(), binarize=True))
        obs, _, done, _ = env.step(act.tolist())
        cp = bpos(env, "dead_chick_1"); eef = np.array(obs["robot0_eef_pos"])
        xy = np.linalg.norm(eef[:2] - cp[:2]) if cp is not None else 9
        if xy < 0.04:
            grip_min = min(grip_min, a[6])
            if near_logged < 40:
                print(f"{t:>3} {xy:>6.3f} {eef[2]:>7.3f} {cp[2]:>7.3f} {a[6]:>16.2f}")
                near_logged += 1
        if xy < min_xy:
            min_xy = xy; best_frame = thermal_img.copy()
        if done: break
    if best_frame is not None:
        iio.imwrite(f"/tmp/grasp_moment_seed{seed}.png", best_frame)
    print(f"[요약 seed{seed}] min_xy={min_xy:.3f}, 병아리근처 최소 grip_raw={grip_min:.2f} "
          f"({'닫기 예측함' if grip_min < 0.5 else '계속 열림(닫기 안함)'}), 프레임 저장", flush=True)
    env.close()
