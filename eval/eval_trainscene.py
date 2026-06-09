"""학습분포 일치 평가: 수집과 동일하게 랜덤 생성한 1마리 장면 + wander 켬 + 수정된 obs 파이프라인
(agentview[::-1]@256 -> thermal@256 -> 224)으로 모델을 폐루프 실행, env done(=1마리 goal) 성공률 측정.
사용: python -u eval_trainscene.py <CKPT> [N_EP] [MAX_STEPS]
"""
import sys, os, subprocess
import numpy as np
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

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/capstone/openvla_ckpts/merged-v3-s30000"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
START_SEED = 20000  # 학습(8000~13000+) 밖
LIBERO_DIR = "/home/capstone/LIBERO"; GEN = os.path.join(LIBERO_DIR, "scripts", "generate_chicken_farm_bddl.py")
GEN_ARGS = ["--n-chicks","7","--n-hens","0","--n-dead","1","--n-straw","50","--n-feed","15","--n-manure","4","--n-straw-visual","1000"]
BDDL_NAME = "_eval_trainscene.bddl"; BDDL_DIR = os.path.join(get_libero_path("bddl_files"), "libero_object")
LANG = "Pick up the blue chick and place it in the basket"

LIFT_THR = 0.03  # 시작 대비 z 이만큼 상승 시 '파지(들어올림)'로 간주
cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False,
                      model_family="openvla", unnorm_key="blue_chick_thermal", center_crop=True)
vla = get_vla(cfg); processor = get_processor(cfg); rs = get_image_resize_size(cfg)

def bpos(env, name):
    sim = env.env.sim
    for suf in ["", "_main"]:
        try: return np.array(sim.data.body_xpos[sim.model.body_name2id(name + suf)])
        except Exception: continue
    return None

succ = 0; grasped_any = 0
for ep in range(N_EP):
    seed = START_SEED + ep
    subprocess.run(["python", GEN] + GEN_ARGS + ["--out", BDDL_NAME, "--no-arena", "--seed", str(seed)],
                   cwd=LIBERO_DIR, capture_output=True, text=True)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(BDDL_DIR, BDDL_NAME),
                             camera_heights=256, camera_widths=256, horizon=MAX_STEPS + 100)
    env.seed(seed); env.reset(); rng = np.random.RandomState(seed)
    obs = None
    for _ in range(10): obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
    wctx = S.setup_wander(env, rng)
    chick = "dead_chick_1"
    start_z = bpos(env, chick)[2]; max_z = start_z; min_xy = 1e9
    done = False
    for t in range(MAX_STEPS):
        S.step_wander(env, wctx, rng)
        img = resize_image(thermal_fx(obs["agentview_image"][::-1], colorbar=True), (rs, rs))
        a = get_action(cfg, vla, {"full_image": img, "state": None}, LANG, processor=processor)
        a = invert_gripper_action(normalize_gripper_action(a, binarize=True))
        obs, _, done, _ = env.step(a.tolist())
        cp = bpos(env, chick); eef = np.array(obs["robot0_eef_pos"])
        if cp is not None:
            max_z = max(max_z, cp[2]); min_xy = min(min_xy, np.linalg.norm(eef[:2] - cp[:2]))
        if done: break
    lifted = (max_z - start_z) > LIFT_THR
    succ += int(done); grasped_any += int(lifted)
    print(f"[ep{ep} seed{seed}] success={done} lifted={lifted} min_xy={min_xy:.3f} "
          f"(steps={t+1})  누적 성공 {succ}/{ep+1} 파지 {grasped_any}/{ep+1}", flush=True)
    env.close()
print(f"\n===== 학습분포 평가 (N={N_EP}) =====")
print(f"성공(바구니): {succ}/{N_EP} = {100*succ/N_EP:.1f}%")
print(f"파지(들어올림): {grasped_any}/{N_EP} = {100*grasped_any/N_EP:.1f}%  ← 옛 모델 0%")
