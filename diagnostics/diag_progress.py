"""부분 성공 계측: step-30000 모델을 평가와 동일 조건으로 N 에피소드 실행하며
각 죽은 병아리의 파지(들림)/배치(바구니 안) 여부 + eef 타깃 정확도를 추적.
1마리/2마리/3마리 성공 수를 집계 → '1마리라도 성공했나' 직접 답.
"""
import sys, numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "/home/capstone/openvla")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")
from experiments.robot.openvla_utils import get_vla, get_processor
from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, resize_image
from thermal_fx import thermal_fx
from libero.libero import benchmark

CKPT = "/home/capstone/openvla_ckpts/merged-v3-s30000"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_STEPS = 700
BASKET_XY_THR = 0.07   # 바구니 contain 반경 근사
LIFT_THR = 0.03        # 시작 대비 이만큼 z 상승 시 '파지(들림)'로 간주

cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False,
                      model_family="openvla", unnorm_key="blue_chick_thermal", center_crop=True)
vla = get_vla(cfg); processor = get_processor(cfg)
rs = get_image_resize_size(cfg)
ts = benchmark.get_benchmark_dict()["libero_object"](); task = ts.get_task(0)
env, desc = get_libero_env(task, "openvla", resolution=256)
inits = ts.get_task_init_states(0)

def body_pos(name):
    sim = env.env.sim
    for suf in ["", "_main"]:
        try:
            return np.array(sim.data.body_xpos[sim.model.body_name2id(name + suf)])
        except Exception:
            continue
    return None

np.set_printoptions(precision=3, suppress=True)
CHICKS = ["dead_chick_1", "dead_chick_2", "dead_chick_3"]
agg = {"placed1": 0, "placed2": 0, "placed3": 0, "grasped_any": 0, "env_done": 0}

for ep in range(N_EP):
    env.reset()
    try: env.set_init_state(inits[ep % len(inits)])
    except Exception as e: print(f"[ep{ep}] init fb: {e}")
    obs = None
    for _ in range(10): obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    basket0 = body_pos("basket_1")
    start_z = {c: (body_pos(c)[2] if body_pos(c) is not None else np.nan) for c in CHICKS}
    max_z = dict(start_z); min_eef_dist = {c: 1e9 for c in CHICKS}
    done = False
    for t in range(MAX_STEPS):
        img = obs["agentview_image"][::-1]
        img = thermal_fx(img, colorbar=True)
        img = resize_image(img, (rs, rs))
        action = get_action(cfg, vla, {"full_image": img, "state": None}, desc, processor=processor)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())
        eef = np.array(obs["robot0_eef_pos"])
        for c in CHICKS:
            p = body_pos(c)
            if p is None: continue
            max_z[c] = max(max_z[c], p[2])
            min_eef_dist[c] = min(min_eef_dist[c], np.linalg.norm(eef[:2] - p[:2]))
        if done: break
    # 에피소드 결과 집계
    basket = body_pos("basket_1"); placed = []; grasped = []
    target = min(CHICKS, key=lambda c: min_eef_dist[c])  # eef가 가장 가까이 간 병아리
    for c in CHICKS:
        p = body_pos(c)
        if p is None: continue
        xy = np.linalg.norm(p[:2] - basket[:2])
        in_basket = xy < BASKET_XY_THR
        lifted = (max_z[c] - start_z[c]) > LIFT_THR
        if in_basket: placed.append(c)
        if lifted: grasped.append(c)
    nplaced = len(placed)
    if nplaced >= 1: agg["placed1"] += 1
    if nplaced >= 2: agg["placed2"] += 1
    if nplaced >= 3: agg["placed3"] += 1
    if grasped: agg["grasped_any"] += 1
    if done: agg["env_done"] += 1
    print(f"[ep{ep}] done={done} placed={nplaced}({placed}) grasped={grasped} "
          f"eef_target={target}(min_dist={min_eef_dist[target]:.3f}) steps={t+1}")

print("\n===== 집계 (N=%d) =====" % N_EP)
print(f"1마리 이상 배치 성공: {agg['placed1']}/{N_EP}")
print(f"2마리 이상 배치 성공: {agg['placed2']}/{N_EP}")
print(f"3마리 전부(=실제 성공): {agg['placed3']}/{N_EP}  (env_done={agg['env_done']})")
print(f"최소 1마리라도 파지(들어올림): {agg['grasped_any']}/{N_EP}")
