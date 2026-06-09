"""결정적 실험: 수집과 동일 분포의 장면(랜덤 생성 BDDL + wander 켬)에서 모델 폐루프 실행.
- 여기서 집으면 → 평가의 고정 BDDL/정지 설정이 OOD였던 것 (평가를 학습과 맞추면 해결)
- 여기서도 못 집으면 → 순수 일반화 정밀도 갭
실시간 출력: python -u 로 실행.
"""
import sys, os, subprocess
import numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "/home/capstone/openvla")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")

from experiments.robot.openvla_utils import get_vla, get_processor
from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
from experiments.robot.libero.libero_utils import resize_image
from thermal_fx import thermal_fx
# 수집 컨트롤러에서 wander/유틸 재사용
import scripted_blue_chick as S
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CKPT = "/home/capstone/openvla_ckpts/merged-v3-s30000"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_STEPS = 700
BASKET_XY_THR = 0.07; LIFT_THR = 0.03
LIBERO_DIR = "/home/capstone/LIBERO"
GEN = os.path.join(LIBERO_DIR, "scripts", "generate_chicken_farm_bddl.py")
GEN_ARGS = ["--n-chicks","7","--n-dead","3","--n-straw","50","--n-feed","15","--n-straw-visual","1000"]
BDDL_NAME = "_diag_trainscene.bddl"
BDDL_DIR = os.path.join(get_libero_path("bddl_files"), "libero_object")
LANG = "Pick up the blue chick and place it in the basket"

cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False,
                      model_family="openvla", unnorm_key="blue_chick_thermal", center_crop=True)
vla = get_vla(cfg); processor = get_processor(cfg)
rs = get_image_resize_size(cfg)
np.set_printoptions(precision=3, suppress=True)
agg = {"placed1":0,"grasped_any":0,"env_done":0}

for ep in range(N_EP):
    seed = 9000 + ep  # 학습(4000~5511) 밖의 새 seed, 같은 분포
    # 1) 수집과 동일하게 랜덤 BDDL 생성 (--no-arena: 기존 arena 재사용)
    cmd = ["python", GEN] + GEN_ARGS + ["--out", BDDL_NAME, "--no-arena", "--seed", str(seed)]
    r = subprocess.run(cmd, cwd=LIBERO_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ep{ep}] BDDL 생성 실패: {r.stderr[-300:]}"); continue
    bddl_path = os.path.join(BDDL_DIR, BDDL_NAME)

    # 2) 수집과 동일 env
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=2000)
    env.seed(seed); env.reset()
    rng = np.random.RandomState(seed)
    # settle
    obs = None
    for _ in range(8):
        obs,_,_,_ = env.step([0,0,0,0,0,0,-1])
    wctx = S.setup_wander(env, rng)  # 살아있는 병아리 배회 (학습과 동일)

    def bpos(name):
        sim=env.env.sim
        for suf in ["","_main"]:
            try: return np.array(sim.data.body_xpos[sim.model.body_name2id(name+suf)])
            except Exception: continue
        return None
    chicks = sorted(S.find_object_body_names(env,"dead_chick_"))
    basket = (S.find_object_body_names(env,"basket_") or [None])[0]
    start_z={c:(bpos(c)[2] if bpos(c) is not None else np.nan) for c in chicks}
    max_z=dict(start_z); min_d={c:1e9 for c in chicks}
    done=False
    for t in range(MAX_STEPS):
        S.step_wander(env, wctx, rng)            # 살아있는 병아리 이동 (학습과 동일)
        img = obs["agentview_image"][::-1]
        img = thermal_fx(img, colorbar=True)
        img = resize_image(img, (rs, rs))
        action = get_action(cfg, vla, {"full_image": img, "state": None}, LANG, processor=processor)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs,_,done,_ = env.step(action.tolist())
        eef=np.array(obs["robot0_eef_pos"])
        for c in chicks:
            p=bpos(c)
            if p is None: continue
            max_z[c]=max(max_z[c],p[2]); min_d[c]=min(min_d[c],np.linalg.norm(eef[:2]-p[:2]))
        if done: break
    bk=bpos(basket); placed=[]; grasped=[]
    tgt=min(chicks,key=lambda c:min_d[c])
    for c in chicks:
        p=bpos(c)
        if p is None: continue
        if np.linalg.norm(p[:2]-bk[:2])<BASKET_XY_THR: placed.append(c)
        if (max_z[c]-start_z[c])>LIFT_THR: grasped.append(c)
    if placed: agg["placed1"]+=1
    if grasped: agg["grasped_any"]+=1
    if done: agg["env_done"]+=1
    print(f"[ep{ep} seed{seed}] done={done} placed={len(placed)}({placed}) grasped={grasped} "
          f"eef_target={tgt}(min_dist={min_d[tgt]:.3f}) steps={t+1}", flush=True)
    env.close()

print(f"\n===== 집계 (수집분포 장면 + wander, N={N_EP}) =====")
print(f"1마리 이상 배치: {agg['placed1']}/{N_EP}")
print(f"1마리라도 파지: {agg['grasped_any']}/{N_EP}")
print(f"env_done(3마리): {agg['env_done']}/{N_EP}")
