"""grasp Z 정밀 진단: 학습분포 장면에서 폐루프 실행하며 타깃 병아리 근처(xy<4cm)일 때
eef_z vs chick_z vs (grasp 목표 z=chick_z+0.049) vs gripper 를 로깅. 하강깊이/그리퍼 타이밍 규명.
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

CKPT="/home/capstone/openvla_ckpts/merged-v3-s30000"
SEEDS=[9000, 9003]   # 앞서 1mm/3mm 나온 장면
MAX_STEPS=700
GRASP_Z_OFFSET=0.049
LIBERO_DIR="/home/capstone/LIBERO"; GEN=os.path.join(LIBERO_DIR,"scripts","generate_chicken_farm_bddl.py")
GEN_ARGS=["--n-chicks","7","--n-dead","3","--n-straw","50","--n-feed","15","--n-straw-visual","1000"]
BDDL_NAME="_diag_trainscene.bddl"; BDDL_DIR=os.path.join(get_libero_path("bddl_files"),"libero_object")
LANG="Pick up the blue chick and place it in the basket"

cfg=SimpleNamespace(pretrained_checkpoint=CKPT,load_in_8bit=False,load_in_4bit=False,
                    model_family="openvla",unnorm_key="blue_chick_thermal",center_crop=True)
vla=get_vla(cfg); processor=get_processor(cfg); rs=get_image_resize_size(cfg)
np.set_printoptions(precision=3,suppress=True)

def bpos(env,name):
    sim=env.env.sim
    for suf in ["","_main"]:
        try: return np.array(sim.data.body_xpos[sim.model.body_name2id(name+suf)])
        except Exception: continue
    return None

for seed in SEEDS:
    subprocess.run(["python",GEN]+GEN_ARGS+["--out",BDDL_NAME,"--no-arena","--seed",str(seed)],
                   cwd=LIBERO_DIR,capture_output=True,text=True)
    env=OffScreenRenderEnv(bddl_file_name=os.path.join(BDDL_DIR,BDDL_NAME),
                           camera_heights=256,camera_widths=256,horizon=2000)
    env.seed(seed); env.reset(); rng=np.random.RandomState(seed)
    obs=None
    for _ in range(8): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
    wctx=S.setup_wander(env,rng)
    chicks=sorted(S.find_object_body_names(env,"dead_chick_"))
    print(f"\n===== seed{seed} chicks={chicks} =====")
    print(f"{'t':>3} {'chick':>14} {'xy_d':>6} {'eef_z':>7} {'chk_z':>7} {'z갭':>7} {'grip':>5}")
    near_logged=0; min_xy=1e9; best=None
    for t in range(MAX_STEPS):
        S.step_wander(env,wctx,rng)
        img=resize_image(thermal_fx(obs["agentview_image"][::-1],colorbar=True),(rs,rs))
        a=np.array(get_action(cfg,vla,{"full_image":img,"state":None},LANG,processor=processor),dtype=np.float32)
        act=invert_gripper_action(normalize_gripper_action(a.copy(),binarize=True))
        obs,_,done,_=env.step(act.tolist())
        eef=np.array(obs["robot0_eef_pos"])
        # 이 스텝에서 eef에 가장 가까운(수평) 죽은 병아리
        dists={c:(np.linalg.norm(eef[:2]-bpos(env,c)[:2]) if bpos(env,c) is not None else 1e9) for c in chicks}
        cnear=min(dists,key=dists.get); xy=dists[cnear]; cp=bpos(env,cnear)
        if xy<min_xy: min_xy=xy; best=(t,cnear,xy,eef[2],cp[2],a[6])
        if xy<0.04 and near_logged<50:   # 어떤 병아리든 근처면 로깅
            gz=cp[2]+GRASP_Z_OFFSET
            print(f"{t:>3} {cnear:>14} {xy:>6.3f} {eef[2]:>7.3f} {cp[2]:>7.3f} {eef[2]-gz:>7.3f} {a[6]:>5.2f}")
            near_logged+=1
        if done: print("DONE"); break
    if best:
        t,c,xy,ez,cz,g=best
        print(f"[요약 seed{seed}] 최근접 {c} xy={xy:.3f}m @t{t}: eef_z={ez:.3f} chick_z={cz:.3f} "
              f"grasp목표z={cz+GRASP_Z_OFFSET:.3f} z갭={ez-(cz+GRASP_Z_OFFSET):+.3f} grip={g:.2f}", flush=True)
    env.close()
