"""짧은 폐루프 진단: 평가와 동일 파이프라인으로 40스텝 실행하며
매 스텝 예측 액션 + 실제 eef 위치 변화를 출력. 로봇이 실제로 움직이는지 확정.
"""
import sys
import numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "/home/capstone/openvla")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")
from experiments.robot.openvla_utils import get_vla, get_processor
from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action, quat2axisangle, resize_image
from thermal_fx import thermal_fx
from libero.libero import benchmark

CKPT = "/home/capstone/openvla_ckpts/merged-v3-s30000"
cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False,
                      model_family="openvla", unnorm_key="blue_chick_thermal", center_crop=True)
vla = get_vla(cfg); processor = get_processor(cfg)
resize_size = get_image_resize_size(cfg)
ts = benchmark.get_benchmark_dict()["libero_object"](); task = ts.get_task(0)
env, desc = get_libero_env(task, "openvla", resolution=256)
inits = ts.get_task_init_states(0); env.reset()
try: env.set_init_state(inits[0])
except Exception as e: print("init fb:", e)
obs=None
for _ in range(10): obs,_,_,_ = env.step(get_libero_dummy_action("openvla"))
np.set_printoptions(precision=3, suppress=True)
prev = np.array(obs["robot0_eef_pos"])
print(f"start eef={prev}")
print(f"{'t':>3} {'pred[:3]':>22} {'grip':>6} {'eef_pos':>22} {'Δeef':>7}")
for t in range(200):
    # 수집 파이프라인 정확 재현: agentview[::-1]@256 -> thermal@256 -> 224 축소
    img = obs["agentview_image"][::-1]
    img = thermal_fx(img, colorbar=True)
    img = resize_image(img, (resize_size, resize_size))
    action = get_action(cfg, vla, {"full_image": img, "state": None}, desc, processor=processor)
    a = np.array(action, dtype=np.float32)
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    obs,_,done,_ = env.step(action.tolist())
    eef = np.array(obs["robot0_eef_pos"])
    print(f"{t:>3} {str(a[:3]):>22} {a[6]:>6.2f} {str(eef):>22} {np.linalg.norm(eef-prev):>7.4f}")
    prev = eef
    if done: print("DONE(success) at",t); break
