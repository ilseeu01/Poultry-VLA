"""실제 평가 관측 재현: env reset → get_libero_image(평가방향 [::-1,::-1]) → thermal_fx
→ 모델 예측. 방향수정([::-1]) 버전도 같이 예측. 이미지도 저장해 학습본과 비교.
"""
import sys, os, glob
import numpy as np, torch
from types import SimpleNamespace
from PIL import Image
import imageio.v2 as iio

sys.path.insert(0, "/home/capstone/openvla")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.libero.libero_utils import get_libero_env, resize_image, get_libero_dummy_action
from libero.libero import benchmark
from thermal_fx import thermal_fx

CKPT = "/home/capstone/openvla_ckpts/merged-v3-s30000"
UNNORM = "blue_chick_thermal"
cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False, model_family="openvla")
vla = get_vla(cfg); processor = get_processor(cfg)

ts = benchmark.get_benchmark_dict()["libero_object"]()
task = ts.get_task(0)
env, desc = get_libero_env(task, "openvla", resolution=256)
inits = ts.get_task_init_states(0)
env.reset();
try:
    env.set_init_state(inits[0])
except Exception as e:
    print("set_init_state fallback:", e)
# 안정화 dummy step
for _ in range(10):
    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

raw = obs["agentview_image"]
np.set_printoptions(precision=3, suppress=True)

# 평가방향: [::-1,::-1] → resize → thermal_fx(colorbar)
img_eval = resize_image(raw[::-1, ::-1], (256, 256))
img_eval_t = thermal_fx(img_eval, colorbar=True)
# 방향수정(수집과 동일): [::-1]
img_fix = resize_image(raw[::-1], (256, 256))
img_fix_t = thermal_fx(img_fix, colorbar=True)

p_eval = np.asarray(get_vla_action(vla, processor, CKPT, {"full_image": img_eval_t}, desc, UNNORM, True), np.float32)
p_fix  = np.asarray(get_vla_action(vla, processor, CKPT, {"full_image": img_fix_t}, desc, UNNORM, True), np.float32)
print("\n[실제 평가 관측 예측]")
print("  평가방향([::-1,::-1]) PRED:", p_eval)
print("  방향수정([::-1])      PRED:", p_fix)

# 이미지 저장 (학습본과 비교용)
iio.imwrite("/tmp/obs_eval_thermal.png", img_eval_t)
iio.imwrite("/tmp/obs_fix_thermal.png", img_fix_t)
# 학습 데모 t=0 thermal
import h5py
with h5py.File(sorted(glob.glob("/home/capstone/openvla/demos/blue_chick_thermal_v2/*demo*.h5"))[0]) as f:
    iio.imwrite("/tmp/obs_train_thermal.png", f["agentview_rgb"][0])
print("\nsaved: /tmp/obs_eval_thermal.png (평가방향), /tmp/obs_fix_thermal.png (방향수정), /tmp/obs_train_thermal.png (학습)")
