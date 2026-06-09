"""해상도/순서 불일치 테스트:
학습: thermal@256(수집) → 로더가 224 축소
평가: [::-1] → resize@224 → thermal@224
같은 env obs로 (A)평가정확재현(224) (B)256-thermal 후 예측 비교.
"""
import sys, glob
import numpy as np, torch, h5py
from types import SimpleNamespace
import imageio.v2 as iio
sys.path.insert(0, "/home/capstone/openvla")
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.libero.libero_utils import get_libero_env, resize_image, get_libero_dummy_action
from libero.libero import benchmark
from thermal_fx import thermal_fx

CKPT = "/home/capstone/openvla_ckpts/merged-v3-s30000"; UNNORM = "blue_chick_thermal"
cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False, model_family="openvla")
vla = get_vla(cfg); processor = get_processor(cfg)
ts = benchmark.get_benchmark_dict()["libero_object"](); task = ts.get_task(0)
env, desc = get_libero_env(task, "openvla", resolution=256)
inits = ts.get_task_init_states(0); env.reset()
try: env.set_init_state(inits[0])
except Exception as e: print("init fb:", e)
for _ in range(10): obs,_,_,_ = env.step(get_libero_dummy_action("openvla"))
raw = obs["agentview_image"]
np.set_printoptions(precision=3, suppress=True)

# (A) 평가 정확 재현: [::-1] -> resize224 -> thermal@224
imgA = thermal_fx(resize_image(raw[::-1], (224,224)), colorbar=True)
# (B) 학습식: thermal@256 -> (processor가 224로) ; 여기선 [::-1] thermal@256 그대로 넣음(get_vla_action 내부 처리)
imgB = thermal_fx(resize_image(raw[::-1], (256,256)), colorbar=True)
# (C) 학습식 + 256 thermal 후 224로 다운(순서: thermal먼저)
imgC = resize_image(thermal_fx(resize_image(raw[::-1], (256,256)), colorbar=True), (224,224))

for name, im in [("A_eval재현(224,thermal@224)", imgA), ("B_thermal@256", imgB), ("C_thermal@256→224", imgC)]:
    p = np.asarray(get_vla_action(vla, processor, CKPT, {"full_image": im}, desc, UNNORM, True), np.float32)
    print(f"{name:32s} PRED: {p}")
iio.imwrite("/tmp/res_A_224.png", imgA); iio.imwrite("/tmp/res_C_256to224.png", imgC)
print("saved /tmp/res_A_224.png, /tmp/res_C_256to224.png")
