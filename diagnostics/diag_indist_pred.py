"""In-distribution 예측 진단: 학습 데모(thermal h5) 이미지를 그대로 모델에 넣어
예측 액션 vs 정답 액션 비교. 모델 붕괴 여부 판정.
"""
import sys, glob, json
import numpy as np
import h5py
import torch
from types import SimpleNamespace

sys.path.insert(0, "/home/capstone/openvla")
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/capstone/openvla_ckpts/merged-v3-s30000"
UNNORM = "blue_chick_thermal"

cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False, model_family="openvla")
vla = get_vla(cfg)
processor = get_processor(cfg)

# 학습 데모에서 in-distribution thermal 이미지 + 정답 액션 추출
demo = sorted(glob.glob("/home/capstone/openvla/demos/blue_chick_thermal_v2/*demo*.h5"))[0]
with h5py.File(demo, "r") as f:
    ag = f["agentview_rgb"][()]
    acts = f["actions"][()]
    lang = str(f.attrs["language"])
T = len(acts)
# 움직임이 큰 step들(정답이 0이 아닌)에서 비교
mags = np.linalg.norm(acts[:, :3], axis=1)
idxs = [int(i) for i in np.argsort(-mags)[:6]]  # 가장 크게 움직이는 6 step
idxs += [T // 4, T // 2]
np.set_printoptions(precision=3, suppress=True)
print(f"\n[demo] {demo.split('/')[-1]}  T={T}  lang='{lang}'")
print(f"{'step':>5} | {'GT action':>40} | {'PRED action':>40} | pos_err")
for t in sorted(set(idxs)):
    obs = {"full_image": ag[t]}
    pred = get_vla_action(vla, processor, CKPT, obs, lang, UNNORM, center_crop=True)
    pred = np.asarray(pred, dtype=np.float32)
    gt = acts[t].astype(np.float32)
    perr = float(np.linalg.norm(pred[:3] - gt[:3]))
    print(f"{t:>5} | {str(gt):>40} | {str(pred):>40} | {perr:.3f}")

# 예측이 전부 ~0이면 모델 붕괴 / GT 따라가면 정상
print("\n해석: PRED pos가 GT pos를 어느정도 따라가면 모델 정상(=평가 분포시프트 문제),")
print("      PRED가 step 무관하게 ~0이면 모델 자체가 no-op로 붕괴.")
