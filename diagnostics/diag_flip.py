"""방향 불일치 확정 테스트: 학습 이미지 vs 좌우반전(=평가 방향) 예측 비교.
수집은 raw[::-1], 평가는 raw[::-1,::-1] = 학습[:, ::-1] (좌우반전).
"""
import sys, glob
import numpy as np, h5py, torch
from types import SimpleNamespace
sys.path.insert(0, "/home/capstone/openvla")
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/capstone/openvla_ckpts/merged-v3-s30000"
UNNORM = "blue_chick_thermal"
cfg = SimpleNamespace(pretrained_checkpoint=CKPT, load_in_8bit=False, load_in_4bit=False, model_family="openvla")
vla = get_vla(cfg); processor = get_processor(cfg)

demo = sorted(glob.glob("/home/capstone/openvla/demos/blue_chick_thermal_v2/*demo*.h5"))[0]
with h5py.File(demo, "r") as f:
    ag = f["agentview_rgb"][()]; acts = f["actions"][()]; lang = str(f.attrs["language"])
mags = np.linalg.norm(acts[:, :3], axis=1)
idxs = [int(i) for i in np.argsort(-mags)[:5]]
np.set_printoptions(precision=3, suppress=True)
print(f"\n{'step':>5} | {'GT pos':>22} | {'PRED(train방향)':>22} | {'PRED(좌우반전=평가)':>22}")
for t in sorted(set(idxs)):
    img = ag[t]
    p_norm = np.asarray(get_vla_action(vla, processor, CKPT, {"full_image": img}, lang, UNNORM, True), np.float32)
    p_flip = np.asarray(get_vla_action(vla, processor, CKPT, {"full_image": img[:, ::-1]}, lang, UNNORM, True), np.float32)
    print(f"{t:>5} | {str(acts[t][:3]):>22} | {str(p_norm[:3]):>22} | {str(p_flip[:3]):>22}")
print("\n좌우반전 예측이 ~0/엉뚱해지면 → 방향 불일치가 평가 정지의 원인 확정.")
