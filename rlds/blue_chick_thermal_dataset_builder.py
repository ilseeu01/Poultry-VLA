"""TFDS builder for blue_chick_thermal: 100 IR thermal demos (HDF5 → RLDS).

LIBERO 호환 스키마:
  observation.image, observation.wrist_image (256x256x3 uint8 PNG)
  observation.state (8,) = [ee_pos(3), axisangle(3), gripper_qpos(2)]
  action (7,) float32 (eef delta + gripper)
  language_instruction
"""
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# v3.0.0: 200 demos (blue_chick_thermal_v3, 부드러운 게인 XY=4/Z=6, goal 1마리) + no-op 필터링
DEMOS_DIRS = [
    Path("/home/capstone/openvla/demos/blue_chick_thermal_v3"),
]
NOOP_POS_THRESHOLD = 1e-3  # 위치/회전 6축 norm 이보다 작고 그리퍼 변화 없으면 no-op


def _is_noop(action, prev_gripper):
    """움직임(앞 6축)이 무시할 수준이고 그리퍼 상태도 안 바뀌면 no-op."""
    no_motion = float(np.linalg.norm(action[:6])) < NOOP_POS_THRESHOLD
    gripper_same = (prev_gripper is not None) and (action[6] == prev_gripper)
    return no_motion and gripper_same


def _quat2axisangle(quat):
    """quat (x,y,z,w) → axis-angle (3,). robosuite 구현 복사."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if den < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(q[3])
    return (q[:3] / den * angle).astype(np.float32)


class BlueChickThermal(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial 100 thermal demos (moving chicks, IR + colorbar).",
        "2.0.0": "200 demos (v2 + thermal_100), no-op transitions filtered.",
        "3.0.0": "200 demos blue_chick_thermal_v3 (smooth gain XY4/Z6, 1-chick goal), no-op filtered.",
    }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Blue chick (cold/dead) pickup task with IR thermal observations.",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8,
                                                     encoding_format="png"),
                        "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8,
                                                          encoding_format="png"),
                        "state": tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "reward": tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last": tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "seed": tfds.features.Scalar(dtype=tf.int64),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        files = []
        for d in DEMOS_DIRS:
            files.extend(str(p) for p in d.glob("*demo*.h5"))
        files = sorted(files)
        return {"train": self._generate_examples(files)}

    def _generate_examples(self, files):
        for fp in files:
            with h5py.File(fp, "r") as f:
                T = int(f.attrs["num_steps"])
                lang = str(f.attrs["language"])
                seed = int(f.attrs["seed"])
                ag = f["agentview_rgb"][:T]
                wrist = f["eye_in_hand_rgb"][:T]
                ee_pos = f["ee_pos"][:T]
                ee_quat = f["ee_quat"][:T]
                gripper = f["gripper_qpos"][:T]
                actions = f["actions"][:T]
            axisang = np.stack([_quat2axisangle(q) for q in ee_quat]).astype(np.float32)
            state = np.concatenate([ee_pos, axisang, gripper], axis=1).astype(np.float32)

            # no-op 필터링: 움직임 없고 그리퍼 안 바뀐 step 제거 (마지막 step은 항상 유지)
            keep = []
            prev_gripper = None
            for t in range(T):
                if t == T - 1 or not _is_noop(actions[t], prev_gripper):
                    keep.append(t)
                    prev_gripper = actions[t][6]
            if len(keep) < 2:
                continue  # 거의 다 no-op인 비정상 에피소드는 스킵

            n = len(keep)
            steps = []
            for i, t in enumerate(keep):
                steps.append({
                    "observation": {
                        "image": ag[t],
                        "wrist_image": wrist[t],
                        "state": state[t],
                    },
                    "action": actions[t].astype(np.float32),
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == n - 1 else 0.0),
                    "is_first": bool(i == 0),
                    "is_last": bool(i == n - 1),
                    "is_terminal": bool(i == n - 1),
                    "language_instruction": lang,
                })
            yield fp, {
                "steps": steps,
                "episode_metadata": {"file_path": fp, "seed": seed},
            }
