#!/usr/bin/env python
"""DAgger on-policy 수집 — 모델(VLA)이 주로 행동, 전문가가 매 step 교정 라벨.

진단된 실패(폐루프 fine-control: 병아리 위 hover/진동, 하강·닫기 commit 불능)를
정면 공략: 모델이 실제로 방문하는 상태들(특히 stuck 상태)에서 전문가의 정답
action을 라벨링 → 원본 데모와 믹스해 재학습(exposure bias 제거).

- 전문가 = ShadowExpert: 원본 scripted_blue_chick 로직을 '현재 상태에서 재계획'
  하는 reactive 정책 (모델이 어디로 가 있든 그 상태의 정답 산출, 닫기 히스테리시스).
- 제어 = β-혼합: SEGMENT_LEN step 블록마다 P(전문가)=β로 실행 주체 선택.
  라벨은 항상 전문가 action. 전문가 블록이 성공 파지/운반 구간도 커버.
- 저장 = 기존 RLDS 빌더 호환 HDF5 (agentview thermal+colorbar, wrist RGB,
  ee_pos/quat/gripper_qpos/joint_pos, actions=전문가 라벨) + executed_by/executed_actions.

사용 (oft env, GPU):
  BLUE_CHICK_THERMAL=1 MUJOCO_GL=osmesa python dagger_collect.py \
      --ckpt <merged_2a30000> --seeds 11000-11049 --n-good 40 \
      --out-dir /home/capstone/openvla/demos/blue_chick_dagger_r1
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_eval import (VLANavPolicy, gen_bddl, _max_dead_chick_z,
                         BASKET_PLACE_RADIUS, SETTLE_STEPS)
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import scripted_blue_chick as C

# 원본 데모와 동일 문자열 (RLDS language_instruction 일치 필수)
LANG = "Pick up the blue chick and place it in the basket"
SEGMENT_LEN = 40       # 제어 주체 블록 길이 (OFT chunk=8의 5배)
GRASP_HOLD = 15        # 제자리 닫기 유지 step (이후 lift-on-faith)
CLOSE_RETRY = 60       # 닫기 시작 후 이만큼 지나도 안 들리면 재접근
HOLD_XY = 0.06         # holding 판정: chick 들림 + eef 근접


def _xy(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class ShadowExpert:
    """현재 sim 상태에서 원본 컨트롤러의 정답 action을 재계획하는 reactive 전문가."""

    def __init__(self, env, basket_pos, chick_z0):
        self.env = env
        self.basket = basket_pos
        self.chick_z0 = chick_z0
        self.closing = False
        self.close_t = 0
        self.descend_t = 0
        self.placed = set()

    def _remaining(self):
        out = []
        for b in sorted(self.chick_z0):
            if b in self.placed:
                continue
            p = C.get_body_pos(self.env, b)
            if p is None:
                continue
            if _xy(p, self.basket) < BASKET_PLACE_RADIUS and p[2] < 0.12:
                # 바구니 반경 + 낮은 z = 안에 '안착'한 것만 placed
                # (z 가드 없으면 운반 중 바구니 위 통과를 placed로 오판 → 공중 투하)
                self.placed.add(b)
                continue
            if not C.in_workspace(p[:2]):
                continue
            out.append((b, p))
        return out

    def action(self, obs):
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        rem = self._remaining()
        if not rem:
            return [0, 0, C.ACTION_CLIP, 0, 0, 0, -1]  # 완료 — 위로 후퇴
        tgt_body, tp = min(rem, key=lambda bp: _xy(bp[1], eef))
        holding = (tp[2] - self.chick_z0[tgt_body] > 0.04) and _xy(tp, eef) < HOLD_XY
        # 한번 holding이면 운반 phase로 래치 (release까지 전문가 강제용; placed되면 풀림)
        self._was_holding = holding or (getattr(self, "_was_holding", False)
                                        and tgt_body not in self.placed)
        if os.environ.get("DAGGER_DEBUG"):
            self._dbg = getattr(self, "_dbg", 0) + 1
            if self._dbg % 25 == 0 or holding != getattr(self, "_ph", None):
                print(f"  [exp] t={self._dbg} tgt={tgt_body} hold={holding} "
                      f"closing={self.closing}/{self.close_t} eef={eef.round(3)} "
                      f"chick=({tp[0]:.3f},{tp[1]:.3f},{tp[2]:.3f}) "
                      f"d_basket={_xy(eef, self.basket):.3f} placed={sorted(self.placed)}")
            self._ph = holding

        if holding:
            self.closing = False
            self.close_t = 0
            self.descend_t = 0
            bx, by = self.basket[0], self.basket[1]
            if _xy(eef, self.basket) > 0.03:
                if eef[2] < C.Z_TRAVEL - 0.05:
                    return C.go_to([eef[0], eef[1], C.Z_TRAVEL], eef, gripper=+1)  # LIFT
                return C.go_to([bx, by, C.Z_PRE_DROP], eef, gripper=+1)            # MOVE
            if eef[2] > C.Z_DROP + 0.025:
                return C.go_to([bx, by, C.Z_DROP], eef, gripper=+1)  # DESCEND_OVER
            return [0, 0, 0, 0, 0, 0, -1]                            # RELEASE

        # not holding
        if self.closing:
            # 닫기 → lift-on-faith: 제자리 닫기 중엔 chick이 안 뜨므로 holding으로
            # 성공 판정 불가 — 일단 들어올려 보고, 떴으면 위 holding 분기가 받음.
            self.close_t += 1
            if self.close_t > CLOSE_RETRY:
                self.closing = False                   # 헛파지 — 재접근
            elif self.close_t <= GRASP_HOLD:
                return [0, 0, 0, 0, 0, 0, +1]          # GRASP — 제자리 닫기 유지
            else:
                return C.go_to([eef[0], eef[1], C.Z_TRAVEL], eef, gripper=+1)  # lift-on-faith
        if _xy(eef, self.basket) < 0.09 and eef[2] < C.Z_TRAVEL - 0.03:
            return C.go_to([self.basket[0], self.basket[1], C.Z_TRAVEL], eef, gripper=-1)  # RETREAT
        xyerr = _xy(tp, eef)
        grasp_z = tp[2] + C.GRASP_Z_OFFSET
        if xyerr > 0.025:
            self.descend_t = 0
            z_tgt = C.Z_TRAVEL if xyerr > 0.10 else C.Z_PRE_GRASP
            return C.go_to([tp[0], tp[1], z_tgt], eef, gripper=-1)   # APPROACH
        if eef[2] > grasp_z + 0.012:
            self.descend_t += 1
            if not (self.descend_t > 35 and eef[2] < grasp_z + 0.035):  # plateau 관용
                return C.go_to([tp[0], tp[1], grasp_z], eef, gripper=-1)  # DESCEND
        self.closing = True
        self.close_t = 1
        self.descend_t = 0
        return [0, 0, 0, 0, 0, 0, +1]                                 # GRASP 시작


def run_episode(env, obs, policy, expert, max_steps, beta, rng, expert_on_holding=False):
    """settle은 호출측에서 완료된 상태로 진입 (chick_z0 캡처와 순서 일치).
    expert_on_holding=True: holding(병아리 든 상태) 감지 시 전문가 강제 제어 →
    모델이 자기 분포에서 파지한 상태로부터의 깨끗한 운반+저하강 투하 라벨만 농축(r4)."""
    wanderers = C.setup_wander(env, rng)
    traj = {k: [] for k in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_quat",
                            "gripper_qpos", "joint_pos", "actions",
                            "executed_actions", "executed_by")}
    expert_mode = True  # 첫 세그먼트는 전문가 (안정적 시작)
    success = False
    z0 = _max_dead_chick_z(env)
    zmax = z0
    n_expert = 0
    for t in range(max_steps):
        if t % SEGMENT_LEN == 0:
            was = expert_mode
            expert_mode = bool(rng.rand() < beta) if t > 0 else True
            if was and not expert_mode and policy is not None:
                policy.reset()  # 모델로 복귀 시 stale chunk 폐기
        # holding이면 전문가 강제 — 운반·투하 phase를 100% 전문가가 라벨 (release 신호 농축)
        if expert_on_holding and getattr(expert, "_was_holding", False):
            if not expert_mode and policy is not None:
                policy.reset()
            expert_mode = True
        exp_a = expert.action(obs)
        act = exp_a if (expert_mode or policy is None) else policy.act(obs, env, None)
        n_expert += int(expert_mode)

        traj["agentview_rgb"].append(obs["agentview_image"][::-1])
        traj["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"][::-1])
        traj["ee_pos"].append(np.array(obs["robot0_eef_pos"], np.float32))
        traj["ee_quat"].append(np.array(obs["robot0_eef_quat"], np.float32))
        traj["gripper_qpos"].append(np.array(obs["robot0_gripper_qpos"], np.float32))
        traj["joint_pos"].append(np.array(obs["robot0_joint_pos"], np.float32))
        traj["actions"].append(np.array(exp_a, np.float32))          # 라벨 = 전문가
        traj["executed_actions"].append(np.array(act, np.float32))
        traj["executed_by"].append(np.int8(expert_mode))

        if wanderers:
            C.step_wander(env, wanderers, rng)
        obs, done, term = C.safe_step(env, list(act))
        zc = _max_dead_chick_z(env)
        if zc == zc:
            zmax = max(zmax, zc)
        if term:
            break
        if done:
            success = True
            break
    lift = zmax - z0 if z0 == z0 else float("nan")
    return traj, success, lift, n_expert, t + 1


def save_h5(path, traj, seed, success):
    import h5py
    from thermal_fx import thermal_fx
    T = len(traj["actions"])
    with h5py.File(path, "w") as f:
        f.attrs["success"] = bool(success)
        f.attrs["language"] = LANG
        f.attrs["seed"] = int(seed)
        f.attrs["num_steps"] = T
        f.attrs["thermal"] = True
        f.attrs["dagger"] = True
        for k, v in traj.items():
            if k == "agentview_rgb":
                arr = np.stack([thermal_fx(im, colorbar=True) for im in v]).astype(np.uint8)
            elif k == "eye_in_hand_rgb":
                arr = np.stack(v).astype(np.uint8)
            else:
                arr = np.stack(v)
            f.create_dataset(k, data=arr, compression="gzip", compression_opts=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="merged ckpt (beta<1.0이면 필수)")
    ap.add_argument("--unnorm-key", default="blue_chick_thermal")
    ap.add_argument("--seeds", default="11000-11049")
    ap.add_argument("--n-good", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--beta", type=float, default=0.35, help="세그먼트가 전문가일 확률")
    ap.add_argument("--n-dead", type=int, default=2)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expert-on-holding", action="store_true",
                    help="r4: holding부터 전문가 강제 (운반·저하강 투하 라벨 농축)")
    ap.add_argument("--only-grasped", action="store_true",
                    help="r4: 파지 성공(lift>4cm)한 에피소드만 저장 (release 신호 순도↑)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))

    policy = None
    if args.beta < 1.0:
        if not args.ckpt:
            ap.error("--beta < 1.0 은 --ckpt 필요")
        policy = VLANavPolicy(args.ckpt, args.unnorm_key)
        print("[dagger] model loaded")
    else:
        print("[dagger] beta=1.0 → 전문가 단독 (모델 미로드, ShadowExpert 검증 모드)")
    print(f"[dagger] seeds={seeds} beta={args.beta}")
    n_saved = 0
    for seed in seeds:
        name = f"_dagger_{seed}.bddl"
        if not gen_bddl(seed, name, args.n_dead):
            continue
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_object", name)
        env = None
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                     camera_widths=256, horizon=args.max_steps + 300)
            env.seed(seed)
            obs = env.reset()
            if policy is not None:
                policy.reset()
            rng = np.random.RandomState(seed)
            dead = sorted(C.find_object_body_names(env, "dead_chick_"))
            for _ in range(SETTLE_STEPS):
                obs, _, term = C.safe_step(env, [0, 0, 0, 0, 0, 0, -1])
                if term:
                    raise RuntimeError("terminated during settle")
            # basket/chick_z0는 settle 후 캡처 (hybrid_eval과 동일)
            basket = C.get_body_pos(env, C.find_object_body_names(env, "basket_")[0])
            chick_z0 = {bd: C.get_body_pos(env, bd)[2] for bd in dead}
            expert = ShadowExpert(env, basket, chick_z0)
            out = run_episode(env, obs, policy, expert, args.max_steps,
                              args.beta, rng, expert_on_holding=args.expert_on_holding)
            if out is None:
                continue
            traj, success, lift, n_expert, steps = out
            if args.only_grasped and not (lift > 0.04):
                print(f"[dagger] seed {seed}: lift={lift*100:.1f}cm 미파지 → skip (--only-grasped)")
                continue
            path = os.path.join(args.out_dir, f"dagger_demo_seed{seed}.h5")
            save_h5(path, traj, seed, success)
            n_saved += 1
            print(f"[dagger] seed {seed}: steps={steps} success={success} "
                  f"lift={lift * 100:.1f}cm expert_steps={n_expert}/{steps} → {path}")
        except Exception as e:
            print(f"[dagger] seed {seed} SKIP: {type(e).__name__}: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        if n_saved >= args.n_good:
            break
    print(f"[dagger] done: {n_saved} episodes in {args.out_dir}")


if __name__ == "__main__":
    main()
