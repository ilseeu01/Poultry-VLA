#!/usr/bin/env python
"""Hybrid policy eval — NAV(정책) + GRASP/운반(스크립트 매크로) 핸드오프 상태기계.

배경(openvla/diag_depth/HYBRID_POLICY_PLAN.md): 전 ablation에서 VLA 단독 grasp 0 —
모델은 죽은 병아리까지 mm 단위로 접근하지만(navigation OK) 잡기높이 하강+닫기
commit을 못 함. → 역할 분담: navigation은 정책(Phase A=스크립트 스텁, Phase B=실제
VLA), 정밀 파지+운반은 scripted_blue_chick.py의 검증된 매크로가 담당.

상태기계:
  NAV(정책 제어)
    --[가장 가까운 남은 dead_chick까지 eef XY < HANDOFF_XY가 HANDOFF_CONSEC step 연속]-->
  RECENTER(XY 미세정렬 + pre-grasp 높이) -> DESCEND(live 추종, grasp_z=chick_z+0.049)
  -> GRASP(닫고 hold 15) -> LIFT(수직 상승; chick이 안 들렸으면 NAV로 재시도)
  -> MOVE_TO_BASKET -> DESCEND_OVER_BASKET -> RELEASE -> RETREAT(안착 검증)
  -> 남은 chick 있으면 NAV 복귀, 없으면 WAIT_SUCCESS.

매크로는 scripted_blue_chick.py(원본, firm grasp+운반 검증됨)의 게인/임계/로직을
그대로 재사용. 평가 장면은 학습분포(generate_chicken_farm_bddl --seed 랜덤 장면
+ wander)로 oft_eval_traindist.py와 동일. grasp 계측(lift>4cm)도 동일 기준.

Phase A (GPU 불필요, conda openvla 또는 oft):
  MUJOCO_GL=osmesa /home/capstone/miniconda3/envs/openvla/bin/python hybrid_eval.py \
      --policy stub --seeds 9000-9007 --grasp-log hybrid_stub.jsonl \
      --save-video-dir /home/capstone/openvla/rollouts/2026_06_12_hybrid
Phase B (GPU 확보 후, conda oft):
  --policy vla --ckpt <merged_ckpt_dir> [--unnorm-key blue_chick_thermal]
  (VLANavPolicy는 oft_eval_traindist.py 경로 그대로의 스켈레톤 — GPU에서 첫 검증 필요)
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse
import json
import subprocess
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/home/capstone/LIBERO")  # editable install MAPPING 깨짐 → 소스 직접
sys.path.insert(0, "/home/capstone/openvla/experiments/robot/libero")  # scripted_blue_chick, thermal_fx

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import scripted_blue_chick as C

GENERATOR = "/home/capstone/LIBERO/scripts/generate_chicken_farm_bddl.py"
LANG = "pick up the blue chick and place it in the basket"

# --- 핸드오프/매크로 파라미터 ---
HANDOFF_XY = 0.03           # NAV→매크로 트리거: eef-chick XY 거리 (m)
HANDOFF_CONSEC = 5          # 트리거 연속 충족 step 수
MACRO_GRASP_HOLD = 15       # 그리퍼 닫고 유지 step (원본 10, 플랜 권고 15)
MAX_GRASP_RETRIES = 2       # chick당 매크로 재시도 한도 (초과 시 skip)
NAV_Z = C.Z_TRAVEL          # 스텁 NAV 비행 높이
CHICK_LIFT_OK = 0.04        # 이만큼 들리면 파지 성공 (grasp 계측과 동일 4cm 기준)
BASKET_PLACE_RADIUS = 0.10  # chick XY가 바구니 중심 이내면 안착으로 인정
SETTLE_STEPS = 10           # oft_eval_traindist와 동일 settle
STATE_TIMEOUT = 200         # 매크로 state 정체 한도 (원본 STATE_TIMEOUT_STEPS)
WAIT_SUCCESS_STEPS = 60     # 전부 처리 후 done 신호 대기 한도

MACRO_PHASES = ("RECENTER", "DESCEND", "GRASP", "LIFT",
                "MOVE_TO_BASKET", "DESCEND_OVER_BASKET", "RELEASE", "RETREAT")
PHASE_CODE = {p: i for i, p in enumerate(("NAV",) + MACRO_PHASES + ("WAIT_SUCCESS",))}


def _max_dead_chick_z(env):
    """Max world-z over dead_chick bodies (lift 계측). NaN if none/error."""
    try:
        sim = env.env.sim
        zs = [float(sim.data.body_xpos[i][2]) for i in range(sim.model.nbody)
              if (sim.model.body_id2name(i) or "").startswith("dead_chick")]
        return max(zs) if zs else float("nan")
    except Exception:
        return float("nan")


def gen_bddl(seed, name, n_dead=2):
    r = subprocess.run([sys.executable, GENERATOR, "--out", name, "--no-arena",
                        "--seed", str(seed), "--n-dead", str(n_dead)],
                       cwd="/home/capstone/LIBERO", capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[hybrid] bddl gen stderr: {r.stderr[-500:]}")
    return r.returncode == 0


class StubNavPolicy:
    """Phase A: 모델 자리 스텁 — 가장 가까운 남은 dead_chick 위로 XY 접근(그리퍼 열림).
    ground-truth target 사용은 스텁이라 허용. --nav-noise로 모델류 부정확성 모사 가능."""

    def __init__(self, noise_std=0.0, rng=None):
        self.noise_std = float(noise_std)
        self.rng = rng if rng is not None else np.random.RandomState(0)

    def reset(self):
        pass

    def act(self, obs, env, nav_target):
        eef = obs["robot0_eef_pos"]
        a = C.go_to([nav_target[0], nav_target[1], NAV_Z], eef, gripper=-1)
        if self.noise_std > 0:
            a[0] = float(np.clip(a[0] + self.rng.normal(0, self.noise_std),
                                 -C.ACTION_CLIP, C.ACTION_CLIP))
            a[1] = float(np.clip(a[1] + self.rng.normal(0, self.noise_std),
                                 -C.ACTION_CLIP, C.ACTION_CLIP))
        return a


class VLANavPolicy:
    """Phase B: 실제 VLA가 NAV 담당 (conda oft + GPU 필요).
    oft_eval_traindist.py의 모델 추론 경로를 그대로 캡슐화.
    ⚠️ GPU 미확보 상태에서 작성된 스켈레톤 — Phase B 첫 실행에서 검증할 것."""

    def __init__(self, ckpt, unnorm_key="blue_chick_thermal"):
        os.environ.setdefault("BLUE_CHICK_THERMAL", "1")  # 양계장 obs 일치 (thermal+상하반전)
        sys.path.insert(0, "/home/capstone/openvla-oft")
        from experiments.robot.libero.run_libero_eval import (
            GenerateConfig, initialize_model, prepare_observation, process_action, get_action)
        from experiments.robot.robot_utils import get_image_resize_size
        self._prepare_observation = prepare_observation
        self._process_action = process_action
        self._get_action = get_action
        self.cfg = GenerateConfig(pretrained_checkpoint=ckpt, model_family="openvla",
                                  num_images_in_input=2, use_proprio=True, center_crop=True,
                                  num_open_loop_steps=8, unnorm_key=unnorm_key,
                                  task_suite_name="libero_object")
        (self.model, self.action_head, self.proprio_projector,
         self.noisy_action_projector, self.processor) = initialize_model(self.cfg)
        self.resize_size = get_image_resize_size(self.cfg)
        self.queue = deque(maxlen=self.cfg.num_open_loop_steps)

    def reset(self):
        # 핸드오프/NAV 복귀 시 stale action chunk 폐기 (chunk가 grasp 시점을 물고 있지 않게)
        self.queue.clear()

    def act(self, obs, env, nav_target_unused):
        observation, _ = self._prepare_observation(obs, self.resize_size)
        if len(self.queue) == 0:
            acts = self._get_action(self.cfg, self.model, observation, LANG,
                                    processor=self.processor, action_head=self.action_head,
                                    proprio_projector=self.proprio_projector,
                                    noisy_action_projector=self.noisy_action_projector,
                                    use_film=self.cfg.use_film)
            self.queue.extend(acts)
        return self._process_action(self.queue.popleft(), self.cfg.model_family).tolist()


_EMPTY_REC = {"seed": None, "steps": 0, "success": False, "terminated": False,
              "lift_m": None, "grasped": False, "handoffs": 0, "placed": [],
              "skipped": [], "retries": {}, "nav_steps": 0, "macro_steps": 0}


def run_episode_hybrid(env, obs, nav_policy, seed, max_steps=700, moving_chicks=True,
                       handoff_xy=HANDOFF_XY, handoff_consec=HANDOFF_CONSEC,
                       frames=None, trace=None, dump=None,
                       trigger_fn=None, trigger_thr=0.5, handoff_mode="approach"):
    """핸드오프 상태기계 1 에피소드. 반환: per-episode record dict."""
    rng = np.random.RandomState(seed)
    wanderers = C.setup_wander(env, rng) if moving_chicks else None
    if dump is not None or trigger_fn is not None:
        from thermal_fx import thermal_fx as _tfx
        import robosuite.utils.transform_utils as _T

    # settle (oft_eval_traindist와 동일: dummy action 10 step, wander 포함)
    for _ in range(SETTLE_STEPS):
        if wanderers:
            C.step_wander(env, wanderers, rng)
        obs, done, terminated = C.safe_step(env, [0, 0, 0, 0, 0, 0, -1])
        if terminated:
            return dict(_EMPTY_REC, seed=seed, terminated=True,
                        error="terminated during settle")

    dead_all = sorted(C.find_object_body_names(env, "dead_chick_"))
    basket_bodies = C.find_object_body_names(env, "basket_")
    if not dead_all or not basket_bodies:
        return dict(_EMPTY_REC, seed=seed, error="objects not found")
    basket_pos = C.get_body_pos(env, basket_bodies[0])

    # settle 후 바닥 기준 z (스폰 정착 아티팩트 회피 — 베이스라인은 반드시 settle 이후)
    chick_z0 = {b: (C.get_body_pos(env, b)[2] if C.get_body_pos(env, b) is not None
                    else float("nan")) for b in dead_all}
    z0_global = _max_dead_chick_z(env)
    zmax_global = z0_global

    placed, skipped = set(), set()
    retries = {b: 0 for b in dead_all}
    phase, prev_phase = "NAV", None
    phase_steps = 0
    macro_target = None
    streak = 0
    handoffs = 0
    nav_steps = macro_steps = 0
    success = False
    terminated = False
    done = False
    t = 0

    def remaining_chicks():
        out = []
        for b in dead_all:
            if b in placed or b in skipped:
                continue
            p = C.get_body_pos(env, b)
            if p is None:
                continue
            if (np.hypot(p[0] - basket_pos[0], p[1] - basket_pos[1]) < BASKET_PLACE_RADIUS
                    and p[2] < 0.12):
                # 바구니 영역 + 낮은 z = 안착 (postgrasp서 들고 통과 시 오판 방지)
                placed.add(b)
                continue
            if not C.in_workspace(p[:2]):
                skipped.add(b)
                continue
            out.append((b, p))
        return out

    def fail_attempt(reason):
        nonlocal phase, macro_target, streak
        retries[macro_target] += 1
        give_up = retries[macro_target] > MAX_GRASP_RETRIES
        print(f"[hybrid][{t}] macro FAIL({reason}) target={macro_target} "
              f"retry={retries[macro_target]}" + (" → SKIP" if give_up else " → NAV 재시도"))
        if give_up:
            skipped.add(macro_target)
        macro_target = None
        streak = 0
        phase = "NAV"

    for t in range(max_steps):
        if phase != prev_phase:
            print(f"[hybrid][{t}] phase→{phase}"
                  + (f" target={macro_target}" if macro_target else ""))
            prev_phase = phase
            phase_steps = 0
        else:
            phase_steps += 1

        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        zc = _max_dead_chick_z(env)
        if zc == zc:
            zmax_global = max(zmax_global, zc)
        action = [0, 0, 0, 0, 0, 0, -1]
        nearest_d = float("nan")

        if phase == "NAV":
            rem = remaining_chicks()
            if not rem:
                phase = "WAIT_SUCCESS"
                continue
            dists = [np.hypot(p[0] - eef[0], p[1] - eef[1]) for _, p in rem]
            i = int(np.argmin(dists))
            tgt_body, tgt_pos = rem[i]
            nearest_d = float(dists[i])
            action = nav_policy.act(obs, env, tgt_pos)
            nav_steps += 1
            if handoff_mode == "postgrasp":
                # ②: 모델이 직접 NAV+grasp+lift. 병아리 들림(>4cm)+근접 시 운반 매크로로.
                # 현재 action(모델)은 이 step 그대로 실행, 다음 step부터 매크로가 운반.
                lifted = None
                for b in dead_all:
                    if b in placed or b in skipped:
                        continue
                    p = C.get_body_pos(env, b)
                    if p is not None and (p[2] - chick_z0[b] > CHICK_LIFT_OK
                                          and np.hypot(p[0] - eef[0], p[1] - eef[1]) < 0.07):
                        lifted = b
                        break
                if lifted is not None:
                    macro_target = lifted
                    handoffs += 1
                    nav_policy.reset()
                    phase = "MOVE_TO_BASKET"  # grasp/lift는 모델이 이미 함 → 운반부터
                    print(f"[hybrid][{t}] POSTGRASP HANDOFF #{handoffs} → {lifted} "
                          f"(lift={(C.get_body_pos(env, lifted)[2]-chick_z0[lifted])*100:.1f}cm)")
            else:
                if dump is not None or trigger_fn is not None:
                    ag = _tfx(obs["agentview_image"][::-1], colorbar=True)
                    wr = obs["robot0_eye_in_hand_image"][::-1]
                    prop = np.concatenate([
                        obs["robot0_eef_pos"],
                        _T.quat2axisangle(np.array(obs["robot0_eef_quat"])),
                        obs["robot0_gripper_qpos"]]).astype(np.float32)
                if dump is not None:
                    # 학습 트리거용: NAV 관측 + GT 거리 라벨 (eval 분포 = NAV 프레임만)
                    dump["agentview"].append(ag)
                    dump["wrist"].append(wr)
                    dump["proprio"].append(prop)
                    dump["dist"].append(nearest_d)
                if trigger_fn is not None:
                    # 학습 트리거: 관측만으로 핸드오프 판정 (GT 거리 미사용)
                    trig_prob = trigger_fn(ag, wr, prop)
                    streak = streak + 1 if trig_prob > trigger_thr else 0
                else:
                    streak = streak + 1 if nearest_d < handoff_xy else 0
                if streak >= handoff_consec:
                    macro_target = tgt_body
                    handoffs += 1
                    streak = 0
                    nav_policy.reset()
                    phase = "RECENTER"
                    print(f"[hybrid][{t}] HANDOFF #{handoffs} → {tgt_body} (xy={nearest_d:.3f})")

        elif phase in MACRO_PHASES:
            macro_steps += 1
            tp = C.get_body_pos(env, macro_target)
            if tp is None:
                fail_attempt("target lost")
                continue
            if phase_steps > STATE_TIMEOUT:
                fail_attempt(f"timeout in {phase}")
                continue
            nearest_d = float(np.hypot(tp[0] - eef[0], tp[1] - eef[1]))

            if phase == "RECENTER":
                # XY 미세정렬 + pre-grasp 높이 (원본 DESCEND_PRE_GRASP, XY만 더 타이트)
                tgt = [tp[0], tp[1], C.Z_PRE_GRASP]
                action = C.go_to(tgt, eef, gripper=-1)
                if C.reached_xy(eef, tgt, xy_thr=0.020) and C.reached_z(eef, C.Z_PRE_GRASP, z_thr=0.025):
                    phase = "DESCEND"

            elif phase == "DESCEND":
                # 원본 DESCEND_TO_GRASP 그대로 — live 추종(coupled), 적응형 grasp_z.
                # 재시도면 4mm씩 더 낮게 — 결정론적 동일 실패 반복 방지 (plateau가 과하강 보호)
                grasp_z = tp[2] + C.GRASP_Z_OFFSET - 0.004 * retries[macro_target]
                tgt = [tp[0], tp[1], grasp_z]
                action = C.go_to(tgt, eef, gripper=-1)
                if C.reached_z(eef, grasp_z, z_thr=0.012) and C.reached_xy(eef, tgt, xy_thr=0.020):
                    phase = "GRASP"
                elif phase_steps > 35 and C.reached_xy(eef, tgt, xy_thr=0.025):
                    if eef[2] < grasp_z + 0.035:
                        print(f"[hybrid][{t}] DESCEND plateau eef.z={eef[2]:.3f} "
                              f"(target {grasp_z:.3f}) — forcing GRASP")
                        phase = "GRASP"
                    else:
                        fail_attempt(f"descend plateau too high (eef.z={eef[2]:.3f})")
                        continue

            elif phase == "GRASP":
                action = [0, 0, 0, 0, 0, 0, +1]
                if phase_steps >= MACRO_GRASP_HOLD:
                    phase = "LIFT"

            elif phase == "LIFT":
                action = C.go_to([eef[0], eef[1], C.Z_TRAVEL], eef, gripper=+1)
                if eef[2] >= C.Z_TRAVEL - 0.03:
                    lift = tp[2] - chick_z0[macro_target]
                    if lift > CHICK_LIFT_OK:
                        print(f"[hybrid][{t}] LIFT ok — {macro_target} lift={lift * 100:.1f}cm")
                        phase = "MOVE_TO_BASKET"
                    else:
                        fail_attempt(f"no lift ({lift * 100:.1f}cm)")
                        continue

            elif phase == "MOVE_TO_BASKET":
                tgt = [basket_pos[0], basket_pos[1], C.Z_PRE_DROP]
                action = C.go_to(tgt, eef, gripper=+1)
                if tp[2] - chick_z0[macro_target] < 0.02:
                    fail_attempt("slipped in transport")
                    continue
                if C.reached_xy(eef, tgt, xy_thr=0.03):
                    phase = "DESCEND_OVER_BASKET"

            elif phase == "DESCEND_OVER_BASKET":
                tgt = [basket_pos[0], basket_pos[1], C.Z_DROP]
                action = C.go_to(tgt, eef, gripper=+1)
                if tp[2] - chick_z0[macro_target] < 0.02:
                    fail_attempt("slipped over basket")
                    continue
                if C.reached_z(eef, C.Z_DROP, z_thr=0.025) and C.reached_xy(eef, tgt, xy_thr=0.03):
                    phase = "RELEASE"

            elif phase == "RELEASE":
                action = [0, 0, 0, 0, 0, 0, -1]
                if phase_steps >= C.RELEASE_HOLD_STEPS:
                    phase = "RETREAT"

            elif phase == "RETREAT":
                tgt = [basket_pos[0], basket_pos[1], C.Z_TRAVEL]
                action = C.go_to(tgt, eef, gripper=-1)
                if eef[2] >= C.Z_TRAVEL - 0.03:
                    in_basket = np.hypot(tp[0] - basket_pos[0], tp[1] - basket_pos[1]) \
                        < BASKET_PLACE_RADIUS
                    if in_basket:
                        placed.add(macro_target)
                        print(f"[hybrid][{t}] PLACED {macro_target} "
                              f"({len(placed)}/{len(dead_all)})")
                        macro_target = None
                        phase = "NAV"
                    else:
                        fail_attempt("release missed basket")
                        continue

        elif phase == "WAIT_SUCCESS":
            # 전부 처리됨 — 위로 빠지며 done 신호 대기 (원본 post-completion retreat)
            action = [0, 0, C.ACTION_CLIP, 0, 0, 0, -1]
            if phase_steps > WAIT_SUCCESS_STEPS:
                print(f"[hybrid][{t}] WAIT_SUCCESS expired without done")
                break

        if frames is not None:
            frames.append(obs["agentview_image"][::-1])
        if trace is not None:
            trace.append([t, eef[0], eef[1], eef[2], PHASE_CODE[phase],
                          nearest_d, float(action[-1])])
        if wanderers:
            C.step_wander(env, wanderers, rng)
        obs, done, terminated = C.safe_step(env, action)
        if terminated:
            print(f"[hybrid][{t}] env terminated (phase={phase})")
            break
        if done:
            success = True
            print(f"[hybrid][{t}] SUCCESS — env done")
            break

    if not success:
        try:
            success = bool(env.check_success())
        except Exception:
            pass

    lift_m = (zmax_global - z0_global) if (z0_global == z0_global) else float("nan")
    # 최종 투하 정밀도: 파지된(들렸던) 병아리의 바구니 중심까지 XY거리 + 안착 z
    fin_d = fin_z = None
    best = None
    for b in dead_all:
        p = C.get_body_pos(env, b)
        if p is None or chick_z0.get(b) is None:
            continue
        if zmax_global - chick_z0[b] > CHICK_LIFT_OK or (lift_m == lift_m and lift_m > CHICK_LIFT_OK):
            d = float(np.hypot(p[0] - basket_pos[0], p[1] - basket_pos[1]))
            if best is None or d < best:
                best, fin_z = d, float(p[2])
    fin_d = round(best, 4) if best is not None else None
    return {
        "seed": seed,
        "steps": t,
        "success": bool(success),
        "terminated": bool(terminated),
        "lift_m": round(float(lift_m), 4) if lift_m == lift_m else None,
        "grasped": bool(lift_m == lift_m and lift_m > CHICK_LIFT_OK),
        "final_dist_basket": fin_d,
        "final_z": round(fin_z, 4) if fin_z is not None else None,
        "handoffs": handoffs,
        "placed": sorted(placed),
        "skipped": sorted(skipped),
        "retries": retries,
        "nav_steps": nav_steps,
        "macro_steps": macro_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["stub", "vla"], default="stub")
    ap.add_argument("--ckpt", default=None, help="(vla) merged checkpoint dir")
    ap.add_argument("--unnorm-key", default="blue_chick_thermal")
    ap.add_argument("--seeds", default="9000-9007")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--n-good", type=int, default=8, help="이만큼 에피소드 완료 시 중단")
    ap.add_argument("--handoff-xy", type=float, default=HANDOFF_XY)
    ap.add_argument("--handoff-consec", type=int, default=HANDOFF_CONSEC)
    ap.add_argument("--handoff-mode", choices=["approach", "postgrasp"], default="approach",
                    help="approach=모델NAV→매크로grasp+운반(기존) / postgrasp=②모델NAV+grasp+lift→매크로 운반+투하")
    ap.add_argument("--nav-noise", type=float, default=0.0,
                    help="(stub) NAV action XY 가우시안 노이즈 std — 모델 부정확성 모사")
    ap.add_argument("--n-dead", type=int, default=2,
                    help="장면 dead_chick 수 (1=단수 language 정합 goal)")
    ap.add_argument("--static-chicks", action="store_true", help="살아있는 병아리 정지(디버그)")
    ap.add_argument("--grasp-log", default=None, help="per-episode JSONL append 경로")
    ap.add_argument("--save-video-dir", default=None, help="설정 시 agentview mp4 저장")
    ap.add_argument("--thermal-video", action="store_true",
                    help="mp4에 thermal_fx 적용 (느림, 기본 raw RGB)")
    ap.add_argument("--trace-dir", default=None, help="per-step trace npy 저장 디렉토리")
    ap.add_argument("--dump-trigger-data", default=None,
                    help="학습 트리거용 NAV 관측+거리 라벨 npz 저장 디렉토리")
    ap.add_argument("--trigger", choices=["gt", "learned"], default="gt",
                    help="핸드오프 판정: gt=sim 거리(XY<handoff-xy) / learned=분류기")
    ap.add_argument("--trigger-ckpt",
                    default="/home/capstone/openvla_ckpts/trigger_ckpt/trigger_resnet18.pt")
    ap.add_argument("--trigger-thr", type=float, default=0.5)
    args = ap.parse_args()

    trigger_fn = None
    if args.trigger == "learned":
        from train_trigger import load_trigger
        trigger_fn = load_trigger(args.trigger_ckpt).predict
        print(f"[hybrid] learned trigger loaded: {args.trigger_ckpt} (thr={args.trigger_thr})")

    if args.policy == "vla":
        if not args.ckpt:
            ap.error("--policy vla 는 --ckpt 필요")
        nav_policy = VLANavPolicy(args.ckpt, args.unnorm_key)
    else:
        nav_policy = StubNavPolicy(noise_std=args.nav_noise)

    if args.grasp_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.grasp_log)), exist_ok=True)
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    print(f"[hybrid] policy={args.policy} seeds={seeds} max_steps={args.max_steps} "
          f"handoff(xy<{args.handoff_xy}, {args.handoff_consec} consec) "
          f"nav_noise={args.nav_noise}")

    results = []
    for seed in seeds:
        bddl_name = f"_hybrid_{seed}.bddl"
        if not gen_bddl(seed, bddl_name, args.n_dead):
            print(f"[hybrid] seed {seed} bddl gen fail, skip")
            continue
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_object", bddl_name)
        env = None
        rec = None
        frames = [] if args.save_video_dir else None
        trace = [] if args.trace_dir else None
        dump = ({"agentview": [], "wrist": [], "proprio": [], "dist": []}
                if args.dump_trigger_data else None)
        try:
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                     camera_widths=256, horizon=args.max_steps + 300)
            env.seed(seed)
            obs = env.reset()  # 나쁜 장면이면 RandomizationError → per-seed skip
            nav_policy.reset()
            rec = run_episode_hybrid(
                env, obs, nav_policy, seed, max_steps=args.max_steps,
                moving_chicks=not args.static_chicks,
                handoff_xy=args.handoff_xy, handoff_consec=args.handoff_consec,
                frames=frames, trace=trace, dump=dump,
                trigger_fn=trigger_fn, trigger_thr=args.trigger_thr,
                handoff_mode=args.handoff_mode)
        except Exception as e:
            print(f"[hybrid] seed {seed} SKIP: {type(e).__name__}: {e}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        if rec is None:
            continue
        rec["policy"] = args.policy
        rec["trigger"] = args.trigger
        results.append(rec)
        print(f"[hybrid] seed {seed}: success={rec['success']} "
              f"lift={rec['lift_m']}m grasped={rec['grasped']} "
              f"handoffs={rec['handoffs']} placed={rec['placed']} steps={rec['steps']}")
        # 후처리 I/O는 에피소드 결과와 분리 — 실패해도 episode는 유효 (경고만)
        try:
            if args.grasp_log:
                with open(args.grasp_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if frames:
                import imageio
                os.makedirs(args.save_video_dir, exist_ok=True)
                tag = "success" if rec["success"] else "fail"
                out = os.path.join(args.save_video_dir,
                                   f"hybrid_{args.policy}_seed{seed}_{tag}.mp4")
                if args.thermal_video:
                    from thermal_fx import thermal_fx
                    frames = [thermal_fx(f) for f in frames]
                w = imageio.get_writer(out, fps=30)
                for f in frames:
                    w.append_data(f)
                w.close()
                print(f"[hybrid] saved video: {out}")
            if trace:
                os.makedirs(args.trace_dir, exist_ok=True)
                np.save(os.path.join(args.trace_dir, f"hybrid_trace_seed{seed}.npy"),
                        np.array(trace, dtype=float))
            if dump and dump["dist"]:
                os.makedirs(args.dump_trigger_data, exist_ok=True)
                np.savez_compressed(
                    os.path.join(args.dump_trigger_data, f"trig_seed{seed}.npz"),
                    agentview=np.stack(dump["agentview"]).astype(np.uint8),
                    wrist=np.stack(dump["wrist"]).astype(np.uint8),
                    proprio=np.stack(dump["proprio"]),
                    dist=np.array(dump["dist"], dtype=np.float32),
                    success=rec["success"], n_dead=args.n_dead,
                    nav_noise=args.nav_noise, policy=args.policy)
                print(f"[hybrid] dumped trigger data: {len(dump['dist'])} NAV frames")
        except Exception as e:
            print(f"[hybrid] seed {seed} WARN post-episode I/O failed: "
                  f"{type(e).__name__}: {e}")
        if len(results) >= args.n_good:
            break

    n = len(results)
    s = sum(r["success"] for r in results)
    g = sum(r["grasped"] for r in results)
    lifts = [r["lift_m"] for r in results if r.get("lift_m") is not None]
    print(f"\n[hybrid] SUMMARY policy={args.policy} n={n} success={s}/{n} grasped={g}/{n} "
          f"lifts(cm)={[round(100 * x, 1) for x in lifts]} "
          f"handoffs={[r['handoffs'] for r in results]}")


if __name__ == "__main__":
    main()
