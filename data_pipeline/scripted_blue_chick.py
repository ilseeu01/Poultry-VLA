"""Ground-truth state 기반 스크립트 컨트롤러 — 파란 병아리(dead_chick) 모두를 바구니에 넣음.

OpenVLA 모델 없이 env 시뮬만 돌려서 successful demo MP4 생성.
나중에 이 trajectory를 demo data로 저장 → fine-tuning에 사용 가능.

사용:
    cd /home/capstone/openvla
    python experiments/robot/libero/scripted_blue_chick.py \
        --bddl pick_up_the_blue_chick_and_place_it_in_the_basket.bddl \
        --max-steps 1200 --out-dir rollouts/2026_05_11
"""
import argparse
import os
import sys
from datetime import datetime

import imageio
import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# 컨트롤러 게인/한계
GAIN_XY = 4.0      # XY 추종 게인 (v3: 8.0→4.0, fine zone 6cm→12.5cm로 확대해 XY 부드럽게·grasp 정밀↑)
GAIN_Z  = 6.0      # Z 추종 게인 (원복: 하강이 잡기 깊이까지 빠르게 도달해야 GRASP가 제 높이서 발동, 헛잡음 방지)
ACTION_CLIP = 0.5  # 한 step 당 최대 action 크기 (env scale 기준 [-1,1])
GRASP_HOLD_STEPS = 10  # 그리퍼 닫고 잡기 안정화 step 수 (no-op 줄이려 단축)
RELEASE_HOLD_STEPS = 8   # chick이 떨어져 basket에 안착할 시간 (no-op 줄이려 단축)

# 작업 z 레벨 (world frame, eef_pos 기준)
# 초기 eef_pos z = 0.261. Panda gripper tip은 eef보다 약 0.10m 아래.
# chick body main z = -0.025 (floor), 높이 ~7cm → chick top ~0.05.
# basket walls z=0.035~0.20, opening 중심 ~0.13.
Z_TRAVEL = 0.30      # 안전 이동 높이 (모든 객체 위, 도달 가능 한도 내)
Z_PRE_GRASP = 0.20   # 잡기 직전 hover (gripper tip ~ z=0.10)
Z_GRASP = 0.05       # (legacy) 고정 grasp z — 지금은 GRASP_Z_OFFSET 기반 적응형 사용
GRASP_Z_OFFSET = 0.049  # 잡을 때 eef가 병아리 body z보다 이만큼 위 (검증된 성공 간격)
Z_PRE_DROP = 0.32    # 바구니 위 hover
Z_DROP = 0.20        # 바구니 안 (gripper tip ~ z=0.10, basket 내부)

# 도달 임계
XY_THRESHOLD = 0.020
Z_THRESHOLD = 0.015  # GRASP 직전 z는 좀 더 엄격히


def get_body_pos(env, body_name):
    """body name으로 world frame 위치 조회."""
    sim = env.env.sim
    try:
        bid = sim.model.body_name2id(body_name)
    except Exception:
        # 객체 이름이 그대로 안 맞을 수 있어 _main 접미사 등 시도
        for suffix in ["_main", ""]:
            try:
                bid = sim.model.body_name2id(body_name + suffix)
                break
            except Exception:
                continue
        else:
            return None
    return np.array(sim.data.body_xpos[bid])


def find_object_body_names(env, prefix):
    """env에서 prefix로 시작하는 객체 body name 모두 검색."""
    sim = env.env.sim
    names = []
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if name and name.startswith(prefix):
            names.append(name)
    return names


def go_to(target_xyz, eef_pos, gripper, debug=""):
    """현재 eef → target_xyz delta action 계산.
    Returns 7-dim action: [dx, dy, dz, 0, 0, 0, gripper]."""
    dx = np.clip((target_xyz[0] - eef_pos[0]) * GAIN_XY, -ACTION_CLIP, ACTION_CLIP)
    dy = np.clip((target_xyz[1] - eef_pos[1]) * GAIN_XY, -ACTION_CLIP, ACTION_CLIP)
    dz = np.clip((target_xyz[2] - eef_pos[2]) * GAIN_Z,  -ACTION_CLIP, ACTION_CLIP)
    return [dx, dy, dz, 0.0, 0.0, 0.0, gripper]


def reached(eef_pos, target_xyz, xy_thr=XY_THRESHOLD, z_thr=Z_THRESHOLD):
    return (abs(eef_pos[0] - target_xyz[0]) < xy_thr and
            abs(eef_pos[1] - target_xyz[1]) < xy_thr and
            abs(eef_pos[2] - target_xyz[2]) < z_thr)


def reached_xy(eef_pos, target_xyz, xy_thr=XY_THRESHOLD):
    return (abs(eef_pos[0] - target_xyz[0]) < xy_thr and
            abs(eef_pos[1] - target_xyz[1]) < xy_thr)


def reached_z(eef_pos, z_target, z_thr=Z_THRESHOLD):
    return abs(eef_pos[2] - z_target) < z_thr


def safe_step(env, action):
    """env.step wrapper — terminated episode 예외 시 (None, True) 반환."""
    try:
        obs, _, done, _ = env.step(action)
        return obs, done, False
    except ValueError as e:
        if "terminated episode" in str(e):
            return None, True, True
        raise


ROBOT_BASE_XY = np.array([-0.6, 0.0])  # OnTheGroundPanda base offset from world origin
REACH_RADIUS = 0.55  # 안전 reach radius (Panda max ~0.85, 그리퍼 자세까지 고려해 보수적)
STATE_TIMEOUT_STEPS = 200  # state 진행 안 되면 다음 chick으로 skip (v3: 80→200, 낮춘 게인으로 느려진 단계 수용)


def in_workspace(xy):
    dx = xy[0] - ROBOT_BASE_XY[0]
    dy = xy[1] - ROBOT_BASE_XY[1]
    return (dx * dx + dy * dy) ** 0.5 < REACH_RADIUS


# --- 살아있는 병아리 배회(wander) ---
# kinematic 이동(qpos 직접 적분) + 회피(장애물 근처서 방향 전환) + 행동 상태(idle/walk/run).
# 이동 방향(heading)으로 몸을 회전 → 항상 앞을 보고 걸음.
WANDER_DT = 0.05                             # control step 간격 (control_freq=20 → 0.05s)
WANDER_BOUNDS = (-0.46, -0.02, -0.30, 0.30)  # xmin,xmax,ymin,ymax (카메라 화각 안)
WANDER_MARGIN = 0.05                         # 경계 안쪽 이만큼서부터 미리 복귀
WANDER_TURN = 0.06                           # rad/step — 평상시 heading random-walk
WANDER_TURN_MAX = 0.20                       # rad/step — 경계 복귀 회전
AVOID_TURN = 0.40                            # rad/step — 회피 회전 (빠르게 빗겨 가도록)
CHICK_YAW_OFFSET = np.pi / 2                 # 메시 머리는 body -Y(-90°)를 향함 → +90° 보정

# 행동 상태: 이름 -> (속도 m/s, 선택 가중치, 지속 최소step, 최대step).
# 실제 양계장 참고 — 닭은 대부분 멈춰 있고 가끔 짧게 살짝 이동.
CHICK_BEHAVIORS = {
    "idle": (0.000, 0.62,  60, 170),   # 대부분 정지 (길게)
    "walk": (0.015, 0.33,  15,  50),   # 가끔 살살 걷기 (짧게)
    "run":  (0.045, 0.05,  10,  28),   # 드물게 빠른 한걸음 (아주 짧게)
}
CHICK_RADIUS = 0.035                         # 병아리 회피 반경
DEAD_RADIUS = 0.035                          # 죽은 병아리 회피 반경
BASKET_RADIUS = 0.085                        # 바구니 회피 반경
INFLUENCE = 0.10                             # 장애물 반발 영향 반경 (hard 반경 + 이만큼)
REP_WEIGHT = 2.8                             # 장애물 반발력 가중치 (클수록 강하게 밀려남)


def yaw_quat(yaw):
    """Z축 yaw 회전 쿼터니언 (w, x, y, z)."""
    h = 0.5 * yaw
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def get_freejoint_addrs(env, body_name):
    """body freejoint의 (qpos 주소, qvel(dof) 주소). freejoint 없으면 None."""
    sim = env.env.sim
    try:
        bid = sim.model.body_name2id(body_name)
    except Exception:
        return None
    if sim.model.body_jntnum[bid] < 1:
        return None
    jadr = sim.model.body_jntadr[bid]
    return int(sim.model.jnt_qposadr[jadr]), int(sim.model.jnt_dofadr[jadr])


def _pick_behavior(rng):
    """가중치에 따라 행동 상태를 고르고 (이름, 지속 step수) 반환."""
    names = list(CHICK_BEHAVIORS)
    weights = np.array([CHICK_BEHAVIORS[n][1] for n in names], dtype=float)
    name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    lo, hi = CHICK_BEHAVIORS[name][2], CHICK_BEHAVIORS[name][3]
    return name, int(rng.randint(lo, hi))


def setup_wander(env, rng):
    """살아있는 병아리 배회 상태 + 회피 장애물 목록 초기화.
    반환: {"chicks":[...], "dead":[...], "basket": name}"""
    live = sorted(b for b in find_object_body_names(env, "chick_")
                  if not b.startswith("dead"))
    chicks = []
    for b in live:
        adrs = get_freejoint_addrs(env, b)
        if adrs is not None:
            beh, dur = _pick_behavior(rng)
            chicks.append({"body": b, "qposadr": adrs[0], "dofadr": adrs[1],
                           "heading": rng.uniform(0, 2 * np.pi),
                           "behavior": beh, "timer": dur})
    dead = sorted(find_object_body_names(env, "dead_chick_"))
    basket = find_object_body_names(env, "basket_")
    print(f"wandering live chicks: {[c['body'] for c in chicks]}")
    return {"chicks": chicks, "dead": dead, "basket": basket[0] if basket else None}


def _obstacles(env, ctx, self_body):
    """회피 대상 (x, y, radius) 목록 — 바닥의 죽은 병아리, 바구니, 다른 살아있는 병아리."""
    sim = env.env.sim
    obs = []
    for d in ctx["dead"]:
        p = get_body_pos(env, d)
        if p is not None and p[2] < 0.10:        # 바닥에 있는 죽은 병아리만 (들린 건 무시)
            obs.append((p[0], p[1], DEAD_RADIUS))
    if ctx["basket"]:
        p = get_body_pos(env, ctx["basket"])
        if p is not None:
            obs.append((p[0], p[1], BASKET_RADIUS))
    for c in ctx["chicks"]:
        if c["body"] != self_body:
            qa = c["qposadr"]
            obs.append((sim.data.qpos[qa], sim.data.qpos[qa + 1], CHICK_RADIUS))
    return obs


def step_wander(env, ctx, rng):
    """살아있는 병아리 한 스텝 — 행동 상태(idle/walk/run) + 포텐셜 필드 회피 + facing.

    회피는 근처 '모든' 장애물의 반발 벡터를 합산해서 처리 → 바구니+여러 병아리가
    동시에 있어도 합력 방향으로 빠져나감 (한 개씩 피하다 무리에 갇히는 문제 해결).
    매 step 반드시 이동 → 멈춤/끼임 없음. hard 반경 침범 시 표면으로 밀어냄."""
    sim = env.env.sim
    xmin, xmax, ymin, ymax = WANDER_BOUNDS
    for c in ctx["chicks"]:
        qa, da = c["qposadr"], c["dofadr"]
        x, y = sim.data.qpos[qa], sim.data.qpos[qa + 1]

        # 행동 상태 갱신 (멈춤/걷기/뜀 랜덤 전환)
        c["timer"] -= 1
        if c["timer"] <= 0:
            c["behavior"], c["timer"] = _pick_behavior(rng)
        speed = CHICK_BEHAVIORS[c["behavior"]][0]

        if speed > 0.0:
            # 1) wander 방향 (heading random-walk)
            c["heading"] += rng.uniform(-WANDER_TURN, WANDER_TURN)
            wx, wy = np.cos(c["heading"]), np.sin(c["heading"])

            # 2) 경계 반발 — margin 안쪽이면 중심 방향 성분 추가
            bx = by = 0.0
            if x < xmin + WANDER_MARGIN:
                bx += 1.0
            if x > xmax - WANDER_MARGIN:
                bx -= 1.0
            if y < ymin + WANDER_MARGIN:
                by += 1.0
            if y > ymax - WANDER_MARGIN:
                by -= 1.0

            # 3) 장애물 반발 — 근처 모든 장애물의 반발 벡터 합산
            rx = ry = 0.0
            hard_hits = []
            for (ox, oy, orad) in _obstacles(env, ctx, c["body"]):
                dx, dy = x - ox, y - oy
                dist = np.hypot(dx, dy) + 1e-6
                hard = CHICK_RADIUS + orad
                infl = hard + INFLUENCE
                if dist < infl:
                    s = (infl - dist) / infl          # 0(영향 끝)~1(접촉)
                    rx += (dx / dist) * s * s
                    ry += (dy / dist) * s * s
                if dist < hard:
                    hard_hits.append((ox, oy, hard))

            # 4) 합성 이동 방향 (wander + 장애물 반발 + 경계 반발)
            mx = wx + REP_WEIGHT * rx + 1.2 * bx
            my = wy + REP_WEIGHT * ry + 1.2 * by
            move_dir = np.arctan2(my, mx) if (mx * mx + my * my) > 1e-9 else c["heading"]

            # 5) heading 회전 — 반발 활성 시 빠르게
            active = (abs(rx) + abs(ry) + abs(bx) + abs(by)) > 1e-6
            lim = AVOID_TURN if active else WANDER_TURN_MAX
            d = (move_dir - c["heading"] + np.pi) % (2 * np.pi) - np.pi
            c["heading"] += np.clip(d, -lim, lim)

            # 6) 전진
            nx = x + speed * np.cos(c["heading"]) * WANDER_DT
            ny = y + speed * np.sin(c["heading"]) * WANDER_DT

            # 7) hard 반경 침범 방지 — 침범 시 표면으로 밀어냄
            for (ox, oy, hard) in hard_hits:
                ddx, ddy = nx - ox, ny - oy
                dd = np.hypot(ddx, ddy) + 1e-6
                if dd < hard:
                    nx = ox + (ddx / dd) * hard
                    ny = oy + (ddy / dd) * hard
            x, y = nx, ny

        # 위치/회전 적용 (idle이면 위치 그대로, 회전만 유지)
        sim.data.qpos[qa] = x
        sim.data.qpos[qa + 1] = y
        sim.data.qpos[qa + 3:qa + 7] = yaw_quat(c["heading"] + CHICK_YAW_OFFSET)
        sim.data.qvel[da:da + 6] = 0.0


def run_episode(env, max_steps, fps=30, settle_steps=8, seed=0, moving_chicks=True):
    """파란 병아리 모두를 바구니에 옮기는 state machine 실행.
    moving_chicks=True면 살아있는 병아리가 배회함 (죽은 병아리는 정지)."""
    # 객체 위치 조회를 위한 body name 찾기 (dead_chick_N_main 형태)
    obs = env.env._get_observations()
    dead_chicks_all = sorted(find_object_body_names(env, "dead_chick_"))
    basket_bodies = find_object_body_names(env, "basket_")
    # 워크스페이스 안의 chick만, 그리고 robot base에 가까운 순으로 정렬
    def dist_from_base(b):
        p = get_body_pos(env, b)
        return np.linalg.norm(p[:2] - ROBOT_BASE_XY) if p is not None else 1e9
    dead_chicks_reachable = [b for b in dead_chicks_all
                              if get_body_pos(env, b) is not None
                              and in_workspace(get_body_pos(env, b)[:2])]
    dead_chicks = sorted(dead_chicks_reachable, key=dist_from_base)
    unreachable = [b for b in dead_chicks_all if b not in dead_chicks]
    print(f"dead_chicks all: {dead_chicks_all}")
    print(f"reachable (sorted by distance): {dead_chicks}")
    if unreachable:
        print(f"unreachable (skipped): {unreachable}")
    print(f"basket bodies: {basket_bodies}")

    # trajectory: 매 step 마다 (obs, action) 기록 — fine-tuning용 demo로 저장 가능
    traj = {
        "agentview_rgb": [],
        "eye_in_hand_rgb": [],
        "ee_pos": [],
        "ee_quat": [],
        "gripper_qpos": [],
        "joint_pos": [],
        "actions": [],
    }

    def record_step(obs_, action_):
        traj["agentview_rgb"].append(obs_["agentview_image"][::-1])
        if "robot0_eye_in_hand_image" in obs_:
            traj["eye_in_hand_rgb"].append(obs_["robot0_eye_in_hand_image"][::-1])
        traj["ee_pos"].append(np.array(obs_["robot0_eef_pos"], dtype=np.float32))
        traj["ee_quat"].append(np.array(obs_["robot0_eef_quat"], dtype=np.float32))
        traj["gripper_qpos"].append(np.array(obs_["robot0_gripper_qpos"], dtype=np.float32))
        traj["joint_pos"].append(np.array(obs_["robot0_joint_pos"], dtype=np.float32))
        traj["actions"].append(np.array(action_, dtype=np.float32))

    if not dead_chicks or not basket_bodies:
        print("WARN: 필요한 객체를 찾지 못함")
        return traj, False

    # settle phase: 객체들이 떨어져 자리 잡을 시간
    settle_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(settle_steps):
        record_step(obs, settle_action)
        obs, done, terminated = safe_step(env, settle_action)
        if terminated:
            print("WARN: env terminated during settle phase")
            return traj, False

    basket_pos = get_body_pos(env, basket_bodies[0])

    # initial diagnostic
    init_eef = obs["robot0_eef_pos"]
    print(f"INIT eef_pos={init_eef}")
    print(f"INIT basket_pos={basket_pos}")
    for i, b in enumerate(dead_chicks):
        print(f"INIT {b} pos={get_body_pos(env, b)}")

    # 살아있는 병아리 배회 설정 (settle 끝난 뒤부터 움직임)
    rng = np.random.RandomState(seed)
    wanderers = setup_wander(env, rng) if moving_chicks else []

    target_idx = 0  # 현재 잡으려는 dead chick 인덱스
    state = "APPROACH_XY"
    state_counter = 0
    state_step_counter = 0  # state 진입 후 step 수
    prev_state = None

    for step in range(max_steps - settle_steps):
        if target_idx >= len(dead_chicks):
            # 모든 dead chick 처리 완료 → retreat
            action = [0, 0, ACTION_CLIP, 0, 0, 0, -1]
            record_step(obs, action)
            if wanderers:
                step_wander(env, wanderers, rng)
            obs, done, terminated = safe_step(env, action)
            if terminated:
                print(f"[step {step}] env terminated post-completion — treating as success")
                return traj, True
            if done:
                print(f"[step {step}] SUCCESS")
                return traj, True
            continue

        eef_pos = obs["robot0_eef_pos"]
        target_pos = get_body_pos(env, dead_chicks[target_idx])
        if target_pos is None:
            target_idx += 1
            continue

        if state != prev_state:
            print(f"[step {step}] state→{state} target={dead_chicks[target_idx]} "
                  f"eef={eef_pos.round(3)} tgt={target_pos.round(3)}")
            prev_state = state
            state_step_counter = 0
        else:
            state_step_counter += 1
        if step % 100 == 0:
            print(f"[step {step}] state={state} eef={eef_pos.round(3)} tgt_xy=({target_pos[0]:.3f},{target_pos[1]:.3f})")
        # APPROACH/DESCEND 단계가 너무 오래 걸리면 다음 chick으로 skip
        if state in ("APPROACH_XY", "DESCEND_PRE_GRASP", "DESCEND_TO_GRASP") and state_step_counter > STATE_TIMEOUT_STEPS:
            print(f"[step {step}] TIMEOUT in {state} for {dead_chicks[target_idx]}, skipping")
            target_idx += 1
            state = "APPROACH_XY"
            prev_state = None
            continue

        if state == "APPROACH_XY":
            # XY만 정렬 (z는 자유롭게 따라옴 — z 추종이 느려서 묶지 않음)
            tgt = [target_pos[0], target_pos[1], Z_TRAVEL]
            action = go_to(tgt, eef_pos, gripper=-1)
            if reached_xy(eef_pos, tgt, xy_thr=0.025):
                state = "DESCEND_PRE_GRASP"
                state_counter = 0

        elif state == "DESCEND_PRE_GRASP":
            # pre-grasp 높이로 내림
            tgt = [target_pos[0], target_pos[1], Z_PRE_GRASP]
            action = go_to(tgt, eef_pos, gripper=-1)
            if reached_z(eef_pos, Z_PRE_GRASP, z_thr=0.025) and reached_xy(eef_pos, tgt, xy_thr=0.025):
                state = "DESCEND_TO_GRASP"
                state_counter = 0

        elif state == "DESCEND_TO_GRASP":
            # 그리퍼 하강 목표를 병아리 '실제 z'에 상대적으로 잡음.
            # 병아리가 짚/사료 위에 얹혀 높이가 제각각이라 고정 Z는 파지 실패 원인.
            # 검증: chick_3 성공 시 eef-chick 간격 0.049 → GRASP_Z_OFFSET로 일관 적용.
            grasp_z = target_pos[2] + GRASP_Z_OFFSET
            tgt = [target_pos[0], target_pos[1], grasp_z]
            action = go_to(tgt, eef_pos, gripper=-1)
            if reached_z(eef_pos, grasp_z, z_thr=0.012) and reached_xy(eef_pos, tgt, xy_thr=0.020):
                state = "GRASP"
                state_counter = 0
            elif state_step_counter > 35 and reached_xy(eef_pos, tgt, xy_thr=0.025):
                # z plateau — 물리적으로 더 못 내려감. 목표 근처면 grasp, 너무 높으면 skip.
                if eef_pos[2] < grasp_z + 0.035:
                    print(f"  DESCEND plateau eef.z={eef_pos[2]:.3f} (target {grasp_z:.3f}) — forcing GRASP")
                    state = "GRASP"
                    state_counter = 0
                else:
                    print(f"  DESCEND plateau eef.z={eef_pos[2]:.3f} (target {grasp_z:.3f}, too high) — skipping")
                    target_idx += 1
                    state = "APPROACH_XY"
                    prev_state = None
                    continue

        elif state == "GRASP":
            # 그리퍼 닫고 K step 유지
            action = [0, 0, 0, 0, 0, 0, +1]
            state_counter += 1
            if state_counter == 1 or state_counter == GRASP_HOLD_STEPS:
                chick_z = target_pos[2]
                print(f"  GRASP step_counter={state_counter} chick_z={chick_z:.3f} eef.z={eef_pos[2]:.3f}")
            if state_counter >= GRASP_HOLD_STEPS:
                state = "LIFT"
                state_counter = 0

        elif state == "LIFT":
            # 그리퍼 닫은 채 위로 들어올림
            tgt = [eef_pos[0], eef_pos[1], Z_TRAVEL]
            action = go_to(tgt, eef_pos, gripper=+1)
            if state_step_counter % 10 == 0:
                chick_z = target_pos[2]
                print(f"  LIFT step_step={state_step_counter} chick_z={chick_z:.3f} eef.z={eef_pos[2]:.3f}")
            if eef_pos[2] >= Z_TRAVEL - 0.03:
                state = "MOVE_TO_BASKET"

        elif state == "MOVE_TO_BASKET":
            # 바구니 위로 이동 (XY만 일치 확인)
            tgt = [basket_pos[0], basket_pos[1], Z_PRE_DROP]
            action = go_to(tgt, eef_pos, gripper=+1)
            if state_step_counter % 10 == 0:
                print(f"  MOVE2BASKET ss={state_step_counter} chick_z={target_pos[2]:.3f} eef={eef_pos.round(2)}")
            if reached_xy(eef_pos, tgt, xy_thr=0.03):
                state = "DESCEND_OVER_BASKET"

        elif state == "DESCEND_OVER_BASKET":
            # 바구니 입구 위로 내림
            tgt = [basket_pos[0], basket_pos[1], Z_DROP]
            action = go_to(tgt, eef_pos, gripper=+1)
            if reached_z(eef_pos, Z_DROP, z_thr=0.025) and reached_xy(eef_pos, tgt, xy_thr=0.03):
                state = "RELEASE"
                state_counter = 0

        elif state == "RELEASE":
            # 그리퍼 열고 떨어뜨림
            action = [0, 0, 0, 0, 0, 0, -1]
            state_counter += 1
            if state_counter >= RELEASE_HOLD_STEPS:
                state = "RETREAT"
                state_counter = 0

        elif state == "RETREAT":
            # 바구니에서 빠져나옴
            tgt = [basket_pos[0], basket_pos[1], Z_TRAVEL]
            action = go_to(tgt, eef_pos, gripper=-1)
            if eef_pos[2] >= Z_TRAVEL - 0.03:
                target_idx += 1
                state = "APPROACH_XY"
                prev_state = None  # force state log on next iteration

        record_step(obs, action)
        if wanderers:
            step_wander(env, wanderers, rng)
        obs, done, terminated = safe_step(env, action)
        if terminated:
            print(f"[step {step}] env terminated (horizon) — last state={state}, target_idx={target_idx}")
            return traj, False

        if done:
            print(f"[step {step}] SUCCESS — all blue chicks in basket")
            return traj, True

    print(f"[step {max_steps}] max_steps reached, last state={state}, target_idx={target_idx}")
    # 최종 chick / basket 상태 출력 (왜 success가 안 됐는지 진단용)
    print("=== FINAL STATE ===")
    for b in dead_chicks_all:
        p = get_body_pos(env, b)
        if p is not None:
            print(f"  {b} final pos={p.round(3)}")
    print(f"  basket final pos={get_body_pos(env, basket_bodies[0]).round(3)}")
    try:
        print(f"  check_success={env.check_success()}")
    except Exception as e:
        print(f"  check_success error: {e}")
    return traj, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl", default="pick_up_the_blue_chick_and_place_it_in_the_basket.bddl")
    ap.add_argument("--problem-folder", default="libero_object")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-dir", default="rollouts/2026_05_11")
    ap.add_argument("--save-trajectory", default=None,
                    help="설정 시 trajectory를 HDF5로 저장. demo 수집용.")
    ap.add_argument("--save-mp4", action="store_true", default=True,
                    help="MP4 저장 여부 (기본 True). demo 수집 시 --no-save-mp4로 끔.")
    ap.add_argument("--no-save-mp4", dest="save_mp4", action="store_false")
    ap.add_argument("--language", default="Pick up the blue chick and place it in the basket",
                    help="HDF5에 저장될 task language instruction")
    ap.add_argument("--seed", type=int, default=0, help="env seed")
    ap.add_argument("--thermal", action="store_true",
                    help="IR 열화상 후처리(thermal_fx)를 MP4 + HDF5 이미지에 적용")
    ap.add_argument("--moving-chicks", action="store_true", default=True,
                    help="살아있는 병아리 배회 (기본 켜짐)")
    ap.add_argument("--static-chicks", dest="moving_chicks", action="store_false",
                    help="살아있는 병아리 정지")
    args = ap.parse_args()

    bddl_path = os.path.join(
        get_libero_path("bddl_files"), args.problem_folder, args.bddl
    )
    if not os.path.isfile(bddl_path):
        print(f"BDDL not found: {bddl_path}", file=sys.stderr)
        sys.exit(1)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        horizon=max(args.max_steps + 200, 2000),
    )
    env.seed(args.seed)
    env.reset()

    print(f"running scripted controller, max_steps={args.max_steps}, seed={args.seed}, "
          f"moving_chicks={args.moving_chicks}")
    traj, success = run_episode(env, max_steps=args.max_steps, fps=args.fps,
                                seed=args.seed, moving_chicks=args.moving_chicks)
    frames = traj["agentview_rgb"]

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    suffix = "success" if success else "fail"

    if args.save_mp4:
        tag = "thermal" if args.thermal else "scripted_blue_chick"
        out_path = os.path.join(
            args.out_dir,
            f"{stamp}--{tag}--{suffix}--{len(frames)}frames.mp4",
        )
        if args.thermal:
            from thermal_fx import thermal_fx
            print(f"applying IR thermal FX to {len(frames)} frames...")
        writer = imageio.get_writer(out_path, fps=args.fps)
        for f in frames:
            writer.append_data(thermal_fx(f) if args.thermal else f)
        writer.close()
        print(f"Saved MP4: {out_path}")

    if args.save_trajectory:
        import h5py
        os.makedirs(os.path.dirname(args.save_trajectory) or ".", exist_ok=True)
        # 학습용 이미지: thermal 모드면 열화상 후처리 + 온도 컬러바 포함.
        # 추론(run_libero_eval)에서도 동일하게 colorbar=True thermal_fx를 적용해야 일치.
        if args.thermal:
            from thermal_fx import thermal_fx
            print(f"applying IR thermal FX to HDF5 images ({len(traj['actions'])} steps)...")
        with h5py.File(args.save_trajectory, "w") as f:
            f.attrs["success"] = bool(success)
            f.attrs["language"] = args.language
            f.attrs["seed"] = int(args.seed)
            f.attrs["bddl"] = args.bddl
            f.attrs["num_steps"] = len(traj["actions"])
            f.attrs["thermal"] = bool(args.thermal)
            f.attrs["moving_chicks"] = bool(args.moving_chicks)
            for key, val in traj.items():
                if len(val) == 0:
                    continue
                if "rgb" in key:
                    if args.thermal:
                        arr = np.stack([thermal_fx(im, colorbar=True) for im in val])
                    else:
                        arr = np.stack(val)
                    arr = arr.astype(np.uint8)
                else:
                    arr = np.stack(val)
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        print(f"Saved trajectory HDF5: {args.save_trajectory} (success={success}, steps={len(traj['actions'])})")


if __name__ == "__main__":
    main()
