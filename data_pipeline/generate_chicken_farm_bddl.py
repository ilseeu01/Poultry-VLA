"""양계장 시뮬레이션용 BDDL 자동 생성기.

사용 예시:
    python generate_chicken_farm_bddl.py \
        --n-chicks 5 --n-hens 3 --n-dead 2 \
        --out pick_up_dead_chick_from_farm.bddl \
        --task "Pick up the dead chick and place it in the basket"

병아리/성체닭/사체닭/바구니를 floor에 비-중첩으로 랜덤 배치한 BDDL을 출력.
goal은 "사체닭(또는 지정한 객체)을 basket에 넣기" 로 자동 설정.
"""
import argparse
import os
import random


# floor의 사용 가능한 XY 범위. reach=0.40 + robot base=-0.6 이라 chick은 사실상 x∈[-0.45, -0.20]만 사용.
# 이전 -0.30 limit는 placement 구간을 10cm로 좁혀 객체 6개 배치 실패율 ↑ → -0.50으로 확장.
FLOOR_XMIN, FLOOR_XMAX = -0.50, 0.30
FLOOR_YMIN, FLOOR_YMAX = -0.35, 0.35

# 객체별 horizontal_radius (XML horizontal_radius_site 기준 — LIBERO의 placement에 사용)
HORIZONTAL_RADIUS = {
    "chick_crouching":   0.030,
    "chick_looking_left": 0.025,
    "chicken":            0.060,
    "dead_chick":         0.030,
    # IR 그라데이션 살아있는 병아리 (chick_crouching mesh 재사용, h_r 동일)
    "chick_warm_green":   0.030,
    "chick_warm_yellow":  0.030,
    "chick_warm_orange":  0.030,
    "chick_warm_red":     0.030,
    "basket":             0.025,
    "straw_piece":        0.005,
    "feed_pellet":        0.005,
    "manure_pile":        0.010,
}

# 살아있는 병아리(IR warm) 변형 목록 — generator가 랜덤으로 선택
WARM_CHICK_TYPES = ["chick_warm_green", "chick_warm_yellow", "chick_warm_orange", "chick_warm_red"]

# 바구니 고정 위치 (양계장에서 폐사 처리 통은 항상 같은 자리).
# iter 11 검증 위치 — 모든 chick에서 RELEASE 성공한 좌표.
FIXED_BASKET_XY = (-0.113, 0.002)

# 객체별 회피 반경 (m) — placement 시 비-중첩 거리 계산용. h_r보다 조금 여유.
RADII = {k: v + 0.005 for k, v in HORIZONTAL_RADIUS.items()}
# 바구니는 mesh 실제 폭(7cm)이 h_r(2.5cm)보다 훨씬 커서 닭이 너무 가까이 가면 walls와 겹침.
# placement 시점부터 충분히 여유 두기.
RADII["basket"] = 0.10

# decoration 객체는 닭/바구니와는 부딪히면 안 되지만, 자기들끼리는 살짝 겹쳐도 OK.
# → sample_positions에서 회피 반경을 절반으로 적용.
DECORATION_TYPES = {"straw_piece", "feed_pellet", "manure_pile"}

DEFAULT_BDDL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "libero", "libero", "bddl_files", "libero_object",
)
DEFAULT_ARENA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "libero", "libero", "assets", "scenes", "libero_floor_poultry_style.xml",
)

# 시각 짚이 깔릴 범위 (BDDL workspace보다 넓게 — 카메라 view 채우기)
VISUAL_STRAW_XMIN, VISUAL_STRAW_XMAX = -0.55, 0.55
VISUAL_STRAW_YMIN, VISUAL_STRAW_YMAX = -0.65, 0.65

# Panda(OnTheGroundPanda) base_xpos_offset["empty"] = (-0.6, 0, 0)
# 그리퍼가 안정적으로 도달하는 반경 (실측 기준 — Panda max reach는 ~0.85지만 자세 제약상 보수적)
ROBOT_BASE_XY = (-0.6, 0.0)
ROBOT_REACH_MAX = 0.40  # Panda max reach (보수적). 0.45는 일부 위치에서 z plateau로 실패.
ROBOT_REACH_MIN = 0.28  # Panda min reach — 너무 가까우면 arm fold 자세로 z=0.05 도달 불가.


def _in_reach(x, y):
    dx, dy = x - ROBOT_BASE_XY[0], y - ROBOT_BASE_XY[1]
    d2 = dx * dx + dy * dy
    return ROBOT_REACH_MIN * ROBOT_REACH_MIN < d2 < ROBOT_REACH_MAX * ROBOT_REACH_MAX


def sample_positions(items, max_tries=2000, seed=None):
    """items: [(obj_type, name), ...] → {name: (x, y)} 비-중첩 배치.

    - basket은 FIXED_BASKET_XY에 고정 (폐사 처리 통은 항상 같은 자리).
    - decoration끼리는 회피 반경을 절반으로 적용해 더 빽빽하게 깔리게 함.
    - basket / chick / hen / dead_chick는 robot reach 안에만 배치 (그리퍼가 잡을 수 있게).
    - decoration은 reach 밖에도 깔려서 양계장 현실감 유지."""
    rng = random.Random(seed)
    placed = {}
    placed_list = []  # (x, y, radius, is_decoration)
    # 컨트롤러가 실제로 집는 건 dead_chick뿐 → basket과 dead_chick만 reach 안에 배치.
    # 살아있는 병아리(warm)는 컨트롤러가 안 건드리므로 floor 어디든 OK (배치 쉬워짐).
    reach_required_types = {"basket", "dead_chick"}
    for obj_type, name in items:
        r = RADII.get(obj_type, 0.05)
        is_dec = obj_type in DECORATION_TYPES
        needs_reach = obj_type in reach_required_types
        # 바구니는 고정 위치 (다른 객체는 그 주변을 피함)
        if obj_type == "basket":
            x, y = FIXED_BASKET_XY
            placed[name] = (x, y)
            placed_list.append((x, y, r, False))
            continue
        for _ in range(max_tries):
            x = rng.uniform(FLOOR_XMIN + r, FLOOR_XMAX - r)
            y = rng.uniform(FLOOR_YMIN + r, FLOOR_YMAX - r)
            if needs_reach and not _in_reach(x, y):
                continue
            ok = True
            for (px, py, pr, p_is_dec) in placed_list:
                # 둘 다 decoration이면 회피 반경 절반 적용
                eff = (r + pr) * (0.5 if (is_dec and p_is_dec) else 1.0)
                if (x - px) ** 2 + (y - py) ** 2 < eff * eff:
                    ok = False
                    break
            if ok:
                placed[name] = (x, y)
                placed_list.append((x, y, r, is_dec))
                break
        else:
            raise RuntimeError(
                f"{name} 배치 실패 — floor가 너무 좁거나 객체 수가 많음"
            )
    return placed


def make_region_block(name, x, y, r):
    return (
        f"      ({name}\n"
        f"          (:target floor)\n"
        f"          (:ranges (\n"
        f"              ({x - r:.3f} {y - r:.3f} {x + r:.3f} {y + r:.3f})\n"
        f"            )\n"
        f"          )\n"
        f"      )"
    )


def build_bddl(
    n_chicks=4, n_hens=3, n_dead=2, n_baskets=1,
    n_straw=40, n_feed=30, n_manure=8,
    task_language="Pick up the dead chick and place it in the basket",
    goal_target=None, seed=None,
):
    items = []
    # 바구니를 먼저 배치 → 닭은 바구니 주변을 피해서 자리 잡음.
    # LIBERO placement도 동일 순서로 진행 → 바구니가 닭 위로 떨어지는 문제 방지.
    for i in range(1, n_baskets + 1):
        items.append(("basket", f"basket_{i}"))
    # 닭들 (큰 것부터: 성체 → 사체 → 살아있는 병아리)
    for i in range(1, n_hens + 1):
        items.append(("chicken", f"hen_{i}"))
    for i in range(1, n_dead + 1):
        items.append(("dead_chick", f"dead_chick_{i}"))
    # 살아있는 병아리는 IR warm 팔레트 (green/yellow/orange/red) 중 랜덤 선택.
    # → VLA 모델이 "파란 chick만 픽업, warm chick은 무시" 구분 학습 가능.
    chick_rng = random.Random(seed)
    for i in range(1, n_chicks + 1):
        obj_type = chick_rng.choice(WARM_CHICK_TYPES)
        items.append((obj_type, f"chick_{i}"))
    # decoration은 나중에 — 남은 공간에 깔림
    for i in range(1, n_straw + 1):
        items.append(("straw_piece", f"straw_{i}"))
    for i in range(1, n_feed + 1):
        items.append(("feed_pellet", f"feed_{i}"))
    for i in range(1, n_manure + 1):
        items.append(("manure_pile", f"manure_{i}"))

    pos = sample_positions(items, seed=seed)

    region_blocks = []
    init_states = []
    # LIBERO의 TableRegionSampler는 ensure_object_boundary_in_range=True 라서
    # [xmin+h_r, xmax-h_r] 범위에서 샘플링. 따라서 region 크기는 h_r보다 약간 크게.
    # → 결과적으로 sampler는 우리가 지정한 중심점 ±2mm 안에서 placement 함.
    EPS = 0.002
    for obj_type, name in items:
        region_name = f"{name}_init_region"
        x, y = pos[name]
        half_len = HORIZONTAL_RADIUS.get(obj_type, 0.05) + EPS
        region_blocks.append(make_region_block(region_name, x, y, half_len))
        init_states.append(f"    (On {name} floor_{region_name})")

    # BDDL parser bug 회피: 같은 type을 두 번 선언하면 마지막 줄만 남음.
    # 따라서 type별로 묶어서 한 줄에 모든 instance를 나열.
    by_type = {}
    type_order = []
    for obj_type, name in items:
        if obj_type not in by_type:
            by_type[obj_type] = []
            type_order.append(obj_type)
        by_type[obj_type].append(name)
    obj_decls = [
        f"    {' '.join(by_type[t])} - {t}" for t in type_order
    ]

    contain_region = (
        "      (contain_region\n"
        "          (:target basket_1)\n"
        "      )"
    )
    region_blocks.append(contain_region)

    # 목표: 모든 파란 병아리(dead_chick)를 바구니에 넣기.
    # goal_target가 명시되면 그 객체 하나만, 미지정이면 dead_chick 전체.
    if goal_target is None:
        if n_dead >= 1:
            goal_targets = [f"dead_chick_{i}" for i in range(1, n_dead + 1)]
        else:
            goal_targets = ["chick_1"]
    else:
        goal_targets = [goal_target]
    goal_target = goal_targets[0]  # obj_of_interest용 (대표 객체)
    goal_conjuncts = " ".join(
        f"(In {t} basket_1_contain_region)" for t in goal_targets
    )

    # problem name은 LIBERO의 기존 Libero_Floor_Manipulation 클래스와 매핑되어야 함.
    # 양계장 task 식별은 language 문자열의 "chick"/"chicken_farm" 키워드로 별도 검출.
    bddl = f"""(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {task_language})
    (:regions
{chr(10).join(region_blocks)}
    )

  (:fixtures
    floor - floor
  )

  (:objects
{chr(10).join(obj_decls)}
  )

  (:obj_of_interest
    {goal_target}
    basket_1
  )

  (:init
{chr(10).join(init_states)}
  )

  (:goal
    (And {goal_conjuncts})
  )

)
"""
    return bddl


def build_arena_xml(n_visual_straw=1000, seed=None):
    """visual-only 짚을 N개 박은 IR(적외선) 카메라 view 스타일 arena XML을 생성.

    IR 미적: 차가운 어두운 배경 (검정~짙은 남색) — 살아있는 chick(warm)와 죽은 chick(cold blue)이 색으로 대비.
    짚/사료 등 decoration은 주변 온도 (cool/cold) → 어두운 톤.
    contype=0 conaffinity=0 → 충돌 없이 순수 시각 요소 (시뮬 부하 거의 0)."""
    rng = random.Random(seed)
    straw_geoms = []
    for i in range(1, n_visual_straw + 1):
        x = rng.uniform(VISUAL_STRAW_XMIN, VISUAL_STRAW_XMAX)
        y = rng.uniform(VISUAL_STRAW_YMIN, VISUAL_STRAW_YMAX)
        # 짚을 바닥에 거의 붙이되 약간 z 변이 → 일부 겹쳐 보임
        z = rng.uniform(0.001, 0.006)
        yaw = rng.uniform(0, 6.283185)
        # 약간의 pitch tilt (대부분 0에 가깝게, 가끔 살짝 들림)
        pitch = rng.uniform(-0.12, 0.12)
        # quat 합성: pitch around Y, then yaw around Z. 단순화: euler → quat
        # capsule은 기본 Z 방향이라, X 방향으로 눕히려면 pi/2 rot around Y 필요.
        # 그 후 yaw 적용. pitch는 자연스러운 기울임용 추가.
        import math
        ry = math.pi / 2 + pitch  # capsule 눕히기 + 약간 기울임
        rz = yaw
        # quat = qz(yaw) * qy(ry)  (적용 순서: 먼저 y회전, 그 다음 z회전)
        cy2, sy2 = math.cos(ry / 2), math.sin(ry / 2)
        cz2, sz2 = math.cos(rz / 2), math.sin(rz / 2)
        # robosuite/MuJoCo quat 순서: (w, x, y, z)
        qw = cz2 * cy2
        qx = -sz2 * sy2
        qy = cz2 * sy2
        qz = sz2 * cy2
        # IR 열화상: 짚은 주변 온도(cold) → 검정. 배경과 거의 동화되도록 near-black.
        # 미세한 밝기 변이만 줘서 완전 평면이 아닌 약간의 질감 유지 (적외선 노이즈 느낌).
        v = rng.uniform(0.0, 0.045)
        r = v
        g = v
        b = v * 1.25  # 아주 살짝 푸른끼 (cold tone)
        # 길이/굵기 변이
        radius = rng.uniform(0.0020, 0.0030)
        half_len = rng.uniform(0.025, 0.040)
        straw_geoms.append(
            f'    <geom name="vstraw_{i}" type="capsule" '
            f'size="{radius:.4f} {half_len:.4f}" '
            f'pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'quat="{qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}" '
            f'rgba="{r:.3f} {g:.3f} {b:.3f} 1" '
            f'contype="0" conaffinity="0" group="1"/>'
        )

    # visual geom 버퍼는 robosuite/binding_utils.py에서 maxgeom=5000 으로 패치됨
    # 적외선(IR) 열화상 카메라 view: 검은 배경 + 검은 floor/straw, 따뜻한 개체만 빛남.
    # texplane/tex-wall texture는 EmptyArena가 존재를 요구하므로 유지하되 floor/wall은
    # flat black material을 따로 써서 진짜 검정으로 렌더.
    arena_xml = f"""<mujoco model="poultry_arena">
  <asset>
    <texture builtin="gradient" height="256" rgb1="0.0 0.0 0.0" rgb2="0.02 0.02 0.05" type="skybox" width="256"/>
    <texture file="../textures/dark_floor_texture.png" type="2d" name="texplane"/>
    <texture file="../textures/dark_blue_wall.png" type="2d" name="tex-wall"/>
    <!-- IR 검은 바닥: flat black (texture 미사용 → EmptyArena의 texplane 덮어쓰기 영향 없음) -->
    <material name="floorplane" rgba="0.0 0.0 0.0 1.0" reflectance="0.0" shininess="0.0" specular="0.0"/>
    <!-- IR 배경 벽: 거의 검정 -->
    <material name="walls_mat" rgba="0.015 0.015 0.030 1.0" reflectance="0.0" shininess="0.0" specular="0.0"/>
    <texture name="textable" builtin="flat" height="512" width="512" rgb1="0.0 0.0 0.0" rgb2="0.0 0.0 0.0"/>
    <material name="table_mat" texture="textable"/>
  </asset>
  <worldbody>
    <body name="floor" pos="0 0 0">
      <!-- contype/conaffinity=5(bit0+bit2): 일반 객체(1)+살아있는 병아리(bit2)와 모두 충돌.
           살아있는 병아리는 바닥하고만 충돌하고 죽은 병아리/바구니와는 안 부딪힘 (배회 시 안 밀어냄). -->
      <geom condim="3" group="1" material="floorplane" name="floor" pos="0 0 0" size="3 3 .125" type="plane" contype="5" conaffinity="5"/>
{chr(10).join(straw_geoms)}
    </body>
    <geom pos="-1.25 2.25 1.5" quat="0.6532815 0.6532815 0.2705981 0.2705981" size="1.06 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_leftcorner_visual" material="walls_mat"/>
    <geom pos="-1.25 -2.25 1.5" quat="0.6532815 0.6532815 -0.2705981 -0.2705981" size="1.06 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_rightcorner_visual" material="walls_mat"/>
    <geom pos="1.25 3 1.5" quat="0.7071 0.7071 0 0" size="1.75 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_left_visual" material="walls_mat"/>
    <geom pos="1.25 -3 1.5" quat="0.7071 -0.7071 0 0" size="1.75 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_right_visual" material="walls_mat"/>
    <geom pos="-2 0 1.5" quat="0.5 0.5 0.5 0.5" size="1.5 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_rear_visual" material="walls_mat"/>
    <geom pos="3 0 1.5" quat="0.5 0.5 -0.5 -0.5" size="3 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_front_visual" material="walls_mat"/>
    <!-- IR 열화상 후처리(thermal_fx) 전제 조명: 디퓨즈 셰이딩으로 개체 표면 밝기 변화를
         만들어야 컬러맵 통과 시 그라데이션이 됨 → 조명 충분히 밝게.
         검은 floor/straw는 albedo=0 이라 조명을 올려도 계속 검정 유지. -->
    <light name="light1" diffuse=".75 .75 .75" dir="0 -.15 -1" directional="false" pos="1 1 4.0" specular="0.1 0.1 0.1" castshadow="false"/>
    <light name="light2" diffuse=".75 .75 .75" dir="0 -.15 -1" directional="false" pos="-3. -3. 4.0" specular="0.1 0.1 0.1" castshadow="false"/>

    <camera mode="fixed" name="frontview" pos="1.0 0 1.45" quat="0.56 0.43 0.43 0.56"/>
    <camera mode="fixed" name="birdview" pos="-0.2 0 3.0" quat="0.7071 0 0 0.7071"/>
    <camera mode="fixed" name="topdown" pos="-0.30 0 0.70" quat="0.7071 0 0 0.7071"/>
    <camera mode="fixed" name="agentview" pos="0.5 0 1.35" quat="0.653 0.271 0.271 0.653"/>
    <camera mode="fixed" name="sideview" pos="-0.05651774593317116 1.2761224129427358 1.4879572214102434" quat="0.009905065491771751 0.006877963156909582 0.5912228352893879 0.806418094001364" />
  </worldbody>
</mujoco>
"""
    return arena_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chicks", type=int, default=8, help="살아있는 병아리 수")
    ap.add_argument("--n-hens", type=int, default=0, help="성체 닭 수 (기본 0: 병아리만)")
    ap.add_argument("--n-dead", type=int, default=2, help="사체 닭(앉아있는 병아리 모델) 수")
    ap.add_argument("--n-baskets", type=int, default=1)
    ap.add_argument("--n-straw", type=int, default=80, help="물리 짚 막대기 개수 (BDDL fixture)")
    ap.add_argument("--n-feed", type=int, default=30, help="사료 알갱이 개수")
    ap.add_argument("--n-manure", type=int, default=8, help="분뇨 덩어리 개수")
    ap.add_argument("--n-straw-visual", type=int, default=1000,
                    help="시각 전용 짚 (arena XML에 박힘, 시뮬 부하 없음)")
    ap.add_argument("--task", default="Pick up the blue chick and place it in the basket")
    ap.add_argument("--goal-target", default=None,
                    help="basket에 넣을 객체 이름. 미지정시 dead_chick_1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="chicken_farm_generated.bddl")
    ap.add_argument("--out-dir", default=DEFAULT_BDDL_DIR)
    ap.add_argument("--arena-out", default=DEFAULT_ARENA_PATH,
                    help="visual 짚이 박힌 arena XML 저장 경로")
    ap.add_argument("--no-arena", action="store_true",
                    help="arena XML 재생성 안 함 (BDDL만)")
    args = ap.parse_args()

    bddl = build_bddl(
        n_chicks=args.n_chicks,
        n_hens=args.n_hens,
        n_dead=args.n_dead,
        n_baskets=args.n_baskets,
        n_straw=args.n_straw,
        n_feed=args.n_feed,
        n_manure=args.n_manure,
        task_language=args.task,
        goal_target=args.goal_target,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out)
    with open(out_path, "w") as f:
        f.write(bddl)
    total = args.n_chicks + args.n_hens + args.n_dead
    dec_total = args.n_straw + args.n_feed + args.n_manure
    print(f"Wrote {out_path}  ({total} chickens + {args.n_baskets} basket + "
          f"{dec_total} physics decorations [straw={args.n_straw}, feed={args.n_feed}, manure={args.n_manure}])")

    if not args.no_arena:
        # arena seed는 BDDL seed와 분리 (seed 같으면 짚 위치가 BDDL 짚과 동일 패턴이 됨)
        arena_seed = (args.seed + 1) if args.seed is not None else None
        arena_xml = build_arena_xml(
            n_visual_straw=args.n_straw_visual, seed=arena_seed,
        )
        os.makedirs(os.path.dirname(args.arena_out), exist_ok=True)
        with open(args.arena_out, "w") as f:
            f.write(arena_xml)
        print(f"Wrote {args.arena_out}  ({args.n_straw_visual} visual-only straws)")


if __name__ == "__main__":
    main()
