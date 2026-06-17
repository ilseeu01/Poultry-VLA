# libero_chickenfarm — LIBERO 양계장 확장 번들

이 디렉토리는 표준 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)에
양계장(chicken farm) task 를 추가하기 위한 **모든 수정 파일**을 담는다.
저장소 루트의 `setup.sh` 가 이 파일들을 핀된 LIBERO 커밋
(`8f1084e`)에 자동으로 주입한다 — 수동으로 할 필요는 없다.

## 구성

| 경로 | LIBERO 내 주입 위치 | 역할 |
|---|---|---|
| `objects/hope_objects.py` | `libero/libero/envs/objects/` | 병아리 객체 클래스 등록 (`@register_object`): `dead_chick`(파란/저온 사체), `chick_warm_{green,yellow,orange,red}`(따뜻한 산닭), `chick_crouching`, `chick_looking_left`, `straw_piece`, `feed_pellet` |
| `problems/libero_floor_manipulation.py` | `libero/libero/envs/problems/` | `Libero_Floor_Manipulation` 문제 정의. BDDL 경로/내용에 `poultry`/`chicken_farm` 포함 시 흙바닥 양계장 씬 자동 적용 |
| `assets/scenes/libero_floor_poultry_style.xml` | `libero/libero/assets/scenes/` | 양계장 바닥 아레나 씬 |
| `assets/stable_hope_objects/*` | `libero/libero/assets/stable_hope_objects/` | 병아리 메시(STL)·머티리얼·물성 XML. `chick_warm_*` 의 STL 은 `../chick_crouching/Chick_crouching.stl` 로의 **상대 심링크**(메시 공유) |
| `assets/stable_scanned_objects/basket` | `libero/libero/assets/stable_scanned_objects/` | 사체 수거 바구니 |
| `bddl/pick_up_the_blue_chick_and_place_it_in_the_basket.bddl` | `libero/libero/bddl_files/libero_object/` | 기준 task BDDL |

씬 생성기 `generate_chicken_farm_bddl.py` 는 저장소의 `data_pipeline/` 에 있으며,
`setup.sh` 가 `<LIBERO>/scripts/` 로 복사한다(기본 출력 경로가 LIBERO 트리를
기준으로 계산되기 때문).

## dead_chick 물성 변형 (논문 ablation)

`assets/stable_hope_objects/dead_chick/` 에는 3가지 물성이 있다:

| 파일 | 마찰 / 관성 / 캡슐 | full-task 성공률 |
|---|---|---|
| `dead_chick.xml` (현재) | 6.0 / 4e-4 / 0.018 (**firm + thick**) | **94%** |
| `dead_chick.xml.affbak` | 2.0 / 5e-5 / 0.013 (slippery, affordance) | 56% |
| `dead_chick.xml.bak` | 원본 | — |

기본값(`dead_chick.xml`)이 최종 94% 결과에 쓰인 firm+thick 물성이다.
`results/FINAL_RESULTS.md` 의 객체 물리 ablation 참조.

## robosuite 패치

양계장 씬은 짚(straw) 비주얼 geom 1000개를 렌더하므로 robosuite 기본
`MjvScene maxgeom=1000` 으로는 버퍼가 부족하다. `setup.sh` 가
`patches/apply_robosuite_patch.py` 로 `maxgeom=5000` 으로 올린다.
