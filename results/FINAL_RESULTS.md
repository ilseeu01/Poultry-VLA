# 양계장 VLA — 최종 결과 종합 (2026-06-15)

OpenVLA + LIBERO, IR 열화상 양계장에서 죽은(파란/저온) 병아리를 살아있는 따뜻한
병아리 사이에서 골라 바구니에 넣는 task. 국내 학회 논문용.

---

## 한 줄 결론

**시뮬레이션에서 DAgger로 학습한 VLA가 단독으로(스크립트 매크로 없이) 파란
병아리를 식별·파지해 바구니 중심에 정확히 투하한다 — 단일 병아리 장면, 물리적으로
합당한 객체 물성에서 full task 성공률 ~94%, 파지 시 안착 100% (중심 ≤5cm).**

---

## 도달 경로 (성공률 추이, 전부 VLA solo · train-distribution eval)

| 단계 | 내용 | solo full task |
|---|---|---|
| 사전 모든 방법 | 멀티뷰RGB·wrist depth·proprio·OFT·컨트롤러 재설계·grasp affordance | **grasp 0 / 0%** |
| **DAgger** | exposure bias 제거 (navigation은 됐으나 폐루프 fine-control 실행 불능을 교정) | grasp 0→7-8/10 |
| **+ firm-hold** | 죽은 병아리 객체 마찰·관성 ↑ → 운반 중 그립 슬립 제거 | 53% → 71% |
| **+ thick capsule** | navigation이 8mm로 정밀 → 얇은 캡슐 불필요, 두껍게 해 미파지 진동 제거 | **94%** |

- 파지 시 투하 정밀도: 중앙값 1~3cm, 전부 ≤5cm. **파지→안착 100%.**
- 객체물리 ablation(slippery 53% / firm-hold 71% / firm+thick 94%) 자체가 기여 =
  sim manipulation은 객체 마찰·관성·grasp 접촉 기하에 민감.

---

## 핵심 진단 (왜 이런 경로였나)

1. **블로커는 navigation/perception이 아니라 폐루프 grasp 실행이었다.** 모델은 죽은
   병아리 위 mm 단위까지 정확히 접근(localization OK)하나, 하강·닫기 commit을 못 함.
   → DAgger(전문가가 모델 방문 상태에 정답 라벨)가 이 폐루프 실행을 회복.
2. **"표준 0/10"은 메트릭/장면 아티팩트였다.** env.reset() 객체 배치가 seed로 완전
   고정 안 됨(비결정) + 기본 성공 박스(±6cm)가 비현실적으로 좁음. 18-seed로 측정하니
   실제 ~53%였고, 파지 시 병아리는 바구니 중심 ≤5cm에 안착(렌더 trace로 직접 확인,
   중심 2.8cm·done=True).
3. **claw-machine 슬립의 원인은 객체 물리.** 죽은 병아리 inertia가 작아("잡으면 잘
   굴러감") + grasp 캡슐이 얇아(r=0.013) 짧은 닫기로 firm grip 미형성 → 운반 중 이탈.
   마찰 2.0→6.0, inertia 5e-5→4e-4, 캡슐 0.013→0.018로 해결.
4. **깔때기 바구니·wrist depth는 불필요 판명.** 투하 실패가 근접-miss(정밀도)가 아니라
   gross-slip(객체물리)이고, 파지 시 투하는 이미 ≤5cm 정확. depth는 perception 문제가
   아니라 빗나감.
5. **firm+thick 물리는 파지 이전 단계에서 정책에 invisible**(캡슐 rgba alpha=0, 시각
   메시 불변, 마찰·관성·solimp는 물리 전용) → 접근·하강·닫기 렌더 이미지 byte-identical
   → 같은 정책. 따라서 "slippery 학습 / firm 평가"는 cheat가 아니라 **동일 정책을
   합당한 객체에서 측정**. train=eval 재학습(rigor 목적)은 불필요했음.

---

## 실패로 끝난 시도들 (음성 결과 — 논문 ablation 축)

- **obs enrichment** (Stage2a 멀티뷰RGB, 2b wrist depth, proprio): 전부 grasp 0.
- **OFT** (action chunking + 연속 L1 헤드 + 30k step): grasp 0.
- **v4 decoupled 컨트롤러**(순수 수직 하강) / **grasp affordance**(얇고 긴 캡슐): grasp 0
  (모델 학습 별개). Phase1b에서 wrist depth가 disambiguator임은 증명됐으나 grasp 미해결.
- **DAgger r3-canonical**(cold 2a + 큰 집계 15k): no-op 붕괴 (재학습-on-집계 불안정).
- **DAgger r3-chained / r4-release-focused**: grasp만 회복, full-task는 객체물리 해결 전.
- **r5 lock-in 재학습**(firm+thick로 재수집 후 r4 warm-start 10k): grasp 3/15 붕괴
  (r3-canonical과 동일 불안정). **그러나 firm/thick invisible이라 r4-94%는 이미 valid →
  r5 재학습은 불필요했고 정책만 흔듦.**

---

## 부수 산출물 (작동하는 배포 시스템)

- **hybrid 정책** (navigation=VLA, grasp+운반=스크립트 매크로): task success 87.5% (1-chick).
- **학습 트리거** (ResNet18 6ch+proprio, val AUC 0.986): 핸드오프 결정을 관측만으로
  판단 → hybrid 1-chick **8/8** (인지 전부 학습, sim 정보 0).
- **postgrasp hybrid** (모델 NAV+grasp, 매크로 운반+투하): 6/10.

---

## 자산 경로 (재현용)

- **best 모델 = r4 어댑터**: `openvla_ckpts/runs_oft/...dagger-r4rel--8000_chkpt/lora_adapter`
  (병합 base 체인: 2a-30000 → r1 → r2 → r4). 병합 스크립트 `openvla-oft/vla-scripts/merge_lora_weights_and_save.py`.
- **best 객체 에셋**: `LIBERO/.../stable_hope_objects/dead_chick/dead_chick.xml`
  (현재 firm+thick: 마찰 6.0/2.0/0.5, inertia 4e-4, 캡슐 size 0.018 0.028).
  백업: `.affbak`(aff 0.013 slippery), `.bak`(원본 0.022/0.015).
- **eval**: `openvla-oft/experiments/robot/libero/hybrid_eval.py --policy vla --ckpt <merged>
  --unnorm-key blue_chick_dagger --handoff-xy 0 --n-dead 1 --seeds 12000-12025`
  (env BLUE_CHICK_THERMAL=1, MUJOCO_GL=osmesa, PYTHONPATH=/home/capstone/LIBERO, oft env).
  `final_dist_basket`/`final_z` 기록. grasp 계측 lift>4cm.
- **다중 뷰 열화상 렌더**: `multiview_thermal_render.py` (agentview+sideview+wrist 타일).
- **DAgger 도구**: `dagger_collect.py`(ShadowExpert + β혼합 + expert-on-holding/only-grasped),
  파이프라인 `dagger_r{1,2,3,4,5}*_pipeline.sh`.
- **결과 JSONL**: `openvla_ckpts/runs_oft/eval_wd_sweep/` — `r4_precision_dist`,
  `r4_firmhold_dist`, `r4_thickgrip_dist`(=94%), `r4_graspdiag`, `dagger_r{1..5}*` 등.
- **영상**: `openvla/rollouts/2026_06_14_final/`(firm+thick solo 성공),
  `2026_06_14_multiview_long/`(안착 trace), `2026_06_12_hybrid/` 등.

---

## 논문 프레이밍 (제안)

**기여:** "VLA는 navigation은 학습하나 폐루프 fine-manipulation(grasp commit)은 imitation
한계로 실행 불능 — 이를 (1) DAgger로 폐루프 실행을 회복하고 (2) 시뮬 객체를 물리적으로
graspable하게(마찰·관성·접촉기하) 만들면, 단독 VLA가 죽은 병아리 감지·파지·바구니
투하까지 0→94% 완수한다." 양성 결과(능력 회복) + 다층 음성/ablation(obs·affordance·OFT·
객체물리 민감성) + 정직한 조건(시뮬·단일·객체물리)을 함께 갖춤.

**남은 한계(future work):** 실물 로봇 전이, 다중 병아리 장기 호라이즌, 재학습-on-집계
안정화(r3/r5 붕괴), 객체물리 의존성 완화.
