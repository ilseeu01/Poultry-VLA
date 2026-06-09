# 🐤 Poultry-VLA

IR 열화상 환경에서 **OpenVLA-7B**로 "살아있는(따뜻한) 병아리들 사이에서 죽은(차가운/파란) 병아리를 골라 바구니에 담기"를
학습한 Vision-Language-Action 프로젝트. (LIBERO / robosuite 시뮬레이터)

> **종합 보고서: [`index.html`](index.html)** 를 브라우저로 열어보세요. (발표용, 전체 내용·수치·실패원인·해결책 포함)

---

## 한 줄 요약

데이터·평가 파이프라인은 정상이며 모델은 죽은 병아리를 **1mm까지 정확히** 찾아간다.
최종 실패(파지 0%)는 데이터 부족이 **아니라**, OpenVLA가 메모리·자기상태(proprioception) 없는 단일프레임 정책이라
**"하강 단계 vs 들기 단계"를 구분하지 못해** 병아리 위에서 평균(제자리)을 예측하는 **구조적 한계**다.
→ **데이터 재구축 불필요. proprioception + action chunking(OpenVLA-OFT)로 학습 레시피만 교체하면 된다.**

## 실패 원인 (4겹, 앞 3겹 해결 / 마지막 1겹 구조적)

| # | 원인 | 상태 |
|---|------|------|
| 1 | 이미지 좌우 반전 (수집 `[::-1]` vs 평가 `[::-1,::-1]`) | ✅ 수정 |
| 2 | thermal 해상도 불일치 (학습 256 / 평가 224) | ✅ 수정 |
| 3 | 평가 셋업 OOD (고정 BDDL + 정지 병아리) → 위치오차 8cm | ✅ 평가 일치로 1mm |
| 4 | **파지 단계 모호성** (메모리·proprio 없는 단일스텝) | ❌ 미해결(구조적) |

**근거(데이터):** 병아리 2cm 이내 같은 위치에서 그리퍼 열림→z −0.30(하강 99.9%), 닫힘→z +0.33(상승 73.5%).
이미지는 거의 동일 → 모델이 단계를 못 구분해 평균(제자리) 예측.

## 코드 구조

```
data_pipeline/   장면 생성 · 데모 수집(6워커) · 컨트롤러 · IR thermal
rlds/            HDF5→RLDS + no-op 필터
training/        finetune.py(어댑터 스냅샷) · train_v3.sh · merge_adapter.py
eval/            run_libero_eval.py(수정) · eval_trainscene.py(학습분포 평가) · eval_ckpt.sh
diagnostics/     diag_*.py 8종 (in-dist예측 · 좌우반전 · 해상도 · 폐루프 · 파지Z · 그리퍼추적 등)
docs/            FAILURE_ANALYSIS.md (상세 분석)
```

## 다음 단계 (성공 경로)

1. **proprioception 입력 추가** (그리퍼 상태 + eef 높이) — 단계 모호성 원리적 해소
2. **action chunking** (미래 K스텝 예측) — 드문 파지 결정 commit
3. **OpenVLA-OFT 재학습** (1+2 + 연속 L1 액션헤드 통합)

데이터(204 데모 + RLDS)·평가·진단은 완성되어 재사용 가능. 남은 건 학습 레시피 교체(약 1~3일).

> 실행에는 [OpenVLA](https://github.com/openvla/openvla)·[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 환경이 필요.
> `training/finetune.py`, `eval/run_libero_eval.py`는 OpenVLA 원본을 본 과제에 맞게 수정한 버전.
