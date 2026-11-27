# 실험 실행 & 옵션 가이드 (`EXPERIMENTS_GUIDE`)

단일 CLI([`scripts/run_experiments.py`](../scripts/run_experiments.py))로 실험 축 비교, Grid 탐색, 산출물 정리까지 실행하는 실전 안내서입니다.

---

## 1. 주요 CLI 플래그

```bash
python scripts/run_experiments.py [옵션]
```

| 플래그 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--check-split` / `--check-files` | `False` | 학습 없이 `split.csv` 집계 및 이미지 파일 누락 검사 (`--expected-classes 12`) |
| `--axis` | `"all"` | 실험 축: `all`, `lr`, `scheduler`, `transform`, `loss`, `activation`, `compare`, `grid` |
| `--single` / `--list-options` | `False` | 원하는 기법만 선택해 **단일 학습 + 즉시 Test 평가** (`--no-test`로 평가 생략) / 옵션 목록 출력 |
| `--head`·`--pool`·`--mix`·`--loss`·`--opt` 등 | `DEFAULTS` | `--single`용 직관적 기법 선택 플래그 (`--set key=val`로 기타 옵션 직접 지정 가능) |
| `--split-csv` / `--data-root` | `BASE` 설정 | 데이터 분할 CSV 및 이미지 루트 경로 재정의 |
| `--out-dir` | `BASE["out_dir"]` | 체크포인트(`best.pt`), 로그, 요약표(`summary.md`), 그래프 저장 폴더 |
| `--epochs` / `--batch-size` | `15` / `32` | 에폭 수 · 배치 크기 재정의 (`--num-workers`로 DataLoader 워커 지정) |
| `--shared-cache` / `--fast-cpu` | `False` | **권장**: 동결 백본 특징을 `_feature_cache/`에 캐싱해 다중 실험 가속 (`--no-disk-cache`로 비활성화) |
| `--cache-file` / `--export-cache` | `None` | 공용 `.pt` 캐시 파일 마운트(읽기 전용 Kaggle Input 지원) / 팀 공유용 `.pt` 번들 1회 추출 |
| `--workers` | `None` (자동) | CPU 병렬 워커 수 지정 (예: `--workers 4`) |
| `--co-train` | `False` | 동일 피처·구조의 다중 헤드를 `torch.bmm`으로 동시 학습 |
| `--halving` / `--halving-factor` | `False` / `3` | 초반 평가 후 하위 조합을 조기 탈락(Successive Halving) |
| `--k-folds` | `1` | `2` 이상 지정 시 `test`를 격리한 계층형 K-Fold CV 실행 |
| `--limit` / `--max-runs` | `None` | 실행 실험 개수 제한 (`--max-runs`는 Grid 셔플 부분 샘플) |
| `--no-skip` / `--no-summarize` | `False` | 완료 실험 건너뛰기 해제 / 종료 후 자동 요약 생략 |

---

## 2. `--axis` 비교 축 요약

| `--axis` | 실험 수 | 비교 대상 |
| :--- | :---: | :--- |
| `lr` / `scheduler` / `transform` | 16 / 6 / 6 | 옵티마이저·LR, 스케줄러, 이미지 증강 프리셋 (`--axis all`로 28개 일괄 실행) |
| `loss` / `activation` | 12 / 7 | 손실함수(CE·LS·Focal·ClassWeight·LogitAdj·Resilient), 활성화함수 7종 |
| `compare` | 8 | 무규제 기준선(`compare_0_baseline`) 대비 단계별 개선 효과 비교 |
| `grid` | 16 | [`configs/grid.py`](../configs/grid.py) `GRID` 기반 핵심 조합 탐색 |

---

## 3. `configs/grid.py` 주요 설정 키

`configs/grid.py`의 `GRID` 딕셔너리 값을 바꾸면 코드 수정 없이 조합을 확장할 수 있습니다. *(전체 후보 목록은 같은 파일의 `SUPPORTED_OPTIONS` 참조)*

| 키 | 대표 지원 값 | 비고 |
| :--- | :--- | :--- |
| `pool_mode` | `"norm_multiscale"`, `"avg"` *(확장: `avgmax`, `gem`, `multiscale`)* | `norm_multiscale` = `layer3`+`layer4` 결합(768D) |
| `head_type` | `"residual"`, `"mlp"`, `"linear"` *(확장: `cosine`, `subcenter`, `dual_anchor`)* | `residual`·`mlp` 기본 권장 |
| `activation` | `"smooth"`, `"smooth_learn"`, `"relu"`, `"leaky_relu"`, `"elu"`, `"gelu"`, `"silu"`, `"mish"` | 비선형 헤드 활성화 |
| `feature_norm_mode` | `"power_l2"`, `"center_power_l2"`, `"center_std"`, `"center_std_power"`, `"l2"` | `center_*`는 train 통계만 사용 |
| `loss_mode` | `"ce"`, `"ce_ls0.05"`, `"focal_g2.0_ls"`, `"cw_inv_sqrt"`, `"cw_effective"`, `"logit_adj_tau0.5"`, `"resilient_*"` | `configs/axes.py` 프리셋 연동 |
| `optimizer_mode` / `scheduler` | `"adamw"`, `"calibrated"`, `"sgd"` / `"warmup_cosine"`, `"cosine"`, `"step"` | `head_lr_scale`로 선형·어댑터 LR 분리 가능 |
| `mix_mode` / `aug_mode` | `"none"`, `"mixup"`, `"cutmix"`, `"both"` / `"rrc_jitter"` 등 6종 | `use_aug_bank=True` 시 3배 증강 캐시 사용 |
| `class_weight_mode` / `swa_epochs` | `None`(자동 `logit_adj_tau=0.5`), `"inv_sqrt"`, `"effective"` / `0`, `4` | 중복 보정 방지 규칙 적용 |
| `feature_noise_std` / `ckpt_metric` | `0.0`, `0.02` / `"val_acc"`, `"macro_f1"` | `feature_noise_std`는 고정 `AugBank` 과적합 완화, `macro_f1`은 불균형 체크포인트 기준 |
| `model` | `"resnet18"`, `"resnet34"`, `"resnet50"` | 기본 `"resnet18"` 외 백본 확장 선택 가능 (`pool_mode` 자동 정합) |

---

## 3.5 각 기법 쉽게 이해하기 (언제 · 왜 쓰나요?)

| 분류 | 기법명 | 쉬운 설명 (핵심 역할) |
| :--- | :--- | :--- |
| **특징 추출** (`pool_mode`) | `avg` / `norm_multiscale` | `avg`는 마지막 층(512D) 의미만 추출하고, `norm_multiscale`은 중간층 **질감·윤곽**(256D)과 마지막층 **의미**(512D)를 균형 정규화해 합친 768D 특징이라 재질 구분에 유리합니다. *(그 외 `avgmax`, `gem` 지원)* |
| **특징 정규화** (`feature_norm_mode`) | `power_l2` / `center_*` | `power_l2`는 유독 튀는 채널 값을 $\operatorname{sign}(x)\sqrt{|x|}$로 눌러 길이를 1로 맞춥니다. `center_power_l2`·`center_std_power`는 학습셋 평균/표준편차까지 정렬해 소규모 데이터에서 안정적입니다. |
| **분류 헤드** (`head_type`) | `linear` / `residual` / `mlp` | `linear`는 단순 선형층(빠른 기준선), `residual`은 `1-Shot` 선형 경계 위에 작은 비선형 어댑터를 얹어 오차만 보정, `mlp`는 2층 비선형 구조로 충분히 학습할 때 최고 정확도를 냅니다. *(코사인 계열: `cosine`, `subcenter`, `dual_anchor`)* |
| **초기화 · 2속도 LR** | `warm_start_ridge` / `head_lr_scale` | 역전파 전 닫힌 해(`1-Shot` Ridge)로 출발점을 먼저 잡고, `head_lr_scale`(예: `0.2`)로 기존 선형 경계는 천천히, 새 잔차 어댑터만 빠르게 학습시킬 수 있습니다. |
| **활성화함수** (`activation`) | `relu` / `gelu`·`silu` / `smooth` | `relu`는 0 이하를 차단하고, `gelu`·`silu`·`mish`는 완만한 곡선으로 기울기를 살립니다. `smooth`는 곡률 연속 완만 활성화(`smooth_learn`은 누수 비율 자동 학습)입니다. |
| **리사이즈 · 증강** | `squash` / `AugBank` / `Mix` | `squash`는 외곽 단서를 자르지 않고 224×224로 맞춥니다. `AugBank`는 3배 증강 뷰를 미리 캐싱하고, `Mixup`·`CutMix`는 두 샘플 특징과 정답을 섞어 과적합을 억제합니다. (`TTA`: 테스트 시 반전·확대 뷰 예측 평균) |
| **손실 · 불균형** (`loss_mode`) | `ce_ls` / `focal` / `cw_*` / `logit_adj` / `resilient_*` | `ce_ls`는 과신을 줄이고, `focal`은 어려운 샘플에 집중합니다. `cw_*`(클래스 가중치)와 `logit_adj`(사전확률 보정)는 소수 클래스를 돕고, `resilient_*`는 극단 이상치 손실 폭주를 억제합니다. |
| **옵티마이저 · SWA** | `adamw` / `calibrated` / `swa_epochs` | `adamw`는 범용 표준, `calibrated`는 시간축 곡률 기반 보정 옵티마이저입니다. `SWA`는 학습 후반 가중치를 평균 내 테스트 일반화를 높입니다. |

---

## 4. 산출물 & 최종 1회 Test 평가

실행이 끝나면 `--out-dir` 아래에 실험별 `best.pt`·`history.json`과 전체 요약(`summary.md`, `summary.csv`, `comparison_<axis>.png`, `*_comparison.md`)이 자동 생성됩니다.

### 직접 조합 선택 후 단일 학습·평가 (Python API)

```python
from fcft import run_single_experiment

res = run_single_experiment({
    "head_type": "mlp", "pool_mode": "norm_multiscale", "mix_mode": "both",
    "loss_mode": "focal_g2.0_ls", "activation": "gelu", "out_dir": "outputs_single",
})
print("Val:", res["best_val_acc"], "| Test:", res["test_accuracy"], "| F1:", res["test_macro_f1"])
```

### Grid 탐색 후 최종 1회 Test 평가

탐색은 항상 `val`로만 순위를 매기고, 최종 보고 직전에만 캐시된 `test` 번들로 1회 평가합니다:

```python
from fcft import evaluate, evaluate_ensemble, extract_shared_feature_bundle, collect_summaries

base_cfg = {"split_csv": "../CNN-FineTuning-Dataset-Split/splits/garbage12_602020_seed42.csv", "data_root": "/path/to/garbage_classification", "out_dir": "outputs_grid", "num_classes": 12}
bundle = extract_shared_feature_bundle(base_cfg, include_flip=True, include_test=True, use_disk_cache=True)
top2 = collect_summaries("outputs_grid")[:2]

m_best = evaluate({**base_cfg, "name": top2[0]["name"]}, tta_mode="flip2", precomputed_test=bundle)
m_ens = evaluate_ensemble([{**base_cfg, "name": r["name"]} for r in top2], out_name="top2_ensemble", tta_mode="multiview4", temperature=1.1, precomputed_test=bundle)
print("Best Test:", m_best["accuracy"], "| Top-2 Ensemble:", m_ens["accuracy"])
```
