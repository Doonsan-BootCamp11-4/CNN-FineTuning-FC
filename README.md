# CNN-FineTuning-FC

사전학습된 `ResNet18` 백본은 **100% 동결**하고 `FC` 분류 헤드만 학습해 **Garbage Classification(12클래스)** 성능을 비교하는 팀 공통 실험 파이프라인입니다.  
디스크 특징 캐시(`_feature_cache/`)와 선언형 설정만으로 CPU 환경에서도 빠르게 실험을 반복·비교할 수 있습니다.

- 원격 저장소: `https://github.com/Doonsan-BootCamp11-4/CNN-FineTuning-FC/tree/main`

---

## 1. 프로젝트 구조

```text
CNN-FineTuning-FC/
├── configs/    # 실험 정의: axes.py(단일 축), grid.py(정예 16조합), compare.py(단계별 비교)
├── fcft/       # 공개 PyTorch 파이프라인(data, model, losses, train, evaluate, kfold, report) + bin/(네이티브 가속)
├── scripts/    # run_experiments.py (통합 CLI 실행기)
├── notebooks/  # kaggle_fc_pipeline.ipynb (Kaggle CPU 노트북)
├── docs/       # 아키텍처 · CLI/옵션 레퍼런스 · 확장 가이드
└── tests/      # 회귀 테스트 4종
```

---

## 2. 핵심 실측 결과 (`garbage12` 전체 15,515장)

동일 **동결 `ResNet18` + FC 학습** 조건에서의 대표 비교 수치입니다.

| 모델 / 구성 | Pool · Head · Opt | Val Acc | Test Acc | Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
| `compare_0_baseline` (기준선) | `avg` · `linear` · `adam` | 0.9455 | 0.9366 | 0.9100 |
| `1-Shot` 선형 (`0-epoch`, `7.79s`) | `norm_multiscale` · `linear` | 0.9439 | 0.9478 | 0.9297 |
| `grid_002` (`Residual` 최고) | `norm_multiscale` · `residual` · `adamw` + `CutMix` | 0.9716 | 0.9691 | 0.9585 |
| **`grid_014` (단일 최고 Test)** | **`norm_multiscale` · `mlp` · `adamw` + `Mixup/CutMix`** | **0.9739** | **0.9723** | **0.9635** |
| **`ENS2` (`grid_013` + `grid_014`)** | **Top-2 확률 앙상블 + `multiview4` TTA** | - | **0.9729** | **0.9643** |

- **캐시 속도**: `_feature_cache/` 디스크 번들 재로드 **`0.43s`**(전체 100%), Warm 16-Grid 재탐색 **`29.40s`** (`--halving` **`23.69s`**), 캐시 Test 평가 **`0.31~0.48s`**.

---

## 3. 빠른 시작

### 1) 설치 및 테스트 확인
```bash
# Linux / macOS
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q

# Windows
py -3.12 -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest tests -q
```

### 2) 실험 실행 순서
```bash
# 1. split CSV 및 이미지 경로 점검
python scripts/run_experiments.py --check-split --check-files

# 2. 정예 16조합 Grid Search (공유 캐시 + CPU 자동 병렬)
python scripts/run_experiments.py --axis grid --shared-cache --out-dir outputs_grid

# (선택) 공용 특징 캐시 1회 내보내기 및 마운트 재사용 (Kaggle Dataset / 팀 공유)
python scripts/run_experiments.py --axis grid --export-cache fc_cache_grid.pt
python scripts/run_experiments.py --axis grid --shared-cache --cache-file fc_cache_grid.pt --out-dir outputs_grid

# 3. 원하는 기법만 골라 단일 학습 + 즉시 Test 평가 (--list-options로 선택지 확인)
python scripts/run_experiments.py --single --head mlp --mix both --loss focal_g2.0_ls --out-dir outputs_single

# 4. 단일 축 비교 (loss / activation / compare / lr / scheduler / transform)
python scripts/run_experiments.py --axis loss --shared-cache --out-dir outputs_loss
```
- **Kaggle 노트북**: [`notebooks/kaggle_fc_pipeline.ipynb`](notebooks/kaggle_fc_pipeline.ipynb)를 Kaggle에 올려 순서대로 실행하면 CPU 런타임 기준으로 표와 학습 곡선까지 한 번에 확인할 수 있습니다.

---

## 4. 파이프라인 핵심 구성 & 세부 문서

| 구분 | 기본 전략 (`configs/grid.py`) | 세부 문서 |
| :--- | :--- | :--- |
| **전처리 · 캐시** | EXIF 자동 정렬, `squash` 리사이즈, `norm_multiscale`(768D) + `3x AugBank` 디스크 캐시 | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| **정규화 · 헤드** | `Power-L2`(선택: `center_power_l2`, `center_std_power`), `residual`/`mlp`, 1-Shot 초기화 + `head_lr_scale` | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| **손실 · 학습 · 기법 설명** | `CutMix`/`Mixup`, `InvSqrt` ↔ `LogitAdj` 상호 배타 보정, `AdamW`/`Calibrated` + `WarmupCosine` + `SWA` | [`docs/EXPERIMENTS_GUIDE.md`](docs/EXPERIMENTS_GUIDE.md) |
| **평가 · 확장** | `val` 기준 탐색 후 최종 모델만 `test` 1회 평가(`flip2`/`multiview4` TTA), K-Fold CV 지원 | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) |
