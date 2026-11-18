# ResNet18 FC 전이 학습 설계

## 목표

ImageNet 사전학습 ResNet18의 백본을 동결하고 FC 레이어만 학습해 Garbage Classification 성능을 낸다.
학습률, 스케줄러, 전처리 실험으로 성능을 최대한 올린 뒤, 최종 test 성능을 다른 팀의 partial fine-tuning 결과와 비교한다.

범위 밖: partial/full fine-tuning (다른 팀 담당), 다른 백본 비교.

## 데이터

- Kaggle Garbage Classification 12클래스 (mostafaabla/garbage-classification, 약 15.5k장)
- split: train/val/test = 60/20/20, 클래스 비율 유지(stratified), seed 42
- split은 별도 공용 저장소 `CNN-FineTuning-Dataset-Split/`에서 만들고 다른 팀과 공유한다.
  - `make_split.py`: 이미지 경로를 정렬한 뒤 분할 → `splits/garbage12_602020_seed42.csv`
  - CSV 컬럼: `path`(데이터셋 루트 기준 상대 경로), `label`, `split`
  - 두 팀은 코드가 아니라 커밋된 CSV를 그대로 읽는다 (환경 차이로 split이 달라지는 것 방지).

## 저장소 구조

```
CNN-FineTuning-FC/
├─ configs/
│  ├─ base.py          # 공통 기본값 BASE (dict)
│  ├─ lr.py            # EXPERIMENTS: lr / optimizer 실험 목록
│  ├─ scheduler.py     # EXPERIMENTS: 스케줄러 실험 목록
│  └─ transform.py     # EXPERIMENTS: 전처리 실험 목록
├─ fcft/
│  ├─ __init__.py
│  ├─ data.py          # split CSV → Dataset, transform 프리셋, DataLoader
│  ├─ model.py         # ResNet18 로드, 백본 동결, FC(512→12) 교체
│  ├─ train.py         # run(config)
│  └─ evaluate.py      # 지표, confusion matrix, 학습 곡선
├─ outputs/            # <실험이름>/history.json, metrics.json, 그래프 (체크포인트는 gitignore)
└─ tests/
   ├─ test_data.py
   └─ test_model.py
```

## 인터페이스

- config는 dict이고, 모두가 같이 쓰는 값만 담는다. 각 실험은 `{**BASE, "name": ..., <바꿀 값>}` 형태.
  - 실행 환경: `split_csv`, `data_root`, `out_dir`, `num_workers` (Kaggle 경로가 기본값, 로컬에서는 덮어쓴다)
  - 모델: `model="resnet18"`, `num_classes=12` (지금은 resnet18만 지원, 키는 나중을 위해 둔다)
  - 학습: `epochs=15`, `batch_size=32`, `seed=42`
- optimizer / scheduler / train transform은 config에 넣지 않는다. 각자 구현해 `run`의 인자로 넘긴다.
  - 접근 방식 검토: (1) `run`의 인자로 넘기기 — 채택. 학습 루프·저장·기록 형식이 같아 결과를 모으기 쉽다.
    (2) 부품만 공통으로 두고 루프는 각자 작성 — 자유도는 높지만 결과 파일 형식이 사람마다 달라진다.
    (3) config에 함수/문자열로 넣기 — 개인 구현이 공통 config에 섞인다.
- `fcft.train.run(config, optimizer=None, scheduler=None, train_transform=None) -> dict`
  - `optimizer`: 학습할 파라미터 → Optimizer 를 만드는 함수. None이면 Adam(lr=1e-3).
  - `scheduler`: Optimizer → scheduler 를 만드는 함수. None이면 쓰지 않는다.
  - `train_transform`: torchvision transform. None이면 val/test와 같은 전처리.
  - 학습 후 val 정확도 기준 best 모델을 저장하고, `outputs/<name>/history.json`(epoch별 train/val loss·acc, lr)을 남긴다. test는 평가하지 않는다.
- `fcft.evaluate`: 저장된 모델로 test 평가 → `metrics.json`(accuracy, 클래스별 precision/recall/F1, confusion matrix), 학습 곡선·confusion matrix 그림.
- `fcft.model.build_model(num_classes=12)`: FC 외 모든 파라미터 `requires_grad=False`.
- `fcft.data`: train transform은 `run`에서 받은 것을 쓴다. val/test transform은 고정(224 resize + ImageNet normalize).

## 구현 순서

한 번에 만들지 않고 하나씩 만들고 확인한다: 로컬 환경(venv, `requirements.txt`, README) → `configs/base.py` → `fcft/model.py` → `fcft/data.py` → `fcft/train.py` → `fcft/evaluate.py`.

## 실행 환경

로컬은 코드가 끝까지 도는지 확인하는 용도, 실제 학습은 Kaggle에서 한다.

### 로컬 (테스트)

- 데이터셋은 받지 않는다.
- venv: `py -3.12 -m venv .venv` (`.venv`는 이미 gitignore). 팀원 모두 같은 `requirements.txt`로 설치한다.
- `requirements.txt`: torch, torchvision(CPU 빌드), pillow, scikit-learn, matplotlib, pytest. 로컬 전용이며 Kaggle에서는 설치하지 않는다(이미 깔려 있음).
- 테스트는 임시 폴더에 가짜 이미지 10장 정도와 split CSV를 만들어, 데이터 로딩 → 학습 1 epoch → 평가가 끝까지 도는지만 본다.
- 모델은 실제와 같이 ImageNet 사전학습 가중치를 쓴다(최초 1회 약 45MB 다운로드 후 캐시). 테스트용 분기는 두지 않는다.
- README에 venv 생성 → 설치 → `python -m pytest tests -q` 순서를 적는다.

### Kaggle (학습)

Kaggle 노트북(GPU). 각 팀원은 공유받은 노트북을 "Copy & Edit"으로 자기 계정에 복사해 실행한다 (GPU 시간은 실행한 계정 기준).
노트북 셀 순서: 저장소 clone/pull → `sys.path`에 저장소 루트 추가 → config 선택 → `run()`.

## 진행 단계와 역할

1. **공통 파이프라인 (3인 함께)**: `fcft/` 파일을 나눠 작성하고 서로 리뷰한다. 기본 config로 1회 학습이 끝까지 돌면 노트북을 확정해 공유한다.
2. **축별 실험 (1인 1축)**
   - A: 학습률 / optimizer (`configs/lr.py`)
   - B: 스케줄러 (`configs/scheduler.py`)
   - C: 전처리 / augmentation (`configs/transform.py`)
   - 각 파일에는 자기가 구현한 optimizer / scheduler / transform과 실험 목록을 둔다.
   - 규칙: 다른 축은 `BASE` 값 고정, seed·epoch 동일, 비교는 val 성능으로만 한다.
   - 각자 자기 축의 표와 학습 곡선을 보고서 한 섹션으로 쓴다.
3. **최종 결과 (3인 함께)**: 각 축의 best 설정을 합쳐 최종 모델을 학습하고, test를 이때 한 번만 평가한다. 다른 팀 결과와 비교해 발표자료/보고서를 만든다.

## 결과물

- 수치 표: 실험별 val 정확도, 최종 test accuracy와 클래스별 F1
- 그래프: 학습 곡선, confusion matrix
- 발표자료/보고서

## 테스트

- 실제 데이터 없이 가짜 이미지로 돈다 (split 자체의 검증은 `CNN-FineTuning-Dataset-Split` 저장소 몫).
- `test_data.py`: 가짜 split CSV로 train/val/test 로더가 만들어지고 개수·shape가 맞는지, 데이터 로딩 → 학습 1 epoch → 평가가 끝까지 도는지
- `test_model.py`: FC 외 파라미터 동결, 출력 shape `(N, 12)`
- 실행: `python -m pytest tests -q`
