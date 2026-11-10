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

- config는 dict. 각 실험은 `{**BASE, "name": ..., <바꿀 값>}` 형태.
  - 키: `name`, `lr`, `optimizer`, `scheduler`, `transform`, `epochs`, `batch_size`, `seed`
  - `BASE` 초기값: `lr=1e-3`, `optimizer="adam"`, `scheduler=None`, `transform="basic"`(224 resize + normalize), `epochs=15`, `batch_size=32`, `seed=42`. 1단계 결과를 보고 조정할 수 있다.
- `fcft.train.run(config) -> dict`: 학습 후 val 정확도 기준 best 모델을 저장하고, `outputs/<name>/history.json`(epoch별 train/val loss·acc, lr)을 남긴다. test는 평가하지 않는다.
- `fcft.evaluate`: 저장된 모델로 test 평가 → `metrics.json`(accuracy, 클래스별 precision/recall/F1, confusion matrix), 학습 곡선·confusion matrix 그림.
- `fcft.model.build_model(num_classes=12)`: FC 외 모든 파라미터 `requires_grad=False`.
- `fcft.data`: transform 프리셋 이름(문자열)으로 train transform을 고른다. val/test transform은 고정(resize + ImageNet normalize).

## 실행 환경

Kaggle 노트북(GPU). 각 팀원은 공유받은 노트북을 "Copy & Edit"으로 자기 계정에 복사해 실행한다 (GPU 시간은 실행한 계정 기준).
노트북 셀 순서: 저장소 clone/pull → `sys.path`에 저장소 루트 추가 → config 선택 → `run()`.

## 진행 단계와 역할

1. **공통 파이프라인 (3인 함께)**: `fcft/` 파일을 나눠 작성하고 서로 리뷰한다. 기본 config로 1회 학습이 끝까지 돌면 노트북을 확정해 공유한다.
2. **축별 실험 (1인 1축)**
   - A: 학습률 / optimizer (`configs/lr.py`)
   - B: 스케줄러 (`configs/scheduler.py`)
   - C: 전처리 / augmentation (`configs/transform.py`, `fcft/data.py`의 transform 프리셋)
   - 규칙: 다른 축은 `BASE` 값 고정, seed·epoch 동일, 비교는 val 성능으로만 한다.
   - 각자 자기 축의 표와 학습 곡선을 보고서 한 섹션으로 쓴다.
3. **최종 결과 (3인 함께)**: 각 축의 best 설정을 합쳐 최종 모델을 학습하고, test를 이때 한 번만 평가한다. 다른 팀 결과와 비교해 발표자료/보고서를 만든다.

## 결과물

- 수치 표: 실험별 val 정확도, 최종 test accuracy와 클래스별 F1
- 그래프: 학습 곡선, confusion matrix
- 발표자료/보고서

## 테스트

- `test_data.py`: train/val/test 간 경로 중복 없음, 클래스 수 12, 비율 약 60/20/20
- `test_model.py`: FC 외 파라미터 동결, 출력 shape `(N, 12)`
- 실행: `python -m pytest tests -q`
