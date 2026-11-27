# 확장 및 테스트 가이드 (`CONTRIBUTING`)

새 전처리·헤드·손실함수·비교 축을 안전하게 추가하기 위한 최소 규칙과 테스트 방법입니다.

---

## 1. 새 실험 요소 추가 위치

| 추가 대상 | 수정 위치 | 핵심 주의사항 |
| :--- | :--- | :--- |
| **증강 프리셋 (`aug_mode`)** | [`fcft/data.py`](../fcft/data.py) `build_aug_transform()` | `ToTensor()` + `Normalize(IMAGENET_MEAN, IMAGENET_STD)`, `(3, 224, 224)` 유지 |
| **헤드 / 정규화 레이어** | [`fcft/model.py`](../fcft/model.py) `build_head()` | `train` 통계가 필요하면 `fit_stats()` 구현 후 `fcft/train.py` `_fit_head_center_stats()` 연결 |
| **손실 / 활성화 프리셋** | [`fcft/grid.py`](../fcft/grid.py) `LOSS_PRESETS`, [`configs/axes.py`](../configs/axes.py) | 활성화는 [`fcft/accel.py`](../fcft/accel.py) `build_activation()` 공용 팩토리 사용 |
| **새 비교 축 (`--axis`)** | [`configs/axes.py`](../configs/axes.py) | `--workers` 병렬 호환을 위해 `lambda` 대신 `OptimizerFactory` / `SchedulerFactory` 사용 (`pickle.dumps` 통과 필수) |

---

## 2. 반드시 지킬 3가지 원칙

1. **백본 동결 유지**: `build_backbone()`의 `requires_grad=False`와 BatchNorm `eval()` 잠금(`_freeze_bn_stats`)을 유지합니다.
2. **Test 세트 격리**: 하이퍼파라미터 탐색(`grid`, `loss`, `activation`, `kfold`)은 `train`/`val`만 사용하고 `test`는 최종 1회만 평가합니다.
3. **데이터 누수 차단**: 평균/표준편차 통계 및 `1-Shot` 초기화는 반드시 `train` 분할로만 계산합니다.

---

## 3. 변경 후 회귀 테스트

```bash
python -m pytest tests -v          # Linux / macOS
py -3.12 -m pytest tests -v        # Windows
```

- [`tests/test_accel_core.py`](../tests/test_accel_core.py): 네이티브 바이너리·볼트 로드, 1-Shot 솔버, 압축 뱅크, 직렬화
- [`tests/test_data_cache.py`](../tests/test_data_cache.py): EXIF 보정, split 검증, 다중 풀링 디스크 캐시 정합성
- [`tests/test_core_pipeline.py`](../tests/test_core_pipeline.py): 헤드·손실·동결 BN 불변, 중단 로그 보존, 앙상블
- [`tests/test_axes_grid.py`](../tests/test_axes_grid.py): 전체 축 계약, `pickle` 호환, Grid/Halving/K-Fold, `spawn` 멀티워커 정합성
