# CNN-FineTuning-FC
FC 레이어만 파인 튜닝

## 로컬 세팅 (테스트용)

로컬에서는 코드가 끝까지 도는지만 확인하고, 실제 학습은 Kaggle에서 한다.
데이터셋은 받지 않는다. 테스트가 가짜 이미지를 만들어 쓴다.

torch는 CPU 빌드를 먼저 따로 설치한다 (기본 PyPI의 리눅스용 torch는 CUDA 포함이라 크다).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest tests -q
```

VS Code에서는 `Tasks: Run Task` → `venv: setup`으로 venv 생성과 설치를 한 번에 할 수 있다.

## Kaggle에서 학습

노트북에 Garbage Classification 데이터셋을 추가하고, GPU와 Internet을 켠 뒤 아래 순서로 셀을 실행한다.

```python
# 0) split CSV 받기
#    폴더 이름은 BASE["split_csv"] 경로와 정확히 같아야 한다:
#    /kaggle/working/CNN-FineTuning-Dataset-Split/splits/garbage12_602020_seed42.csv
!git clone <CNN-FineTuning-Dataset-Split 저장소 주소> /kaggle/working/CNN-FineTuning-Dataset-Split || git -C /kaggle/working/CNN-FineTuning-Dataset-Split pull

# 1) 저장소 받기 (이미 있으면 최신으로)
!git clone https://github.com/Doonsan-BootCamp11-4/CNN-FineTuning-FC.git /kaggle/working/CNN-FineTuning-FC || git -C /kaggle/working/CNN-FineTuning-FC pull

# 2) import 경로 추가
import sys
sys.path.insert(0, "/kaggle/working/CNN-FineTuning-FC")

# 3) config 만들고 학습 → test 평가
import torch
from configs.base import BASE
from fcft.train import run
from fcft.evaluate import evaluate

config = {**BASE, "name": "sgd_lr0.01"}
run(
    config,
    optimizer=lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),  # 안 넘기면 Adam(lr=1e-3)
    scheduler=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"]),  # 안 넘기면 없음
    # train_transform=...,  # 안 넘기면 val/test와 같은 전처리 (아래 주의 참고)
)
evaluate(config)  # outputs/<name>/ 에 metrics.json, confusion_matrix.png, curves.png
```

`train_transform`을 직접 만들 때는 augmentation 뒤에 반드시 val/test와 같은 형식을 맞춘다.
Normalize를 빼도 에러 없이 학습되지만 val 점수가 낮게 나와 실험 비교가 틀려진다.

```python
from torchvision import transforms as T
train_transform = T.Compose([
    T.RandomHorizontalFlip(),                                         # 실험할 augmentation
    T.Resize((224, 224)),                                             # 필수
    T.ToTensor(),                                                     # 필수
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),        # 필수 (ImageNet 평균/표준편차)
])
```

test 평가는 최종 모델에서 한 번만 한다 (설계 문서 "진행 단계" 참고). 실험 비교는 `history.json`의 val 성능으로 한다.
