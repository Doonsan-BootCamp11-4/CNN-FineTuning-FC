# FC 파인튜닝 공통 파이프라인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로컬에서 가짜 데이터로 검증하고 Kaggle에서 학습할 수 있는 공통 파이프라인(config, model, data, train, evaluate)을 만든다.

**Architecture:** config dict에는 공통 값만 두고, optimizer·scheduler·train transform은 `run`의 인자로 받는다. 각 Task는 하나의 파일을 만들고, Task 하나가 끝날 때마다 사용자가 확인한 뒤 다음 Task로 넘어간다.

**Tech Stack:** Python 3.12(로컬 venv), torch/torchvision(로컬은 CPU 빌드), Pillow, scikit-learn, matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-11-10-fc-transfer-learning-design.md`

## Global Constraints

- 모든 함수는 인자·반환 타입 힌트를 쓴다.
- 커밋 메시지는 한국어, `feat:`/`add:`/`fix:`/`test:` 접두어(기존 기록과 같은 형식).
- 로컬 테스트는 실제 데이터셋 없이 `tmp_path`의 가짜 이미지로만 돈다. 테스트용 분기(`pretrained=False` 등)는 두지 않는다.
- 테스트의 config는 `num_workers=0`(Windows에서 worker 프로세스 문제 회피).
- val/test transform: `Resize((224, 224))` → `ToTensor()` → `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.
- 클래스 순서는 CSV `label` 값을 정렬한 순서. run과 evaluate가 같은 순서를 써야 한다.

## Review Focus

1. 동결된 BatchNorm의 running mean/var가 학습 중에 바뀌면 FC만 학습한 것이 아니게 된다 → Task 5 `test_backbone_unchanged`.
2. 데이터셋에 흑백·RGBA 이미지가 섞여 있으면 채널 수가 달라 배치가 깨진다 → Task 4 `test_non_rgb_image`.
3. `ReduceLROnPlateau`는 `step(metric)`이 필요해 다른 scheduler와 호출 방식이 다르다 → Task 5 `test_plateau_scheduler`.
4. 넘긴 `train_transform`이 val/test에 섞여 들어가면 val 점수가 달라진다 → Task 4 `test_train_transform_only_on_train`.
5. 넘긴 optimizer가 무시되고 기본 Adam이 쓰이면 실험이 의미 없어진다 → Task 5 `test_custom_optimizer_and_scheduler`.

---

### Task 1: 로컬 환경

**Files:**
- Create: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Produces: `.venv` 안에서 `python -m pytest tests -q`가 실행되는 환경.

- [ ] **Step 1: `requirements.txt` 작성** — `torch`, `torchvision`, `pillow`, `scikit-learn`, `matplotlib`, `pytest` (버전 고정 없음).

- [ ] **Step 2: README에 로컬 세팅 절차 추가** — torch는 CPU 빌드를 먼저 따로 설치한다(리눅스 PyPI의 torch는 CUDA 포함이라 크다).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest tests -q
```

- [ ] **Step 3: 실제로 venv를 만들고 설치해 확인**

Run: `.venv\Scripts\python -c "import torch, torchvision, sklearn, matplotlib, pytest; print(torch.__version__)"`
Expected: `+cpu`가 붙은 버전 출력

- [ ] **Step 4: Commit** — `add: 로컬 venv 세팅과 requirements.txt`

---

### Task 2: `configs/base.py`

**Files:**
- Create: `configs/base.py` (현재 빈 파일), `configs/__init__.py`가 없으면 만들지 않는다(namespace package로 import 됨)

**Interfaces:**
- Produces: `BASE: dict` — 키와 값:
  - `name="base"`
  - `split_csv="/kaggle/working/CNN-FineTuning-Dataset-Split/splits/garbage12_602020_seed42.csv"`
  - `data_root="/kaggle/input/garbage-classification/garbage_classification"`
  - `out_dir="outputs"`, `num_workers=2`
  - `model="resnet18"`, `num_classes=12`
  - `epochs=15`, `batch_size=32`, `seed=42`

- [ ] **Step 1: `BASE` 작성** — 실행 환경 / 모델 / 학습 세 묶음으로 주석을 단다.

- [ ] **Step 2: 확인**

Run: `.venv\Scripts\python -c "from configs.base import BASE; print(sorted(BASE))"`
Expected: 위 11개 키 출력

- [ ] **Step 3: Commit** — `feat: 공통 config BASE 추가`

---

### Task 3: `fcft/model.py`

**Files:**
- Modify: `fcft/model.py` (기존 `hello_world` 삭제)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `build_model(name: str = "resnet18", num_classes: int = 12) -> nn.Module`
  - spec은 `build_model(num_classes=12)`이지만 config에 `model` 키가 있으므로 이름을 받는다(계획 작성자 판단).
  - `name`이 `"resnet18"`이 아니면 `ValueError`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_only_fc_trainable() -> None:
    model = build_model("resnet18", 12)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}

def test_output_shape() -> None:
    assert build_model().eval()(torch.randn(2, 3, 224, 224)).shape == (2, 12)

def test_unknown_model() -> None:
    with pytest.raises(ValueError):
        build_model("vgg16")
```

- [ ] **Step 2: 실패 확인** — `python -m pytest tests/test_model.py -q` → ImportError

- [ ] **Step 3: 구현** — `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, 모든 파라미터 `requires_grad=False` 후 `fc`를 `nn.Linear(512, num_classes)`로 교체.

- [ ] **Step 4: 통과 확인** — `python -m pytest tests/test_model.py -q` → 3 passed (최초 1회 가중치 다운로드)

- [ ] **Step 5: Commit** — `feat: ResNet18 백본 동결 모델`

---

### Task 4: `fcft/data.py`

**Files:**
- Create: `tests/conftest.py`
- Modify: `fcft/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `EVAL_TRANSFORM: T.Compose` (Global Constraints의 값)
  - `SplitDataset(rows: list[dict[str, str]], data_root: str | Path, classes: list[str], transform: Callable) -> Dataset`, 속성 `classes: list[str]`, 항목은 `(Tensor, int)`. 이미지는 `.convert("RGB")`.
  - `build_loaders(config: dict, train_transform: Callable | None = None) -> dict[str, DataLoader]` — 키 `"train"`, `"val"`, `"test"`. train만 shuffle. `train_transform`이 None이면 train도 `EVAL_TRANSFORM`.
- Produces (테스트용): `tests/conftest.py`의 fixture `config(tmp_path) -> dict`
  - 클래스 `glass`, `paper`, `plastic` × 이미지 4장(split: train, train, val, test) = 12장, 40×30 단색 jpg
  - `paper/0.jpg`는 흑백(`"L"`) 이미지로 저장
  - 반환: `{**BASE, "name": "smoke", "epochs": 1, "batch_size": 2, "num_workers": 0, "num_classes": 3, "split_csv": ..., "data_root": str(tmp_path), "out_dir": str(tmp_path / "out")}`

- [ ] **Step 1: fixture와 실패하는 테스트 작성**

```python
def test_loaders(config: dict) -> None:
    loaders = build_loaders(config)
    assert loaders["train"].dataset.classes == ["glass", "paper", "plastic"]
    assert [len(loaders[s].dataset) for s in ("train", "val", "test")] == [6, 3, 3]
    x, y = next(iter(loaders["val"]))
    assert x.shape == (2, 3, 224, 224)

def test_non_rgb_image(config: dict) -> None:
    ds = build_loaders(config)["train"].dataset
    assert all(ds[i][0].shape[0] == 3 for i in range(len(ds)))

def test_train_transform_only_on_train(config: dict) -> None:
    tf = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    loaders = build_loaders(config, train_transform=tf)
    assert next(iter(loaders["train"]))[0].shape[-1] == 64
    assert next(iter(loaders["val"]))[0].shape[-1] == 224
```

- [ ] **Step 2: 실패 확인** — `python -m pytest tests/test_data.py -q` → ImportError

- [ ] **Step 3: 구현** — CSV는 표준 `csv.DictReader`로 읽는다.

- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 6 passed

- [ ] **Step 5: Commit** — `feat: split CSV 기반 DataLoader`

---

### Task 5: `fcft/train.py`

**Files:**
- Modify: `fcft/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_model(name, num_classes)`, `build_loaders(config, train_transform)`
- Produces:
  - `run(config: dict, optimizer: Callable[[Iterable[Parameter]], Optimizer] | None = None, scheduler: Callable[[Optimizer], Any] | None = None, train_transform: Callable | None = None) -> dict`
    - 반환: `{"name": str, "best_val_acc": float, "out_dir": str}`
    - 파일: `<out_dir>/<name>/best.pt`(state_dict, val acc 최고일 때 저장), `<out_dir>/<name>/history.json`(epoch별 `{"epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"}` 리스트, `lr`은 해당 epoch 시작 시점 값)
    - optimizer 기본값: `Adam(params, lr=1e-3)`. requires_grad인 파라미터만 넘긴다.
    - scheduler는 epoch마다 1회 step. `ReduceLROnPlateau`면 `step(val_loss)`.
    - seed는 `random`, `numpy`, `torch` 모두 설정.
    - 학습 중에도 동결된 BatchNorm은 eval 모드로 둔다.

- [ ] **Step 1: 실패하는 테스트 작성** (`config` fixture 사용)

```python
def test_run_outputs(config: dict) -> None:
    result = run(config)
    out = Path(result["out_dir"])
    assert (out / "best.pt").exists()
    history = json.loads((out / "history.json").read_text())
    assert len(history) == 1 and set(history[0]) == {"epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"}

def test_custom_optimizer_and_scheduler(config: dict) -> None:
    result = run({**config, "epochs": 2},
                 optimizer=lambda p: torch.optim.SGD(p, lr=0.5),
                 scheduler=lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.1))
    lrs = [h["lr"] for h in json.loads((Path(result["out_dir"]) / "history.json").read_text())]
    assert lrs == pytest.approx([0.5, 0.05])

def test_plateau_scheduler(config: dict) -> None:
    run(config, scheduler=lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o))  # 예외 없이 끝나면 통과

def test_backbone_unchanged(config: dict) -> None:
    before = {k: v.clone() for k, v in build_model("resnet18", 3).state_dict().items() if not k.startswith("fc.")}
    result = run(config)
    after = torch.load(Path(result["out_dir"]) / "best.pt")
    assert all(torch.equal(before[k], after[k]) for k in before)
```

- [ ] **Step 2: 실패 확인** — `python -m pytest tests/test_train.py -q` → ImportError

- [ ] **Step 3: 구현** — 1 epoch는 `run_once_epoch(model, loaders, criterion, device, optimizer) -> dict[str, float]`가 `_train`(학습)과 `_eval`(평가)을 차례로 부른다. optimizer 생성은 `_build_optimizer`, lr 갱신은 `_step_scheduler`로 나눈다.

- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 10 passed

- [ ] **Step 5: Commit** — `feat: run() 학습 루프`

---

### Task 6: `fcft/evaluate.py`와 Kaggle 사용법

**Files:**
- Modify: `fcft/evaluate.py`, `README.md`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `run(config)`이 남긴 `best.pt`, `history.json`
- Produces: `evaluate(config: dict) -> dict`
  - 반환/저장(`metrics.json`): `{"accuracy": float, "per_class": <sklearn classification_report output_dict>, "confusion_matrix": list[list[int]]}`
  - 그림: `confusion_matrix.png`, `curves.png`(loss, acc 두 칸, train/val)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_evaluate(config: dict) -> None:
    out = Path(run(config)["out_dir"])
    metrics = evaluate(config)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 3
    for f in ("metrics.json", "confusion_matrix.png", "curves.png"):
        assert (out / f).exists()
```

- [ ] **Step 2: 실패 확인** — `python -m pytest tests/test_evaluate.py -q` → ImportError

- [ ] **Step 3: 구현** — matplotlib은 파일 저장만 하고 `plt.close`.

- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 11 passed

- [ ] **Step 5: README에 Kaggle 노트북 셀 순서 추가** — clone/pull → `sys.path.insert(0, "/kaggle/working/CNN-FineTuning-FC")` → `run(config, optimizer=..., ...)` → `evaluate(config)`.

- [ ] **Step 6: Commit** — `feat: test 평가와 그래프`
