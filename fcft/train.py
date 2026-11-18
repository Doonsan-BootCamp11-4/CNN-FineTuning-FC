import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from fcft.data import build_loaders
from fcft.model import build_model


def set_seed(seed: int = 42) -> None:
    """가중치 초기화·셔플 순서를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_bn_stats(model: nn.Module) -> None:
    """동결된 BatchNorm을 eval 모드로 둔다.

    BatchNorm은 requires_grad=False여도 train 모드에서는 running_mean/var가 배치마다 바뀌므로,
    FC만 학습하려면 학습 중에도 eval 모드로 고정해야 한다.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def _train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer,
) -> tuple[float, float]:
    """train 데이터로 한 바퀴 학습한다. (평균 loss, 정확도)를 돌려준다."""
    model.train()
    freeze_bn_stats(model)

    total_loss, correct, n = 0.0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
        
    return total_loss / n, correct / n


# with torch.no_grad():대신 이것도 됨
@torch.no_grad()
def _eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """가중치를 바꾸지 않고 loss와 정확도만 잰다. (평균 loss, 정확도)를 돌려준다."""
    model.eval()

    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        
        total_loss += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)

    return total_loss / n, correct / n


def run_once_epoch(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer,
) -> dict[str, float]:
    """1 epoch = train 학습 한 바퀴 + val 평가 한 바퀴."""
    train_loss, train_acc = _train(model, loaders["train"], criterion, device, optimizer)

    val_loss, val_acc = _eval(model, loaders["val"], criterion, device)

    return {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}


def _build_optimizer(
    model: nn.Module,
    optimizer: Callable[[Iterable[Parameter]], Optimizer] | None,
) -> Optimizer:
    """학습할 파라미터(FC)만 optimizer에 넘긴다. optimizer가 None이면 Adam(lr=1e-3)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return optimizer(params) if optimizer is not None else torch.optim.Adam(params, lr=1e-3)


def _step_scheduler(sched: Any, val_loss: float) -> None:
    """epoch 끝에 lr을 갱신한다. ReduceLROnPlateau만 지표(val_loss)를 받는다."""
    if isinstance(sched, ReduceLROnPlateau):
        sched.step(val_loss)
    elif sched:
        sched.step()


def run(
    config: dict,
    optimizer: Callable[[Iterable[Parameter]], Optimizer] | None = None,
    scheduler: Callable[[Optimizer], Any] | None = None,
    train_transform: Callable | None = None,
) -> dict:
    """학습 후 val acc 기준 best 모델을 <out_dir>/<name>/best.pt에 저장한다. test는 평가하지 않는다.

    optimizer: 학습할 파라미터 → Optimizer 를 만드는 함수. None이면 Adam(lr=1e-3).
    scheduler: Optimizer → scheduler 를 만드는 함수. None이면 쓰지 않는다.
    train_transform: train에 쓸 transform. None이면 val/test와 같은 전처리.
    """
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["out_dir"]) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(config, train_transform)
    model = build_model(config["model"], config["num_classes"]).to(device)
    opt = _build_optimizer(model, optimizer)
    sched = scheduler(opt) if scheduler else None
    criterion = nn.CrossEntropyLoss()

    history, best_acc = [], -1.0
    for epoch in range(1, config["epochs"] + 1):
        lr = opt.param_groups[0]["lr"]  # 이번 epoch 시작 시점의 lr
        m = run_once_epoch(model, loaders, criterion, device, opt)
        history.append({"epoch": epoch, "lr": lr, **m})
        print(f"[{config['name']}] {epoch}/{config['epochs']} "
              f"train {m['train_loss']:.4f}/{m['train_acc']:.4f}  "
              f"val {m['val_loss']:.4f}/{m['val_acc']:.4f}  lr {lr:.2e}")

        if m["val_acc"] > best_acc:
            best_acc = m["val_acc"]
            torch.save(model.state_dict(), out_dir / "best.pt")
        _step_scheduler(sched, m["val_loss"])

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"name": config["name"], "best_val_acc": best_acc, "out_dir": str(out_dir)}
