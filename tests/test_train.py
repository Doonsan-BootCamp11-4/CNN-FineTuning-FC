import json
from pathlib import Path

import pytest
import torch

from fcft.model import build_model
from fcft.train import run


def read_history(result: dict) -> list[dict]:
    return json.loads((Path(result["out_dir"]) / "history.json").read_text())


# [정상 동작] 기본 인자로 1 epoch 돌리면 best.pt와 history.json이 생기고,
# history에는 epoch마다 정해진 키(epoch, lr, train/val loss·acc)가 남는지.
def test_run_outputs(config: dict) -> None:
    result = run(config)
    assert (Path(result["out_dir"]) / "best.pt").exists()
    history = read_history(result)
    assert len(history) == 1
    assert set(history[0]) == {"epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"}


# [인자 전달] 넘긴 optimizer와 scheduler가 기본값(Adam) 대신 실제로 쓰이는지.
# SGD lr=0.5에 StepLR(매 epoch ×0.1)을 주면 epoch별 lr이 0.5 → 0.05가 되어야 한다.
def test_custom_optimizer_and_scheduler(config: dict) -> None:
    result = run(
        {**config, "epochs": 2},
        optimizer=lambda params: torch.optim.SGD(params, lr=0.5),
        scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1),
    )
    assert [h["lr"] for h in read_history(result)] == pytest.approx([0.5, 0.05])


# [분기] ReduceLROnPlateau는 다른 scheduler와 달리 step(val_loss)처럼 지표를 넘겨야 한다.
# 지표 없이 step()을 부르면 TypeError가 나므로, 예외 없이 끝나면 통과.
def test_plateau_scheduler(config: dict) -> None:
    run(config, scheduler=lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt))


class _StopAfterFirstEpoch:
    """첫 epoch 끝(scheduler.step 시점)에 예외를 내서 학습 중단(Kaggle 세션 종료 등)을 흉내 낸다."""

    def step(self) -> None:
        raise RuntimeError("학습 중단")


# [예외 상황] 학습이 중간에 끊겨도 best.pt와 history.json이 같은 실행의 것이어야 한다.
# history.json을 학습이 끝난 뒤에만 쓰면, 끊긴 실행은 best.pt만 새것이고 history.json은 없거나 이전 실행 것으로 남는다.
def test_history_saved_each_epoch(config: dict) -> None:
    with pytest.raises(RuntimeError):
        run({**config, "epochs": 2}, scheduler=lambda opt: _StopAfterFirstEpoch())
    out = Path(config["out_dir"]) / config["name"]
    assert (out / "best.pt").exists()
    assert len(json.loads((out / "history.json").read_text())) == 1


# [요구사항] FC만 학습했다면 백본 가중치와 BatchNorm 통계(running_mean/var)가 사전학습 값 그대로여야 한다.
# BatchNorm은 requires_grad=False여도 train 모드에서는 통계가 바뀌므로, 동결된 BN을 eval 모드로 두지 않으면 실패한다.
def test_backbone_unchanged(config: dict) -> None:
    before = {k: v.clone() for k, v in build_model("resnet18", 3).state_dict().items() if not k.startswith("fc.")}
    result = run(config)
    after = torch.load(Path(result["out_dir"]) / "best.pt")
    assert all(torch.equal(before[k], after[k]) for k in before)
