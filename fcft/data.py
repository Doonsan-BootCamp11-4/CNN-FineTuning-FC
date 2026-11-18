from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# val/test 전처리는 고정한다 (ImageNet 사전학습 모델 기준).
EVAL_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class SplitDataset(Dataset):
    """split CSV의 행 목록을 (이미지 텐서, 클래스 번호)로 돌려주는 Dataset."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        data_root: str | Path,
        classes: list[str],
        transform: Callable,
    ) -> None:
        self.data_root = Path(data_root)
        self.classes = classes

        class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = [(r["path"], class_to_idx[r["label"]]) for r in rows]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    # Dataset 클래스는 __getitem__ 메소드 필수 구현
    def __getitem__(self, i: int) -> tuple[Tensor, int]:
        path, label = self.samples[i]
        # 흑백·RGBA 이미지도 3채널로 맞춘다. with로 열어 읽은 즉시 파일을 닫는다.
        with Image.open(self.data_root / path) as f:
            img = f.convert("RGB")
        return self.transform(img), label


def build_loaders(
    config: dict, train_transform: Callable | None = None
) -> dict[str, DataLoader]:
    """split CSV → {"train", "val", "test"} DataLoader.

    클래스 순서는 label 이름 정렬순. train_transform이 None이면 train도 EVAL_TRANSFORM을 쓴다.
    """
    with open(config["split_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # class set -> list를 통해 정렬
    classes = sorted(set(r["label"] for r in rows))
    # FC 출력 크기(num_classes)와 실제 클래스 수가 다르면 학습·평가 중 알아보기 어려운 에러가 나므로 미리 막는다
    if len(classes) != config["num_classes"]:
        raise ValueError(f"num_classes={config['num_classes']}인데 CSV의 클래스는 {len(classes)}개: {classes}")

    if train_transform is None:
        train_transform = EVAL_TRANSFORM

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"

        dataset = SplitDataset(
            [r for r in rows if r["split"] == split],
            config["data_root"],
            classes,
            train_transform if is_train else EVAL_TRANSFORM,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=is_train,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),  # GPU로 옮기는 속도 향상 (CPU에서는 의미 없어 끔)
        )
    return loaders
