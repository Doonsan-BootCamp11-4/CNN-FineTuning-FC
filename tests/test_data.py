import pytest
from torchvision import transforms as T

from fcft.data import build_loaders


# [정상 동작] 가짜 데이터(3클래스 × train 2 / val 1 / test 1장)로 기본 경로가 맞게 동작하는지.
# - 클래스 순서: CSV label을 정렬한 순서여야 run/evaluate에서 클래스 번호가 일치한다.
# - split별 개수: CSV의 split 열대로 나뉘는지 (train 6, val 3, test 3).
# - 배치 shape: val/test 전처리(224 resize, 3채널)가 적용되는지.
def test_loaders(config: dict) -> None:
    loaders = build_loaders(config)
    assert loaders["train"].dataset.classes == ["glass", "paper", "plastic"]
    assert [len(loaders[s].dataset) for s in ("train", "val", "test")] == [6, 3, 3]
    x, _ = next(iter(loaders["val"]))
    assert x.shape == (2, 3, 224, 224)


# [동등 분할] 입력 이미지를 "RGB"와 "RGB가 아닌 것(흑백·RGBA 등)" 두 그룹으로 나누고,
# 두 번째 그룹의 대표값으로 흑백(L) 이미지 1장(paper/0.jpg)을 넣었다.
# 흑백이 그대로 들어가면 1채널이 되어 Normalize(3채널 평균)에서 에러가 나므로, 모두 3채널인지 본다.
# 통과하는 이유: SplitDataset.__getitem__의 .convert("RGB")가 흑백의 밝기 값 하나를 R, G, B에 똑같이 복사해 3채널로 만든다.
# paper/0.jpg는 0번이라 train에 들어 있으므로 train dataset을 전부 돌면 이 이미지도 검사된다.
# .convert("RGB")를 지우면 이 테스트가 실패해야 한다.
def test_non_rgb_image(config: dict) -> None:
    ds = build_loaders(config)["train"].dataset
    assert all(ds[i][0].shape[0] == 3 for i in range(len(ds)))


# [분기] train_transform을 넘겼을 때 train에만 쓰이고 val/test는 고정 전처리를 쓰는지.
# 크기가 다른 transform(64)을 넘겨서, train은 64, val은 224로 나오면 두 분기가 각각 맞게 동작한 것이다.
def test_train_transform_only_on_train(config: dict) -> None:
    tf = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    loaders = build_loaders(config, train_transform=tf)
    assert next(iter(loaders["train"]))[0].shape[-1] == 64
    assert next(iter(loaders["val"]))[0].shape[-1] == 224


# [동등 분할] config의 num_classes를 "CSV 클래스 수와 같음 / 다름" 두 그룹으로 나눈다.
# 같음 그룹은 위 테스트들이 확인하고, 여기서는 다름 그룹의 대표값(12, 가짜 데이터는 3클래스)으로 에러가 나는지 본다.
# 다르면 FC 출력 크기와 클래스 번호가 어긋나 학습 중(CUDA assert)이나 평가 때 알아보기 어려운 에러가 난다.
def test_num_classes_mismatch(config: dict) -> None:
    with pytest.raises(ValueError):
        build_loaders({**config, "num_classes": 12})
