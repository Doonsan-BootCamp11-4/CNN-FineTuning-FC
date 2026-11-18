import pytest
import torch

from fcft.model import build_model


# [요구사항] FC만 학습한다는 프로젝트 전제를 확인한다.
# requires_grad=True인 파라미터가 정확히 fc.weight, fc.bias 두 개뿐이어야 한다.
def test_only_fc_trainable() -> None:
    model = build_model("resnet18", 12)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


# [출력 형식] 224×224 이미지 배치를 넣으면 (배치 크기, 클래스 수) 모양의 점수가 나오는지.
def test_output_shape() -> None:
    assert build_model().eval()(torch.randn(2, 3, 224, 224)).shape == (2, 12)


# [동등 분할] 모델 이름을 "지원함(resnet18)"과 "지원 안 함" 두 그룹으로 나눈다.
# 지원함 그룹은 위 테스트들이 확인하고, 여기서는 지원 안 함 그룹의 대표값(vgg16)으로 ValueError가 나는지 본다.
def test_unknown_model() -> None:
    with pytest.raises(ValueError):
        build_model("vgg16")
