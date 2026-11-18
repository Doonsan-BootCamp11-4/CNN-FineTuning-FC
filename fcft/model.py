from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(name: str = "resnet18", num_classes: int = 12) -> nn.Module:
    """ImageNet 사전학습 백본을 동결하고 FC만 새로 붙여 학습 가능하게 둔다."""
    if name != "resnet18":
        raise ValueError(f"지원하지 않는 모델: {name}")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Feature extractor는 학습 금지를 위한 False 설정
    for p in model.parameters():
        p.requires_grad = False

    # 기존 모델의 분류층 fc 레이어 변경
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 새로 만든 층은 requires_grad=True
    return model
