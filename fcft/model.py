"""Model, pooling, and classifier head builders for frozen-backbone ResNet18 transfer learning."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from fcft.accel import ResidualAdapterHead, SmoothActivation, build_activation, extract_dual_descriptors, is_accel_available, spectral_power_norm


class GeMPool2d(nn.Module):
    """Generalized Mean Pooling layer returning shape (B, C, 1, 1)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1), keepdim=True).pow(1.0 / self.p)


class AvgMaxPool2d(nn.Module):
    """Concatenated global average and max pooling returning shape (B, 2C, 1, 1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_p = F.adaptive_avg_pool2d(x, (1, 1))
        max_p = F.adaptive_max_pool2d(x, (1, 1))
        return torch.cat([avg_p, max_p], dim=1)


class MultiScaleResNet18(nn.Module):
    """Frozen ResNet18 concatenating pooled layer3 (256D) and layer4 (512D) features."""

    def __init__(self, backbone: nn.Module, balance_blocks: bool = False):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.balance_blocks = bool(balance_blocks)
        self.fc: nn.Module = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        p3 = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)
        p4 = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)
        if self.balance_blocks:
            p3 = F.normalize(p3, p=2, dim=-1)
            p4 = F.normalize(p4, p=2, dim=-1)
        return self.fc(torch.cat([p3, p4], dim=1))


class FeatureNormLayer(nn.Module):
    """Apply optional signed square-root power scaling followed by L2 normalization."""

    def __init__(self, power: bool = True):
        super().__init__()
        self.power = bool(power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.sign(x) * torch.sqrt(x.abs() + 1e-12) if self.power else x
        return F.normalize(z, p=2, dim=-1)


class FeatureCenterNormLayer(nn.Module):
    """Center and optionally standardize embeddings using train-split statistics before L2 normalization."""

    def __init__(self, dim: int, scale_std: bool = False, power: bool = True):
        super().__init__()
        self.scale_std = bool(scale_std)
        self.power = bool(power)
        self.register_buffer("mean", torch.zeros(1, int(dim), dtype=torch.float32))
        if self.scale_std:
            self.register_buffer("inv_std", torch.ones(1, int(dim), dtype=torch.float32))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_stats(self, raw_features: torch.Tensor) -> None:
        if raw_features.numel() == 0:
            return
        self.mean.copy_(raw_features.mean(dim=0, keepdim=True).to(self.mean))
        if self.scale_std:
            sd = raw_features.std(dim=0, keepdim=True, unbiased=False).clamp(min=5e-4).to(self.inv_std)
            self.inv_std.copy_(1.0 / sd)
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean.to(x)) if bool(self.fitted.item()) else x
        if self.scale_std and bool(self.fitted.item()):
            z = z * self.inv_std.to(z)
        if self.power:
            z = torch.sign(z) * torch.sqrt(z.abs() + 1e-12)
        return F.normalize(z, p=2, dim=-1)


class SpectralNormLayer(nn.Module):
    """Apply compiled curvature-gated signed-power L2 feature calibration."""

    def __init__(self, curv_strength: float = 1.0):
        super().__init__()
        self.curv_strength = float(curv_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_accel_available():
            return spectral_power_norm(x, curv_strength=self.curv_strength)
        z = torch.sign(x) * torch.sqrt(x.abs() + 1e-12)
        return F.normalize(z, p=2, dim=-1)


class MLPClassifierHead(nn.Module):
    """Bottleneck LayerNorm + configurable activation + Dropout MLP classification head."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        leak: float = 0.0625,
        activation: str = "smooth",
        learnable_leak: bool = False,
        channel_leak: bool = False,
    ):
        super().__init__()
        n_ch = int(hidden_dim) if bool(channel_leak) else 1
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            build_activation(activation, leak=leak, learnable_leak=bool(learnable_leak), num_channels=n_ch),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualAnchorHybridHead(nn.Module):
    """Cosine dual-anchor prototype classifier fused with a zero-init residual adapter."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale: float = 16.0,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        leak: float = 0.0625,
        activation: str = "smooth",
        learnable_leak: bool = False,
        channel_leak: bool = False,
    ):
        super().__init__()
        self.anchors = nn.Parameter(torch.empty(int(num_classes), int(in_dim)))
        nn.init.xavier_uniform_(self.anchors)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.res_weight = nn.Parameter(torch.tensor(0.5))
        self.adapter = ResidualAdapterHead(
            in_dim=in_dim, num_classes=int(num_classes), hidden_dim=hidden_dim, depth=1, dropout=dropout,
            leak=leak, activation=activation, learnable_leak=bool(learnable_leak), channel_leak=bool(channel_leak),
        )
        nn.init.zeros_(self.adapter.fc.weight)
        nn.init.zeros_(self.adapter.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        a_norm = F.normalize(self.anchors, p=2, dim=-1)
        return F.linear(x_norm, a_norm) * self.scale + self.res_weight * self.adapter(x)


class CosineClassifierHead(nn.Module):
    """L2-normalized cosine similarity classifier with learnable temperature."""

    def __init__(self, in_dim: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(num_classes), int(in_dim)))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        return F.linear(x_norm, w_norm) * self.scale


class SubCenterHead(nn.Module):
    """Multi-prototype cosine classifier taking max similarity across K sub-centers per class."""

    def __init__(self, in_dim: int, num_classes: int, num_centers: int = 3, scale: float = 16.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_centers = max(1, int(num_centers))
        self.weight = nn.Parameter(torch.empty(self.num_classes * self.num_centers, int(in_dim)))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        logits = F.linear(x_norm, w_norm) * self.scale
        return logits.view(x.size(0), self.num_classes, self.num_centers).max(dim=-1).values


def build_head(
    in_dim: int,
    num_classes: int,
    head_type: str = "linear",
    feature_norm: bool = False,
    feature_norm_mode: Optional[str] = None,
    feature_dropout: float = 0.0,
    **head_kwargs,
) -> nn.Module:
    """Build classification head with configurable feature normalization and dropout."""
    n_cls = int(num_classes)
    act_name = str(head_kwargs.get("activation", "smooth"))
    if head_type == "linear":
        core: nn.Module = nn.Linear(in_dim, n_cls)
    elif head_type == "cosine":
        core = CosineClassifierHead(in_dim, n_cls, scale=float(head_kwargs.get("scale", 16.0)))
    elif head_type == "subcenter":
        core = SubCenterHead(
            in_dim, n_cls, num_centers=int(head_kwargs.get("num_centers", 3)), scale=float(head_kwargs.get("scale", 16.0)),
        )
    elif head_type == "residual":
        core = ResidualAdapterHead(
            in_dim=in_dim, num_classes=n_cls, hidden_dim=int(head_kwargs.get("hidden_dim", 256)),
            depth=int(head_kwargs.get("depth", 1)), dropout=float(head_kwargs.get("dropout", 0.2)),
            leak=float(head_kwargs.get("leak", 0.0625)), activation=act_name,
            learnable_leak=bool(head_kwargs.get("learnable_leak", False)),
            channel_leak=bool(head_kwargs.get("channel_leak", False)),
        )
    elif head_type == "mlp":
        core = MLPClassifierHead(
            in_dim=in_dim, num_classes=n_cls, hidden_dim=int(head_kwargs.get("hidden_dim", 256)),
            dropout=float(head_kwargs.get("dropout", 0.3)), leak=float(head_kwargs.get("leak", 0.0625)),
            activation=act_name, learnable_leak=bool(head_kwargs.get("learnable_leak", False)),
            channel_leak=bool(head_kwargs.get("channel_leak", False)),
        )
    elif head_type == "dual_anchor":
        core = DualAnchorHybridHead(
            in_dim=in_dim, num_classes=n_cls, scale=float(head_kwargs.get("scale", 16.0)),
            hidden_dim=int(head_kwargs.get("hidden_dim", 256)), dropout=float(head_kwargs.get("dropout", 0.2)),
            leak=float(head_kwargs.get("leak", 0.0625)), activation=act_name,
            learnable_leak=bool(head_kwargs.get("learnable_leak", False)),
            channel_leak=bool(head_kwargs.get("channel_leak", False)),
        )
    else:
        raise ValueError(f"Unsupported head_type: {head_type!r}")
    norm_k = feature_norm_mode if feature_norm_mode is not None else ("power_l2" if feature_norm else "none")
    norm_builders = {
        "l2": lambda: FeatureNormLayer(power=False),
        "power_l2": lambda: FeatureNormLayer(power=True),
        "center_power_l2": lambda: FeatureCenterNormLayer(in_dim, scale_std=False, power=True),
        "center_std": lambda: FeatureCenterNormLayer(in_dim, scale_std=True, power=False),
        "center_std_power": lambda: FeatureCenterNormLayer(in_dim, scale_std=True, power=True),
        "spectral": lambda: SpectralNormLayer(curv_strength=float(head_kwargs.get("curv_strength", 1.0))),
    }
    prefix: list[nn.Module] = []
    if norm_k in norm_builders:
        prefix.append(norm_builders[norm_k]())
    elif norm_k not in (None, "none"):
        raise ValueError(f"Unsupported feature_norm_mode: {norm_k!r}")
    if float(feature_dropout) > 0.0:
        prefix.append(nn.Dropout(p=float(feature_dropout)))
    return nn.Sequential(*prefix, core) if prefix else core


def fuse_backbone_conv_bn(module: nn.Module) -> nn.Module:
    """Fuse Conv2d+BatchNorm2d pairs in a frozen eval() ResNet backbone to cut extraction overhead."""
    module.eval()
    if hasattr(module, "conv1") and hasattr(module, "bn1") and isinstance(module.conv1, nn.Conv2d) and isinstance(module.bn1, nn.BatchNorm2d):
        module.conv1 = torch.nn.utils.fuse_conv_bn_eval(module.conv1, module.bn1)
        module.bn1 = nn.Identity()
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        seq = getattr(module, layer_name, None)
        if isinstance(seq, nn.Sequential):
            for blk in seq:
                for c_nm, b_nm in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
                    conv_m = getattr(blk, c_nm, None)
                    bn_m = getattr(blk, b_nm, None)
                    if isinstance(conv_m, nn.Conv2d) and isinstance(bn_m, nn.BatchNorm2d):
                        setattr(blk, c_nm, torch.nn.utils.fuse_conv_bn_eval(conv_m, bn_m))
                        setattr(blk, b_nm, nn.Identity())
                ds = getattr(blk, "downsample", None)
                if isinstance(ds, nn.Sequential) and len(ds) >= 2 and isinstance(ds[0], nn.Conv2d) and isinstance(ds[1], nn.BatchNorm2d):
                    ds[0] = torch.nn.utils.fuse_conv_bn_eval(ds[0], ds[1])
                    ds[1] = nn.Identity()
    return module


_BACKBONE_BUILDERS = {
    "resnet18": (models.resnet18, lambda: models.ResNet18_Weights.DEFAULT, 768),
    "resnet34": (models.resnet34, lambda: models.ResNet34_Weights.DEFAULT, 768),
    "resnet50": (models.resnet50, lambda: models.ResNet50_Weights.DEFAULT, 3072),
}


def build_backbone(name: str = "resnet18", pool_mode: str = "avg", fuse_conv_bn: bool = False, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Load frozen pretrained ResNet backbone (resnet18/34/50) with configurable pooling mode."""
    if name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unsupported model: {name!r}. Supported: {', '.join(_BACKBONE_BUILDERS)}.")
    ctor, w_fn, ms_dim = _BACKBONE_BUILDERS[name]
    model = ctor(weights=(w_fn() if pretrained else None))
    for param in model.parameters():
        param.requires_grad = False
    base_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    if fuse_conv_bn:
        fuse_backbone_conv_bn(model)

    if pool_mode == "avg":
        out_mod: nn.Module = model
        out_dim = base_dim
    elif pool_mode == "avgmax":
        model.avgpool = AvgMaxPool2d()
        out_mod = model
        out_dim = base_dim * 2
    elif pool_mode == "gem":
        model.avgpool = GeMPool2d(p=3.0)
        out_mod = model
        out_dim = base_dim
    elif pool_mode == "multiscale":
        out_mod = MultiScaleResNet18(model, balance_blocks=False)
        out_dim = ms_dim
    elif pool_mode == "norm_multiscale":
        out_mod = MultiScaleResNet18(model, balance_blocks=True)
        out_dim = ms_dim
    else:
        raise ValueError(f"Unsupported pool_mode: {pool_mode!r}")

    out_mod.eval()
    return out_mod, out_dim


class ChannelStandardizer(nn.Module):
    """Store train-split per-channel mean/inv_std buffers and L2-normalize standardized descriptors."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, int(dim), dtype=torch.float32))
        self.register_buffer("inv_std", torch.ones(1, int(dim), dtype=torch.float32))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_stats(self, raw_features: torch.Tensor) -> None:
        if raw_features.numel() == 0:
            return
        mu = raw_features.mean(dim=0, keepdim=True).to(self.mean)
        sd = raw_features.std(dim=0, keepdim=True, unbiased=False).clamp(min=5e-4).to(self.inv_std)
        self.mean.copy_(mu)
        self.inv_std.copy_(1.0 / sd)
        self.fitted.fill_(True)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        z = (raw_features.to(self.mean) - self.mean) * self.inv_std if bool(self.fitted.item()) else raw_features
        return F.normalize(z, p=2, dim=-1)


class DescriptorClassifierModel(nn.Module):
    """Backbone-free classifier operating solely on dual-scale native subpixel descriptors."""

    def __init__(self, desc_weight: float = 1.0):
        super().__init__()
        self.standardizer = ChannelStandardizer(128)
        self.desc_weight = float(desc_weight)
        self.fc: nn.Module = nn.Identity()

    def extract_raw_parts(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        return None, extract_dual_descriptors(x, denormalize=True)

    def assemble_features(self, _bb_feat: Optional[torch.Tensor], raw_desc: torch.Tensor) -> torch.Tensor:
        return self.standardizer(raw_desc) * self.desc_weight

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.assemble_features(*self.extract_raw_parts(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class HybridClassifierModel(nn.Module):
    """Fusion classifier combining frozen ResNet18 features and dual-scale surface descriptors."""

    def __init__(self, backbone: nn.Module, desc_weight: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.bb_norm = FeatureNormLayer()
        self.standardizer = ChannelStandardizer(128)
        self.desc_weight = float(desc_weight)
        self.fc: nn.Module = nn.Identity()

    def extract_raw_parts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            bb_feat = self.backbone(x)
        desc_raw = extract_dual_descriptors(x, denormalize=True)
        return bb_feat, desc_raw

    def assemble_features(self, bb_feat: torch.Tensor, raw_desc: torch.Tensor) -> torch.Tensor:
        bb_z = self.bb_norm(bb_feat)
        desc_z = self.standardizer(raw_desc) * self.desc_weight
        return torch.cat([bb_z, desc_z], dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.assemble_features(*self.extract_raw_parts(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


def build_model(
    name: str = "resnet18",
    num_classes: int = 12,
    head_type: str = "linear",
    feature_norm: bool = False,
    pool_mode: str = "avg",
    feature_mode: str = "backbone",
    desc_weight: float = 1.0,
    feature_norm_mode: Optional[str] = None,
    feature_dropout: float = 0.0,
    **head_kwargs,
) -> nn.Module:
    """Build frozen-backbone, pure-descriptor, or hybrid-fusion classifier model."""
    n_cls = int(num_classes)
    if feature_mode == "backbone":
        model, in_features = build_backbone(name=name, pool_mode=pool_mode)
    elif feature_mode == "descriptor":
        model, in_features = DescriptorClassifierModel(desc_weight=desc_weight), 128
    elif feature_mode == "hybrid":
        backbone, bb_dim = build_backbone(name=name, pool_mode=pool_mode)
        model, in_features = HybridClassifierModel(backbone, desc_weight=desc_weight), bb_dim + 128
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode!r}")
    model.fc = build_head(
        in_features,
        n_cls,
        head_type=head_type,
        feature_norm=feature_norm,
        feature_norm_mode=feature_norm_mode,
        feature_dropout=feature_dropout,
        **head_kwargs,
    )
    return model


def model_from_checkpoint(ckpt: dict, config: dict, device: torch.device) -> nn.Module:
    """Reconstruct and load an eval-ready model from a saved checkpoint dictionary."""
    model = build_model(
        name=config.get("model", "resnet18"),
        num_classes=int(config["num_classes"]),
        head_type=ckpt.get("head_type", "linear"),
        feature_norm=bool(ckpt.get("feature_norm", False)),
        pool_mode=ckpt.get("pool_mode", "avg"),
        feature_mode=ckpt.get("feature_mode", "backbone"),
        desc_weight=float(ckpt.get("desc_weight", 1.0)),
        feature_norm_mode=ckpt.get("feature_norm_mode", None),
        feature_dropout=float(ckpt.get("feature_dropout", 0.0)),
        **dict(ckpt.get("head_kwargs", {})),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model
