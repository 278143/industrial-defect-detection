"""Anomalib PatchCore with a torchvision ImageNet-pretrained ResNet-18 backbone."""

import torch
from torchvision.models import ResNet18_Weights, resnet18


def create_model(config, cache_dir, pretrained=True):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    if config["backbone"] != "resnet18":
        raise ValueError("This baseline currently supports resnet18 only")
    torch.hub.set_dir(str(cache_dir))
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    return PatchcoreModel(
        backbone=backbone,
        layers=config["layers"],
        pre_trained=False,
        num_neighbors=config["num_neighbors"],
    )
