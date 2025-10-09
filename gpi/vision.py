"""Vision model helpers for the GPI vision policy."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision


def _build_resnet(name: str = "resnet18") -> nn.Module:
    model_fn = getattr(torchvision.models, name)
    model = model_fn(weights=None)
    model.fc = nn.Identity()
    return model


def _replace_bn_with_gn(module: nn.Module, features_per_group: int = 16) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(max(1, child.num_features // features_per_group), child.num_features)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, features_per_group)
    return module


def create_models(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    vision_encoder = _build_resnet("resnet18")
    vision_encoder = _replace_bn_with_gn(vision_encoder)
    vision_encoder = vision_encoder.to(device)
    state_predictor = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
    ).to(device)
    for module in state_predictor:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return vision_encoder, state_predictor


def load_models(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    vision_encoder, state_predictor = create_models(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vision_encoder.load_state_dict(checkpoint["vision_encoder"])
    state_predictor.load_state_dict(checkpoint["state_predictor"])
    vision_encoder.eval()
    state_predictor.eval()
    return vision_encoder, state_predictor


def predict_object_state(image: np.ndarray | torch.Tensor, vision_encoder: nn.Module, state_predictor: nn.Module, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                if image.shape[0] != 3:
                    image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
            else:
                raise ValueError(f"Unexpected image shape {image.shape}")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        features = vision_encoder(image)
        prediction = state_predictor(features)
    return prediction.squeeze(0).cpu().numpy().astype(np.float32)


__all__ = ["create_models", "load_models", "predict_object_state"]
