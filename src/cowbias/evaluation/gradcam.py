"""Genuine binary-logit Grad-CAM with frozen weights and removable activation hook."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cowbias.training.reproducibility import state_sha256


def target_score(logits, target):
    if type(target) is not int or target not in (0, 1):
        raise ValueError("Target must be non_cow=0 or cow=1")
    return logits if target == 1 else -logits


def final_activation(model):
    modules = list(model.features.children())
    conv = [i for i, m in enumerate(modules) if isinstance(m, nn.Conv2d)]
    if not conv or conv[-1] + 1 >= len(modules) or not isinstance(modules[conv[-1] + 1], nn.ReLU):
        raise ValueError("Unsupported final convolution/activation structure")
    return modules[conv[-1] + 1]


def gradcam(model, image, target):
    target_score(torch.zeros(1), target)
    if image.shape != (1, 3, model.input_size, model.input_size):
        raise ValueError("Grad-CAM requires one matching image")
    if image.dtype != torch.float32 or not torch.isfinite(image).all():
        raise ValueError("Invalid Grad-CAM image")
    model.eval()
    model.requires_grad_(False)
    before = state_sha256(model.state_dict())
    captured = []
    hook = final_activation(model).register_forward_hook(
        lambda module, inputs, out: captured.append(out)
    )
    try:
        with torch.enable_grad():
            # Input gradient tracking is needed because parameter gradients are disabled.
            x = image.detach().clone().requires_grad_(True)
            logits = model(x)
            activation = captured[0]
            (gradients,) = torch.autograd.grad(target_score(logits, target).sum(), activation)
            if not torch.isfinite(gradients).all():
                raise ValueError("Nonfinite Grad-CAM gradients")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            heat = (weights * activation).sum(dim=1, keepdim=True).relu()
            heat = F.interpolate(heat, image.shape[-2:], mode="bilinear", align_corners=False)
            heat = heat[0, 0].detach()
            peak = float(heat.max())
            heat = heat / peak if peak > 0 else torch.zeros_like(heat)
            result = heat.cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite Grad-CAM heatmap")
            return result, {
                "target": target,
                "score": "logit" if target else "-logit",
                "gradient_l1": float(gradients.detach().abs().sum()),
                "raw_peak": peak,
                "zero_mass": not bool(result.sum()),
                "state_before": before,
                "state_after": state_sha256(model.state_dict()),
                "layer": next(n for n, m in model.named_modules() if m is final_activation(model)),
            }
    finally:
        hook.remove()
        if before != state_sha256(model.state_dict()):
            raise ValueError("Model state mutated during attribution")


def attribution(heat, mask):
    heat, mask = np.asarray(heat, dtype=np.float64), np.asarray(mask, dtype=np.float64)
    if (
        heat.ndim != 2
        or heat.shape != mask.shape
        or not np.isfinite(heat).all()
        or not np.isfinite(mask).all()
    ):
        raise ValueError("Invalid heatmap/mask dimensions or values")
    if np.any(heat < 0) or np.any(heat > 1) or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("Heatmap and mask must be in [0,1]")
    mass, area = float(heat.sum()), float(mask.mean())
    # Undefined localization for an exactly zero heatmap is represented by null, not a fake zero.
    fg = float((heat * mask).sum() / mass) if mass > 0 else None
    bg = float((heat * (1 - mask)).sum() / mass) if mass > 0 else None
    return {
        "heatmap_mass": mass,
        "foreground_attribution_share": fg,
        "background_attribution_share": bg,
        "foreground_area_fraction": area,
        "foreground_attribution_enrichment": fg / area if fg is not None and area > 0 else None,
    }
