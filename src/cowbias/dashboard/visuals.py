"""Render saved maps without changing scientific arrays."""

import io
import json

import numpy as np
from matplotlib import colormaps
from PIL import Image, PngImagePlugin


def overlay(image, heat):
    if not heat.any():
        return image.copy()
    colors = colormaps["inferno"](heat)[..., :3]
    return np.rint((image.astype(float) / 255 * 0.5 + colors * 0.5) * 255).astype(np.uint8)


def selected_png(image, heat, metadata):
    buffer = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("CowBias provenance", json.dumps(metadata, sort_keys=True))
    Image.fromarray(overlay(image, heat)).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def attribution_text(value, percent=False):
    return (
        "Undefined (zero attribution mass)"
        if value is None
        else (f"{value:.2%}" if percent else f"{value:.4f}")
    )
