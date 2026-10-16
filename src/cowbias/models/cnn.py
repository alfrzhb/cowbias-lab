"""One compact, untrained architecture shared by both experimental arms."""

import torch
from torch import nn

ARCHITECTURE_ID = "cowbias-small-cnn-v1"


class SmallCNN(nn.Module):
    """Three Conv/ReLU/MaxPool blocks, adaptive pooling and one binary logit."""

    def __init__(self, input_size: int = 32) -> None:
        super().__init__()
        if type(input_size) is not int or input_size < 8:
            raise ValueError("CNN input_size must be an integer >= 8")
        self.input_size = input_size
        layers: list[nn.Module] = []
        channels = 3
        for output_channels in (16, 32, 64):
            layers.extend(
                [nn.Conv2d(channels, output_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            )
            channels = output_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.input_size, self.input_size):
            raise ValueError(f"Expected [N, 3, {self.input_size}, {self.input_size}] input")
        if (
            images.shape[0] == 0
            or images.dtype != torch.float32
            or not torch.isfinite(images).all()
        ):
            raise ValueError("CNN input must be nonempty, finite float32 tensors")
        return self.classifier(self.features(images)).squeeze(1)

    def architecture(self) -> dict:
        return {
            "id": ARCHITECTURE_ID,
            "input_size": self.input_size,
            "channels": [16, 32, 64],
            "pool_size": 4,
            "hidden_size": 64,
            "output_logits": 1,
            "activation": "relu",
            "stochastic_layers": False,
        }


def model_from_metadata(architecture: dict) -> SmallCNN:
    if not isinstance(architecture, dict) or architecture.get("id") != ARCHITECTURE_ID:
        raise ValueError("Unsupported checkpoint architecture")
    model = SmallCNN(architecture.get("input_size"))
    if model.architecture() != architecture:
        raise ValueError("Incompatible checkpoint architecture parameters")
    return model
