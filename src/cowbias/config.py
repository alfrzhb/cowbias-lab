"""Validated configuration for the controlled synthetic experiment."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The experiment configuration violates its scientific contract."""


@dataclass(frozen=True)
class DatasetConfig:
    image_size: int
    model_image_size: int
    train_size: int
    val_size: int
    test_size: int
    baseline_majority_ratio: float
    treatment_background_ratio: float
    negative_class: str

    def __post_init__(self) -> None:
        for name in ("image_size", "model_image_size", "train_size", "val_size", "test_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"dataset.{name} must be a positive integer")
        if not 24 <= self.image_size <= 1024:
            raise ConfigError("dataset.image_size must be between 24 and 1024")
        if not 8 <= self.model_image_size <= self.image_size:
            raise ConfigError("dataset.model_image_size must be between 8 and image_size")
        for name in ("train_size", "val_size", "test_size"):
            if getattr(self, name) % 4:
                raise ConfigError(f"dataset.{name} must be divisible by four for balanced groups")
        for name in ("baseline_majority_ratio", "treatment_background_ratio"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ConfigError(f"dataset.{name} must be a finite ratio between 0 and 1")
        if not 0.5 < self.baseline_majority_ratio < 1:
            raise ConfigError("dataset.baseline_majority_ratio must be > 0.5 and < 1")
        majority = self.train_size / 2 * self.baseline_majority_ratio
        if abs(majority - round(majority)) > 1e-8:
            raise ConfigError("train_size and baseline_majority_ratio must yield integral counts")
        if self.treatment_background_ratio != 0.5:
            raise ConfigError("dataset.treatment_background_ratio must be 0.5 per label")
        if self.negative_class != "horse":
            raise ConfigError("dataset.negative_class supports only 'horse' (label 'non_cow')")


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    device: str

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, (list, tuple)) or not self.seeds:
            raise ConfigError("training.seeds must be a nonempty list of integers")
        if any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds):
            raise ConfigError("training.seeds must contain integers in [0, 2**32)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("training.seeds must be unique")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ConfigError(f"training.{name} must be a positive integer")
        if self.optimizer != "adamw":
            raise ConfigError("training.optimizer supports only 'adamw'")
        if type(self.learning_rate) not in (int, float) or not 0 < self.learning_rate < 1:
            raise ConfigError("training.learning_rate must be finite and between 0 and 1")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ConfigError("training.device must be auto, cpu, cuda, or mps")


@dataclass(frozen=True)
class ArtifactsConfig:
    root: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root.strip():
            raise ConfigError("artifacts.root must be a nonempty path string")


@dataclass(frozen=True)
class ExperimentConfig:
    project: str
    dataset: DatasetConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig

    def __post_init__(self) -> None:
        if self.project != "cowbias-lab":
            raise ConfigError("project must be 'cowbias-lab'")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(raw: Any) -> ExperimentConfig:
    """Reject missing/unknown fields rather than silently applying defaults."""
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a YAML mapping")
    if set(raw) != {"project", "dataset", "training", "artifacts"}:
        raise ConfigError("Expected root fields: project, dataset, training, artifacts")
    try:
        return ExperimentConfig(
            project=raw["project"],
            dataset=DatasetConfig(**raw["dataset"]),
            training=TrainingConfig(**raw["training"]),
            artifacts=ArtifactsConfig(**raw["artifacts"]),
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Missing or unsupported configuration field: {exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    return config_from_dict(raw)
