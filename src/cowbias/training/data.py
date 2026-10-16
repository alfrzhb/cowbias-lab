"""Audited Stage 1 data adapted to deterministic PyTorch training inputs."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cowbias.config import ExperimentConfig
from cowbias.data.audit import audit_dataset
from cowbias.data.dataset import ManifestDataset
from cowbias.data.pipeline import sha256_file

PREPROCESSING = {
    "layout": "CHW",
    "dtype": "float32",
    "resize": "Pillow bilinear",
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "augmentation": "none",
}


def identity_sha256(identities: list[str]) -> str:
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class AuditedData:
    root: Path
    config: ExperimentConfig
    seed: int
    manifest_sha256: str
    config_sha256: str
    report: dict

    @classmethod
    def open(cls, root: str | Path, config: ExperimentConfig, seed: int) -> "AuditedData":
        root = Path(root).resolve()
        report = audit_dataset(root, config, seed=seed)
        if report["status"] != "PASS" or not report["determinism"]["record_assignments_replayed"]:
            raise ValueError("Training requires a successful current Stage 1 audit")
        return cls(
            root,
            config,
            seed,
            report["manifest_sha256"],
            sha256_file(root / "dataset_config.json"),
            report,
        )

    def assert_unchanged(self) -> None:
        if sha256_file(self.root / "manifest.csv") != self.manifest_sha256:
            raise ValueError("Dataset manifest changed after audit")
        if sha256_file(self.root / "dataset_config.json") != self.config_sha256:
            raise ValueError("Dataset configuration changed after audit")


class TorchManifestDataset(Dataset):
    """Train/validation-only adapter. Background metadata never enters the CNN."""

    def __init__(self, audited: AuditedData, *, split: str, variant: str) -> None:
        if split not in ("train", "val"):
            raise ValueError(
                "Stage 2 forbids test-set model access; only train and val are allowed"
            )
        if (split == "val") != (variant == "shared_eval"):
            raise ValueError("Validation must be shared_eval; training requires its own variant")
        if variant not in ("baseline", "treatment", "shared_eval"):
            raise ValueError("Unsupported training variant")
        audited.assert_unchanged()
        self.audited = audited
        self.split, self.variant = split, variant
        self.source = ManifestDataset(
            audited.root,
            split=split,
            variant=variant,
            image_size=audited.config.dataset.model_image_size,
        )
        self.records = self.source.records
        samples = [self.source[i] for i in range(len(self.source))]
        self.images = torch.stack(
            [
                torch.from_numpy(s["image"].astype(np.float32))
                .permute(2, 0, 1)
                .contiguous()
                .div(255)
                .sub(0.5)
                .div(0.5)
                for s in samples
            ]
        )
        self.targets = torch.tensor([s["label"] for s in samples], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        return {
            "image": self.images[index],
            "target": self.targets[index],
            **{
                key: row[key]
                for key in (
                    "image_id",
                    "foreground_id",
                    "object_label",
                    "background_label",
                    "split",
                    "variant",
                )
            },
        }


def make_loader(
    dataset: TorchManifestDataset, seed: int, batch_size: int, *, shuffle: bool
) -> DataLoader:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if dataset.split == "val" and shuffle:
        raise ValueError("Shared validation ordering must not be shuffled")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
