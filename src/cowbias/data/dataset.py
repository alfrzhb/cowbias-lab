"""Manifest I/O and framework-independent dataset access for Stage 1 inspection."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MANIFEST_FIELDS = (
    "image_id",
    "foreground_id",
    "background_asset_id",
    "object_label",
    "background_label",
    "split",
    "variant",
    "pair_id",
    "seed",
    "file_path",
    "mask_path",
    "foreground_family_id",
    "background_family_id",
)


def artifact_path(root: Path, relative: str) -> Path:
    """Reject absolute, missing, traversal, and symlink escapes in manifests."""
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid relative artifact path: {relative!r}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes dataset root: {relative!r}")
    if not result.is_file():
        raise ValueError(f"Artifact file missing: {relative}")
    return result


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not set(required) <= set(reader.fieldnames):
                raise ValueError(f"{path.name}: required columns are {', '.join(required)}")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path.name}: empty or malformed CSV records")
    return rows


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to export empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ManifestDataset:
    """Return RGB uint8 HWC pixels, uint8 mask, labels and record metadata.

    Filtering is explicit: validation/test use variant='shared_eval'. The original
    48px image is returned unless a model input size is requested. No training,
    normalization or random augmentation is performed here.
    """

    def __init__(
        self, root: str | Path, *, split: str, variant: str, image_size: int | None = None
    ) -> None:
        self.root = Path(root)
        if split not in ("train", "val", "test") or variant not in (
            "baseline",
            "treatment",
            "shared_eval",
        ):
            raise ValueError("Unsupported dataset split or variant")
        if image_size is not None and (type(image_size) is not int or image_size <= 0):
            raise ValueError("image_size must be a positive integer")
        self.image_size = image_size
        self.records = [
            r
            for r in read_csv(self.root / "manifest.csv", MANIFEST_FIELDS)
            if r["split"] == split and r["variant"] == variant
        ]
        if not self.records:
            raise ValueError(f"No records for split={split}, variant={variant}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(artifact_path(self.root, record["file_path"])) as handle:
            image = handle.convert("RGB")
        with Image.open(artifact_path(self.root, record["mask_path"])) as handle:
            mask = handle.convert("L")
        if self.image_size is not None:
            dims = (self.image_size, self.image_size)
            image = image.resize(dims, Image.Resampling.BILINEAR)
            mask = mask.resize(dims, Image.Resampling.NEAREST)
        return {
            "image": np.array(image),
            "mask": np.array(mask),
            "label": int(record["object_label"] == "cow"),
            "metadata": record.copy(),
        }
