"""Assign original source identities to splits before rendering or compositing."""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from cowbias.config import DatasetConfig

OBJECTS = ("cow", "non_cow")
BACKGROUNDS = ("grass", "beach")
SPLITS = ("train", "val", "test")


def keyed_seed(seed: int, key: str) -> int:
    """Stable independent streams; never use Python's process-randomized hash()."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Source:
    source_id: str
    family_id: str
    kind: str
    label: str
    split: str
    seed: int
    # Background asset association is assigned without considering the object label.
    foreground_id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def allocate_sources(config: DatasetConfig, seed: int) -> list[Source]:
    """Independent procedural originals; each family has exactly one source asset.

    Global renderer primitives are shared code, not finite reusable texture/animal
    templates. Any future source derivatives must inherit their original family ID.
    """
    rng = np.random.default_rng(keyed_seed(seed, "source-allocation"))
    sizes = (config.train_size, config.val_size, config.test_size // 2)
    foregrounds = []
    for label in OBJECTS:
        count = sum(sizes) // 2
        prefix = "f" if label == "cow" else "h"
        ids = [f"s{seed}-{prefix}-{i:05d}" for i in rng.permutation(count)]
        offset = 0
        for split, size in zip(SPLITS, sizes, strict=True):
            for source_id in ids[offset : offset + size // 2]:
                foregrounds.append(Source(source_id, source_id, "foreground", label, split, seed))
            offset += size // 2
    backgrounds = []
    for i, fg in enumerate(foregrounds):
        for label in BACKGROUNDS:
            source_id = f"s{seed}-bg-{i:05d}-{label}"
            backgrounds.append(
                Source(source_id, source_id, "background", label, fg.split, seed, fg.source_id)
            )
    return foregrounds + backgrounds


def background_assignments(
    sources: list[Source], config: DatasetConfig, seed: int
) -> list[tuple[Source, str, str, str]]:
    """Return (foreground, background label, variant, pair ID) with exact quotas."""
    records = []
    for split in SPLITS:
        for label in OBJECTS:
            group = sorted(
                (
                    s
                    for s in sources
                    if s.kind == "foreground" and s.split == split and s.label == label
                ),
                key=lambda s: s.source_id,
            )
            for variant in ("baseline", "treatment") if split == "train" else ("shared_eval",):
                rng = np.random.default_rng(keyed_seed(seed, f"assign:{split}:{label}:{variant}"))
                order = rng.permutation(len(group))
                if split == "test":
                    for index in order:
                        fg = group[index]
                        for bg in BACKGROUNDS:
                            records.append((fg, bg, variant, f"pair-{fg.source_id}"))
                    continue
                grass_count = len(group) // 2
                if variant == "baseline":
                    majority = round(len(group) * config.baseline_majority_ratio)
                    grass_count = majority if label == "cow" else len(group) - majority
                for rank, index in enumerate(order):
                    records.append(
                        (group[index], "grass" if rank < grass_count else "beach", variant, "")
                    )
    return records
