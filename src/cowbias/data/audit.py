"""Fail-closed scientific invariants, file integrity and deterministic source replay."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from cowbias.config import ExperimentConfig, config_from_dict
from cowbias.data.dataset import MANIFEST_FIELDS, artifact_path, read_csv, write_csv
from cowbias.data.generate import GENERATOR_VERSION, compose, render_background, render_foreground
from cowbias.data.splits import BACKGROUNDS, OBJECTS, SPLITS, Source, allocate_sources

SOURCE_FIELDS = (
    "source_id",
    "family_id",
    "kind",
    "label",
    "split",
    "seed",
    "foreground_id",
    "file_path",
    "sha256",
)


class AuditError(ValueError):
    """Dataset cannot be used until the reported invariant violation is resolved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def check_no_leakage(records: list[dict], catalog: list[dict] | None = None) -> dict:
    """Cross-variant reuse within train is intentional; cross-split reuse is forbidden."""
    counts = {}
    for field in (
        "foreground_id",
        "background_asset_id",
        "foreground_family_id",
        "background_family_id",
    ):
        owners: dict[str, set[str]] = defaultdict(set)
        for row in records:
            require(bool(row.get(field)), f"Missing source identity: {field}")
            owners[row[field]].add(row["split"])
        leaked = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
        require(not leaked, f"Cross-split leakage in {field}: {leaked}")
        counts[field] = {
            split: sum(split in splits for splits in owners.values()) for split in SPLITS
        }
    if catalog is not None:
        for field in ("source_id", "family_id", "sha256"):
            owners = defaultdict(set)
            for row in catalog:
                owners[row[field]].add(row["split"])
            leaked = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
            require(not leaked, f"Cross-split source catalog leakage in {field}: {leaked}")
    return {"cross_split_overlaps": 0, "unique_identities_by_split": counts}


def _open(root: Path, relative: str, mode: str, size: int) -> Image.Image:
    try:
        with Image.open(artifact_path(root, relative)) as image:
            image.load()
            require(image.format == "PNG", f"Not a PNG file: {relative}")
            require(image.mode == mode, f"Unexpected image mode in {relative}: {image.mode}")
            require(
                image.size == (size, size), f"Unexpected dimensions in {relative}: {image.size}"
            )
            return image.copy()
    except (OSError, ValueError) as exc:
        raise AuditError(f"Unreadable or invalid artifact {relative}: {exc}") from exc


def _audit(root: Path, config: ExperimentConfig, seed: int) -> dict:
    records = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    catalog = read_csv(root / "sources.csv", SOURCE_FIELDS)
    leakage = check_no_leakage(records, catalog)
    require(len({r["image_id"] for r in records}) == len(records), "Duplicate composite image_id")
    require(len({r["file_path"] for r in records}) == len(records), "Duplicate composite file_path")
    require(len({r["mask_path"] for r in records}) == len(records), "Duplicate composite mask_path")
    require(len({r["source_id"] for r in catalog}) == len(catalog), "Duplicate source_id")
    require(len({r["file_path"] for r in catalog}) == len(catalog), "Duplicate source file_path")
    for row in records:
        require(row["object_label"] in OBJECTS, f"Unsupported object label: {row['object_label']}")
        require(row["background_label"] in BACKGROUNDS, "Unsupported background label")
        require(row["split"] in SPLITS, "Unsupported split")
        require(row["variant"] in ("baseline", "treatment", "shared_eval"), "Unsupported variant")
        require(row["seed"] == str(seed), "Manifest seed does not match experiment seed")
        require(
            (row["split"] == "train") == (row["variant"] != "shared_eval"),
            "Training variants must use train; held-out splits must use shared_eval",
        )
        require(
            (row["split"] == "test") == bool(row["pair_id"]),
            "Test requires pair_id; train and unpaired validation require empty pair_id",
        )
    d = config.dataset
    half, majority = d.train_size // 2, round(d.train_size / 2 * d.baseline_majority_ratio)
    baseline = {
        "cow/grass": majority,
        "cow/beach": half - majority,
        "non_cow/grass": half - majority,
        "non_cow/beach": majority,
    }
    expected = {
        "train/baseline": baseline,
        "train/treatment": {f"{o}/{b}": d.train_size // 4 for o in OBJECTS for b in BACKGROUNDS},
        "val/shared_eval": {f"{o}/{b}": d.val_size // 4 for o in OBJECTS for b in BACKGROUNDS},
        "test/shared_eval": {f"{o}/{b}": d.test_size // 4 for o in OBJECTS for b in BACKGROUNDS},
    }
    actual: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        actual[f"{row['split']}/{row['variant']}"][
            f"{row['object_label']}/{row['background_label']}"
        ] += 1
    require(
        dict(actual) == expected,
        f"Dataset group counts differ: expected {expected}, got {dict(actual)}",
    )
    distribution_rows = [
        {
            "dataset": name,
            "object_label": group.split("/")[0],
            "background_label": group.split("/")[1],
            "count": count,
        }
        for name, groups in actual.items()
        for group, count in sorted(groups.items())
    ]
    training = {}
    for variant in ("baseline", "treatment"):
        ids = [r["foreground_id"] for r in records if r["variant"] == variant]
        require(len(set(ids)) == d.train_size, f"{variant}: training foregrounds must occur once")
        training[variant] = set(ids)
    require(
        training["baseline"] == training["treatment"],
        "Baseline/treatment training foreground identity sets differ",
    )
    val_ids = [r["foreground_id"] for r in records if r["split"] == "val"]
    require(
        len(set(val_ids)) == d.val_size,
        "Validation foregrounds must be independent unique instances",
    )
    # Match the persisted pre-compositing allocation against the configured seed.
    expected_sources = {s.source_id: s for s in allocate_sources(d, seed)}
    allocation = json.loads((root / "source_allocation.json").read_text())
    require(
        allocation == [s.as_dict() for s in allocate_sources(d, seed)],
        "Pre-compositing source allocation differs from deterministic replay",
    )
    require(
        set(expected_sources) == {r["source_id"] for r in catalog},
        "Source catalog identities differ from deterministic allocation",
    )
    images, source_lookup = {}, {}
    size = d.image_size
    for row in catalog:
        source = expected_sources[row["source_id"]]
        require(
            all(str(value) == row[key] for key, value in source.as_dict().items()),
            f"Source metadata differs from frozen allocation: {source.source_id}",
        )
        mode = "RGBA" if source.kind == "foreground" else "RGB"
        actual_image = _open(root, row["file_path"], mode, size)
        file_hash = hashlib.sha256(artifact_path(root, row["file_path"]).read_bytes()).hexdigest()
        require(file_hash == row["sha256"], f"Source checksum mismatch: {source.source_id}")
        renderer = render_foreground if source.kind == "foreground" else render_background
        regenerated = renderer(source, size)
        require(
            np.array_equal(actual_image, regenerated),
            f"Deterministic source replay failed: {source.source_id}",
        )
        images[source.source_id] = actual_image
        source_lookup[source.source_id] = source
    pairs: dict[str, list[tuple[dict, np.ndarray, np.ndarray]]] = defaultdict(list)
    for row in records:
        fg_id, bg_id = row["foreground_id"], row["background_asset_id"]
        require(
            fg_id in source_lookup and bg_id in source_lookup, "Manifest references unknown source"
        )
        fg: Source = source_lookup[fg_id]
        bg: Source = source_lookup[bg_id]
        require(fg.kind == "foreground" and bg.kind == "background", "Source kind mismatch")
        require(fg.split == bg.split == row["split"], "Manifest/source split mismatch")
        require(
            fg.label == row["object_label"] and bg.label == row["background_label"],
            "Manifest/source label mismatch",
        )
        require(
            fg.family_id == row["foreground_family_id"]
            and bg.family_id == row["background_family_id"],
            "Manifest/source family mismatch",
        )
        require(bg.foreground_id == fg_id, "Background asset association differs from allocation")
        image = _open(root, row["file_path"], "RGB", size)
        mask = _open(root, row["mask_path"], "L", size)
        alpha = np.array(images[fg_id].getchannel("A"))
        require(
            np.array_equal(mask, alpha), f"Foreground geometry/mask mismatch: {row['image_id']}"
        )
        require(
            bool(np.any(alpha == 255)) and bool(np.any(alpha == 0)),
            f"Empty or fully occluding foreground mask: {row['image_id']}",
        )
        require(
            np.array_equal(image, compose(images[fg_id], images[bg_id])),
            f"Composite differs from original foreground/background: {row['image_id']}",
        )
        if row["split"] == "test":
            pairs[row["pair_id"]].append((row, np.array(image), np.array(mask)))
    require(len(pairs) == d.test_size // 2, "Incorrect number of counterfactual pairs")
    test_foregrounds = set()
    for pair_id, members in pairs.items():
        require(len(members) == 2, f"Pair {pair_id} must contain exactly two images")
        a, b = members
        require(
            a[0]["foreground_id"] == b[0]["foreground_id"]
            and a[0]["object_label"] == b[0]["object_label"],
            f"Pair foreground differs: {pair_id}",
        )
        require(
            {a[0]["background_label"], b[0]["background_label"]} == set(BACKGROUNDS),
            f"Pair requires grass and beach: {pair_id}",
        )
        require(
            a[0]["foreground_id"] not in test_foregrounds,
            f"Foreground appears in multiple test pairs: {pair_id}",
        )
        test_foregrounds.add(a[0]["foreground_id"])
        require(np.array_equal(a[2], b[2]), f"Pair mask differs: {pair_id}")
        opaque = a[2] == 255
        require(
            np.array_equal(a[1][opaque], b[1][opaque]), f"Pair foreground pixels differ: {pair_id}"
        )
        require(
            not np.array_equal(a[1][a[2] == 0], b[1][b[2] == 0]),
            f"Pair background did not change: {pair_id}",
        )
    write_csv(root / "group_distribution.csv", distribution_rows)
    table = "| Dataset | cow/grass | cow/beach | non_cow/grass | non_cow/beach |\n"
    table += "|---|---:|---:|---:|---:|\n"
    for name, groups in actual.items():
        table += (
            f"| {name} | "
            + " | ".join(str(groups[f"{o}/{b}"]) for o in OBJECTS for b in BACKGROUNDS)
            + " |\n"
        )
    (root / "group_distribution.md").write_text(table)
    return {
        "status": "PASS",
        "data_origin": "synthetic",
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "generator_parameters": config.as_dict()["dataset"],
        "counts": {name: sum(groups.values()) for name, groups in actual.items()},
        "groups": {name: dict(groups) for name, groups in actual.items()},
        "class_balance": {
            name: {
                label: sum(n for g, n in groups.items() if g.startswith(label + "/"))
                for label in OBJECTS
            }
            for name, groups in actual.items()
        },
        "leakage": leakage,
        "source_assets": len(catalog),
        "unique_foregrounds": sum(s.kind == "foreground" for s in source_lookup.values()),
        "unique_background_assets": sum(s.kind == "background" for s in source_lookup.values()),
        "intentional_training_foreground_reuse": d.train_size,
        "counterfactual": {
            "pairs": len(pairs),
            "unique_foregrounds": len(test_foregrounds),
            "identical_masks": True,
            "identical_foreground_geometry": True,
            "identical_opaque_foreground_pixels": True,
            "full_composite_reconstruction": True,
        },
        "determinism": {"source_allocation_replayed": True, "all_source_pixels_regenerated": True},
        "readability": {
            "source_files": len(catalog),
            "image_files": len(records),
            "mask_files": len(records),
            "image_size": size,
        },
        "manifest_sha256": hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest(),
        "limitations": [
            "Procedural illustrations, not real photographs",
            "Soft-alpha boundary pixels depend on the background by design",
            "Shared renderer grammar does not test unseen species or rendering styles",
        ],
    }


def audit_dataset(
    root: str | Path, config: ExperimentConfig | None = None, seed: int | None = None
) -> dict:
    """Re-audit existing files; overwrite stale PASS with FAIL if validation fails."""
    root = Path(root)
    try:
        snapshot = json.loads((root / "dataset_config.json").read_text())
        saved_config = config_from_dict(snapshot["config"])
        if config is None:
            config = saved_config
        require(
            config == saved_config,
            "Provided configuration differs from saved dataset configuration",
        )
        seed = snapshot["seed"] if seed is None else seed
        require(
            type(seed) is int and 0 <= seed < 2**32 and seed == snapshot["seed"],
            "Dataset seed is invalid or differs from snapshot",
        )
        require(
            snapshot["data_origin"] == "synthetic"
            and snapshot["generator_version"] == GENERATOR_VERSION,
            "Generator version/origin mismatch",
        )
        report = _audit(root, config, seed)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"status": "FAIL", "error": str(exc)}
        if root.is_dir():
            (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        raise AuditError(f"Dataset audit failed: {exc}") from exc
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
