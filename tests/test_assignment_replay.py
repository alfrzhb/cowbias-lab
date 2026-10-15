"""Realistic, distribution-preserving corruption of actual generated datasets."""

import json
import shutil
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from cowbias.data.audit import AuditError, audit_dataset
from cowbias.data.dataset import MANIFEST_FIELDS, read_csv, write_csv
from cowbias.data.generate import compose


def _groups(rows):
    return Counter(
        (r["split"], r["variant"], r["object_label"], r["background_label"]) for r in rows
    )


@pytest.mark.parametrize("recomposite", [False, True])
def test_quota_preserving_background_swap(generated, tmp_path, recomposite):
    root = tmp_path / "swapped"
    shutil.copytree(generated, root)
    rows = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    before = _groups(rows)
    selected = [
        next(
            r
            for r in rows
            if r["variant"] == "baseline"
            and r["object_label"] == "cow"
            and r["background_label"] == bg
        )
        for bg in ("grass", "beach")
    ]
    sources = read_csv(root / "sources.csv", ("source_id", "kind", "file_path"))
    by_id = {s["source_id"]: s for s in sources}
    by_foreground = {
        (s["foreground_id"], s["label"]): s for s in sources if s["kind"] == "background"
    }
    for row in selected:
        category = "beach" if row["background_label"] == "grass" else "grass"
        bg = by_foreground[row["foreground_id"], category]
        row["background_label"] = category
        row["background_asset_id"] = bg["source_id"]
        row["background_family_id"] = bg["family_id"]
        if recomposite:
            with (
                Image.open(root / by_id[row["foreground_id"]]["file_path"]) as fg_image,
                Image.open(root / bg["file_path"]) as bg_image,
            ):
                expected = compose(fg_image, bg_image)
                expected.save(root / row["file_path"])
            with Image.open(root / row["file_path"]) as actual:
                assert np.array_equal(actual, expected)
    assert _groups(rows) == before  # Every 97:3 and 50:50 quota is preserved.
    write_csv(root / "manifest.csv", rows)
    # Replay runs after ALL previous source, image, geometry and pair checks.
    # Reaching its error proves a correctly recomposited swap passed those checks.
    message = (
        "Deterministic assignment replay.*background_label" if recomposite else "Composite differs"
    )
    with pytest.raises(AuditError, match=message):
        audit_dataset(root)
    assert json.loads((root / "audit.json").read_text())["status"] == "FAIL"


def test_another_valid_background_asset_is_not_the_expected_source(generated, tmp_path):
    root = tmp_path / "wrong-asset"
    shutil.copytree(generated, root)
    rows = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    row = rows[0]
    donor = next(
        r
        for r in rows
        if r["split"] == row["split"]
        and r["background_label"] == row["background_label"]
        and r["foreground_id"] != row["foreground_id"]
    )
    row["background_asset_id"] = donor["background_asset_id"]
    row["background_family_id"] = donor["background_family_id"]
    sources = {s["source_id"]: s for s in read_csv(root / "sources.csv", ("source_id",))}
    with (
        Image.open(root / sources[row["foreground_id"]]["file_path"]) as fg,
        Image.open(root / sources[row["background_asset_id"]]["file_path"]) as bg,
    ):
        compose(fg, bg).save(root / row["file_path"])
    write_csv(root / "manifest.csv", rows)
    with pytest.raises(AuditError, match="Background asset association differs"):
        audit_dataset(root)


@pytest.mark.parametrize(
    "mutation", ["pair_identity", "pair_order", "train_order", "image_identity"]
)
def test_structurally_valid_seed_inconsistent_records(generated, tmp_path, mutation):
    root = tmp_path / mutation
    shutil.copytree(generated, root)
    rows = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    before = _groups(rows)
    if mutation == "pair_identity":
        pair = next(r["pair_id"] for r in rows if r["split"] == "test")
        for row in rows:
            if row["pair_id"] == pair:
                row["pair_id"] = "renamed-valid-two-member-pair"
    elif mutation == "pair_order":
        indices = [i for i, row in enumerate(rows) if row["split"] == "test"][:2]
        a, b = indices
        rows[a], rows[b] = rows[b], rows[a]
    elif mutation == "train_order":
        # Both variants retain matching foreground order, but it is not the seeded order.
        for variant in ("baseline", "treatment"):
            a, b = [i for i, row in enumerate(rows) if row["variant"] == variant][:2]
            rows[a], rows[b] = rows[b], rows[a]
    else:
        rows[0]["image_id"] = "unique-but-not-generated-image-id"
    assert _groups(rows) == before
    write_csv(root / "manifest.csv", rows)
    field = "pair_id" if mutation == "pair_identity" else "image_id"
    with pytest.raises(AuditError, match=f"Deterministic assignment replay.*{field}"):
        audit_dataset(root)
