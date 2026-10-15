import json
import shutil
from copy import deepcopy

import pytest
from PIL import Image

from cowbias.data.audit import AuditError, audit_dataset, check_no_leakage
from cowbias.data.dataset import MANIFEST_FIELDS, artifact_path, read_csv, write_csv


@pytest.mark.parametrize(
    "field",
    ["foreground_id", "background_asset_id", "foreground_family_id", "background_family_id"],
)
def test_source_isolation_and_injected_leakage(generated, field):
    rows = read_csv(generated / "manifest.csv", MANIFEST_FIELDS)
    assert check_no_leakage(rows)["cross_split_overlaps"] == 0
    corrupted = deepcopy(rows)
    train = next(r for r in corrupted if r["split"] == "train")
    test = next(r for r in corrupted if r["split"] == "test")
    test[field] = train[field]
    with pytest.raises(AuditError, match=f"Cross-split leakage in {field}"):
        check_no_leakage(corrupted)


def test_audit_rejects_injected_leakage_and_invalidates_stale_pass(generated, tmp_path):
    root = tmp_path / "injected"
    shutil.copytree(generated, root)
    rows = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    next(r for r in rows if r["split"] == "test")["foreground_id"] = rows[0]["foreground_id"]
    write_csv(root / "manifest.csv", rows)
    with pytest.raises(AuditError, match="Cross-split leakage"):
        audit_dataset(root)
    assert json.loads((root / "audit.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_pair", "requires pair_id"),
        ("duplicate_id", "Duplicate composite image_id"),
        ("wrong_group", "Dataset group counts differ"),
        ("one_member_pair", "number of counterfactual pairs"),
        ("corrupt_image", "Unreadable or invalid artifact"),
        ("wrong_mask_size", "dimensions"),
        ("changed_geometry", "geometry/mask mismatch"),
        ("changed_foreground", "Composite differs"),
    ],
)
def test_audit_rejects_corruption(generated, tmp_path, mutation, message):
    root = tmp_path / mutation
    shutil.copytree(generated, root)
    rows = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    test = next(r for r in rows if r["split"] == "test")
    if mutation == "missing_pair":
        test["pair_id"] = ""
    elif mutation == "duplicate_id":
        rows[1]["image_id"] = rows[0]["image_id"]
    elif mutation == "wrong_group":
        rows[0]["background_label"] = "beach" if rows[0]["background_label"] == "grass" else "grass"
    elif mutation == "one_member_pair":
        test["pair_id"] = "orphan"
    elif mutation == "corrupt_image":
        (root / test["file_path"]).write_bytes(b"not-an-image")
    elif mutation == "wrong_mask_size":
        Image.new("L", (1, 1)).save(root / test["mask_path"])
    elif mutation == "changed_geometry":
        Image.new("L", (32, 32)).save(root / test["mask_path"])
    elif mutation == "changed_foreground":
        Image.new("RGB", (32, 32), "red").save(root / test["file_path"])
    write_csv(root / "manifest.csv", rows)
    with pytest.raises(AuditError, match=message):
        audit_dataset(root)


def test_catalog_family_alias_is_rejected(generated):
    rows = read_csv(generated / "manifest.csv", MANIFEST_FIELDS)
    catalog = [
        {"source_id": "a", "family_id": "original", "sha256": "x", "split": "train"},
        {"source_id": "b", "family_id": "original", "sha256": "y", "split": "test"},
    ]
    with pytest.raises(AuditError, match="source catalog leakage in family_id"):
        check_no_leakage(rows, catalog)


def test_identical_asset_alias_is_rejected(generated):
    rows = read_csv(generated / "manifest.csv", MANIFEST_FIELDS)
    catalog = [
        {"source_id": "a", "family_id": "a", "sha256": "same", "split": "train"},
        {"source_id": "b", "family_id": "b", "sha256": "same", "split": "test"},
    ]
    with pytest.raises(AuditError, match="source catalog leakage in sha256"):
        check_no_leakage(rows, catalog)


@pytest.mark.parametrize("path", ["../secret.png", "/etc/passwd", ""])
def test_manifest_paths_cannot_escape(generated, path):
    with pytest.raises(ValueError, match="Invalid relative"):
        artifact_path(generated, path)
