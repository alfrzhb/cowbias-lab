"""Reject the wrong session/seed/configuration before Stage 2 consumes data."""

import hashlib
import json
import shutil
from dataclasses import replace

import pytest

from cowbias.data.audit import AuditError, audit_dataset
from cowbias.data.pipeline import create_contact_sheet


def test_wrong_seed_and_configuration_are_rejected(generated, small_config, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(generated, root)
    with pytest.raises(AuditError, match="seed is invalid or differs"):
        audit_dataset(root, small_config, seed=123)
    incompatible = replace(small_config, dataset=replace(small_config.dataset, image_size=48))
    with pytest.raises(AuditError, match="configuration differs"):
        audit_dataset(root, incompatible, seed=42)
    assert audit_dataset(root, small_config, seed=42)["determinism"]["record_assignments_replayed"]


@pytest.mark.parametrize("corruption", ["missing_snapshot", "generator_version", "data_origin"])
def test_incompatible_artifact_directories(generated, tmp_path, corruption):
    root = tmp_path / corruption
    shutil.copytree(generated, root)
    snapshot_path = root / "dataset_config.json"
    if corruption == "missing_snapshot":
        snapshot_path.unlink()
    else:
        snapshot = json.loads(snapshot_path.read_text())
        snapshot[corruption] = "incompatible"
        snapshot_path.write_text(json.dumps(snapshot))
    with pytest.raises(AuditError):
        audit_dataset(root)
    assert json.loads((root / "audit.json").read_text())["status"] == "FAIL"


def test_contact_sheet_does_not_modify_exported_images(generated):
    def image_checksums():
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (generated / "images").glob("*.png")
        }

    before = image_checksums()
    create_contact_sheet(generated)
    assert image_checksums() == before
