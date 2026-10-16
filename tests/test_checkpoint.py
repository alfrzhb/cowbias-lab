import json
import shutil

import pytest

torch = pytest.importorskip("torch")

from cowbias.data.pipeline import sha256_file
from cowbias.training.checkpoint import load_checkpoint, verify_training_run


def test_real_checkpoint_roundtrip(trained_run):
    summaries = verify_training_run(trained_run)
    assert len(summaries) == 2 and all(r["verification"] == "PASS" for r in summaries)
    path = trained_run / "seed_42" / "baseline.pt"
    model, meta = load_checkpoint(
        path,
        expected_sha256=sha256_file(path),
        expected_seed=42,
        expected_variant="baseline",
        expected_run_kind="full",
    )
    assert meta["epochs_completed"] == 7
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2,)
    with pytest.raises(ValueError, match="every configured seed"):
        verify_training_run(trained_run, require_full=True)


def test_checkpoint_checksum_and_metadata_rejection(trained_run, tmp_path):
    original = trained_run / "seed_42" / "baseline.pt"
    bad = tmp_path / "bad.pt"
    shutil.copy2(original, bad)
    bad.write_bytes(bad.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(bad, expected_sha256=sha256_file(original))
    with pytest.raises(ValueError, match="seed mismatch"):
        load_checkpoint(original, expected_sha256=sha256_file(original), expected_seed=123)
    with pytest.raises(ValueError, match="run_kind mismatch"):
        load_checkpoint(original, expected_sha256=sha256_file(original), expected_run_kind="smoke")


@pytest.mark.parametrize("corruption", ["architecture", "shape", "nonfinite"])
def test_incompatible_checkpoint_rejected(trained_run, tmp_path, corruption):
    payload = torch.load(trained_run / "seed_42" / "baseline.pt", weights_only=True)
    if corruption == "architecture":
        payload["metadata"]["architecture"]["channels"] = [1, 2, 3]
    elif corruption == "shape":
        payload["state_dict"]["features.0.weight"] = torch.zeros(1)
    else:
        payload["state_dict"]["features.0.weight"].fill_(float("nan"))
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, expected_sha256=sha256_file(path))


def test_portable_handoff_without_dataset(trained_run, tmp_path):
    target = tmp_path / "downloaded"
    shutil.copytree(trained_run, target)
    refs = json.loads((target / "dataset_references.json").read_text())
    assert refs["42"]["generator_version"] == "1.1"
    assert len(verify_training_run(target)) == 2
    path = target / "seed_42" / "validation_probe.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="probe SHA-256"):
        verify_training_run(target)
