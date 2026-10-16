import json

import pytest

torch = pytest.importorskip("torch")

from cowbias.training.data import TorchManifestDataset, make_loader
from cowbias.training.treatment import verify_fair_inputs


def test_foregrounds_preprocessing_and_batch_permutation(audited):
    fairness = verify_fair_inputs(audited)
    assert fairness["training_records_per_variant"] == 200
    assert fairness["source_leakage"] == 0
    baseline = TorchManifestDataset(audited, split="train", variant="baseline")
    treatment = TorchManifestDataset(audited, split="train", variant="treatment")
    assert [r["foreground_id"] for r in baseline.records] == [
        r["foreground_id"] for r in treatment.records
    ]
    assert torch.equal(baseline.targets, treatment.targets)
    assert not torch.equal(baseline.images, treatment.images)
    a = make_loader(baseline, 42, 64, shuffle=True)
    b = make_loader(treatment, 42, 64, shuffle=True)
    for _ in range(2):
        assert [x for batch in a for x in batch["foreground_id"]] == [
            x for batch in b for x in batch["foreground_id"]
        ]
    v1 = TorchManifestDataset(audited, split="val", variant="shared_eval")
    v2 = TorchManifestDataset(audited, split="val", variant="shared_eval")
    assert v1.records == v2.records and torch.equal(v1.images, v2.images)


def test_paired_training_metadata(trained_run):
    a, b = [
        json.loads((trained_run / "seed_42" / f"{variant}_metadata.json").read_text())
        for variant in ("baseline", "treatment")
    ]
    for field in (
        "initialization_sha256",
        "foreground_permutation_sha256",
        "optimizer",
        "architecture",
        "training_config",
        "validation_order_sha256",
        "foreground_order_sha256",
    ):
        assert a[field] == b[field]
    assert a["initial_parameters_equal"] and b["initial_parameters_equal"]
    assert a["selected_state_sha256"] != b["selected_state_sha256"]
    assert a["data_origin"] == "synthetic"
