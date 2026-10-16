import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from cowbias.data.pipeline import sha256_file
from cowbias.evaluation.data import TestDataset
from cowbias.evaluation.evaluator import evaluate, infer, main, verify_frozen_run, verify_identity
from cowbias.evaluation.verification import main as verify_cli
from cowbias.evaluation.verification import verify_evaluation
from cowbias.training.checkpoint import load_checkpoint
from cowbias.training.data import TorchManifestDataset
from cowbias.training.reproducibility import state_sha256


@pytest.fixture(scope="module")
def evaluated(trained_run, tmp_path_factory):
    output = tmp_path_factory.mktemp("evaluation") / "smoke"
    # The fixture is a real tiny dataset and actually optimized checkpoints, never full evidence.
    data = trained_run.parent / "data-run"
    evaluate(trained_run, data, output, seeds=[42], smoke=True)
    return output


def test_adapter_exact_preprocessing_mask_and_boundary(audited):
    d = TestDataset(audited)
    assert len(d) == 8
    sample = d.source[0]
    expected = torch.from_numpy(sample["image"].astype(np.float32)).permute(2, 0, 1) / 255
    assert torch.allclose(d[0], (expected - 0.5) / 0.5)
    assert np.array_equal(d.masks[0], sample["mask"].astype(np.float32) / 255)
    with pytest.raises(ValueError):
        TestDataset(audited, split="train")
    with pytest.raises(ValueError):
        TestDataset(None)
    with pytest.raises(ValueError):
        TorchManifestDataset(audited, split="test", variant="shared_eval")


def test_actual_inference_no_grad_or_update(trained_run, audited):
    path = trained_run / "seed_42/baseline.pt"
    model, _ = load_checkpoint(path, expected_sha256=sha256_file(path))
    before = state_sha256(model.state_dict())
    calls = []
    h = model.register_forward_hook(
        lambda m, i, o: calls.append((torch.is_grad_enabled(), o.requires_grad))
    )
    rows = infer(
        model,
        TestDataset(audited),
        run_id="test",
        seed=42,
        variant="baseline",
        checkpoint_hash=sha256_file(path),
        device=torch.device("cpu"),
    )
    h.remove()
    assert len(rows) == 8 and calls and all(v == (False, False) for v in calls)
    assert before == state_sha256(model.state_dict()) and all(
        p.grad is None for p in model.parameters()
    )


def test_smoke_saved_independent_verification(evaluated):
    report = verify_evaluation(evaluated)
    assert report["prediction_rows"] == 16 and report["model_pair_evaluations"] == 8
    assert report["attribution_rows"] == 16
    assert verify_cli(["--run", str(evaluated)]) == 0
    assert verify_cli(["--run", str(evaluated), "--require-full"]) == 1
    assert json.loads((evaluated / "run_status.json").read_text())["status"] == "complete"


def test_cli_rejects_existing_output_and_bad_seeds(trained_run, evaluated, tmp_path):
    data = trained_run.parent / "data-run"
    common = ["--training-run", str(trained_run), "--data-run", str(data), "--smoke"]
    assert main([*common, "--output", str(evaluated), "--seeds", "42"]) == 1
    assert main([*common, "--output", str(tmp_path / "bad"), "--seeds", "7"]) == 1
    assert main([*common, "--output", str(tmp_path / "two"), "--seeds", "42", "42"]) == 1
    assert (
        main(
            [
                "--training-run",
                str(trained_run),
                "--data-run",
                str(data),
                "--output",
                str(tmp_path / "full"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_predictions",
        "duplicate_id",
        "metrics",
        "pairs",
        "disagreements",
        "summary",
        "attribution",
        "heatmap",
        "incomplete",
        "manifest",
        "checkpoint_hash",
    ],
)
def test_artifact_corruption_fails(evaluated, tmp_path, corruption):
    target = tmp_path / corruption
    shutil.copytree(evaluated, target)
    seed = target / "seed_42"
    paths = {
        "metrics": seed / "metrics.json",
        "pairs": seed / "counterfactual_pairs.csv",
        "disagreements": seed / "model_disagreements.csv",
        "summary": target / "summary.json",
        "attribution": seed / "attribution.csv",
        "heatmap": seed / "baseline_heatmaps.npz",
        "manifest": seed / "manifest.csv",
        "checkpoint_hash": target / "source_references.json",
    }
    if corruption == "missing_predictions":
        (seed / "predictions.csv").unlink()
    elif corruption == "incomplete":
        (target / "run_status.json").write_text('{"status":"running"}')
    elif corruption == "duplicate_id":
        p = seed / "predictions.csv"
        p.write_text(p.read_text() + p.read_text().splitlines()[1] + "\n")
    else:
        p = paths[corruption]
        p.write_bytes(p.read_bytes() + b"corrupt")
    with pytest.raises((ValueError, OSError)):
        verify_evaluation(target)
    assert verify_cli(["--run", str(target)]) == 1


@pytest.mark.parametrize("mutation", ["metric", "pairs", "summary", "duplicate"])
def test_independent_arithmetic_even_if_inventory_rehashed(evaluated, tmp_path, mutation):
    target = tmp_path / "altered"
    shutil.copytree(evaluated, target)
    if mutation == "metric":
        p = target / "seed_42/metrics.json"
        d = json.loads(p.read_text())
        d["models"]["baseline"]["overall_accuracy"] = 0.123
        p.write_text(json.dumps(d))
    elif mutation == "pairs":
        p = target / "seed_42/counterfactual_pairs.csv"
        lines = p.read_text().splitlines()
        cells = lines[1].split(",")
        cells[lines[0].split(",").index("flip")] = "9"
        lines[1] = ",".join(cells)
        p.write_text("\n".join(lines) + "\n")
    elif mutation == "duplicate":
        p = target / "seed_42/predictions.csv"
        lines = p.read_text().splitlines()
        lines[2] = lines[1]
        p.write_text("\n".join(lines) + "\n")
    else:
        p = target / "summary.json"
        d = json.loads(p.read_text())
        d["statistics"][0]["mean"] = 0.123
        p.write_text(json.dumps(d))
    inventory = target / "evaluation_manifest.json"
    d = json.loads(inventory.read_text())
    d["files"][str(p.relative_to(target))] = sha256_file(p)
    inventory.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        verify_evaluation(target)


def test_model_and_data_integrity_fail_closed(trained_run, tmp_path):
    training = tmp_path / "training"
    shutil.copytree(trained_run, training)
    before = sha256_file(training / "verification.json")
    assert len(verify_frozen_run(training, False)) == 2
    assert before == sha256_file(training / "verification.json")
    (training / "seed_42/baseline.pt").unlink()
    with pytest.raises((ValueError, OSError)):
        verify_frozen_run(training, False)
    with pytest.raises((ValueError, OSError)):
        verify_identity(trained_run, Path("/nonexistent"), [42], strict_source=False)
