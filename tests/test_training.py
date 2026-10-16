import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from cowbias.models.cnn import SmallCNN
from cowbias.training.data import AuditedData, TorchManifestDataset, make_loader
from cowbias.training.experiment import main
from cowbias.training.reproducibility import clone_state, seed_everything
from cowbias.training.train import selection_key, train_epoch, validate, validation_statistics


def test_preprocessing_and_labels(audited):
    data = TorchManifestDataset(audited, split="train", variant="baseline")
    sample = data[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (3, 32, 32)
    assert sample["image"].min() >= -1 and sample["image"].max() <= 1
    original = data.source[0]
    expected = torch.from_numpy(original["image"].astype(np.float32)).permute(2, 0, 1) / 255
    assert torch.allclose(sample["image"], (expected - 0.5) / 0.5)
    assert all(
        float(data[i]["target"]) == (r["object_label"] == "cow") for i, r in enumerate(data.records)
    )
    assert sample["background_label"] in ("grass", "beach")  # metadata only, not image channels


@pytest.mark.parametrize(
    "split,variant",
    [("test", "shared_eval"), ("val", "baseline"), ("train", "shared_eval"), ("train", "unknown")],
)
def test_training_boundary(audited, split, variant):
    with pytest.raises(ValueError):
        TorchManifestDataset(audited, split=split, variant=variant)


def test_real_optimizer_update_and_validation_no_update(audited):
    seed_everything(42)
    train = TorchManifestDataset(audited, split="train", variant="baseline")
    val = TorchManifestDataset(audited, split="val", variant="shared_eval")
    model = SmallCNN()
    before = clone_state(model)
    stats = train_epoch(
        model,
        make_loader(train, 42, 64, shuffle=True),
        torch.optim.AdamW(model.parameters(), lr=0.001),
        torch.device("cpu"),
    )
    assert np.isfinite(stats["loss"])
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    before_validation = clone_state(model)
    result = validate(model, make_loader(val, 43, 64, shuffle=False), torch.device("cpu"))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before_validation.items())
    assert len(result["group_accuracy"]) == 4 and sum(result["group_counts"].values()) == 8
    assert not model.training
    with pytest.raises(ValueError, match="shared validation"):
        validate(model, make_loader(train, 42, 64, shuffle=False), torch.device("cpu"))
    with pytest.raises(ValueError, match="training variant"):
        train_epoch(
            model,
            make_loader(val, 42, 64, shuffle=False),
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
        )


def test_model_selection_policy():
    def metric(wga, acc):
        return {"worst_group_accuracy": wga, "accuracy": acc}

    assert selection_key(metric(0.5, 0.6), 3) > selection_key(metric(0.4, 0.9), 1)
    assert selection_key(metric(0.5, 0.7), 3) > selection_key(metric(0.5, 0.6), 1)
    assert selection_key(metric(0.5, 0.7), 1) > selection_key(metric(0.5, 0.7), 3)
    with pytest.raises(ValueError):
        selection_key(metric(float("nan"), 0.5), 1)


def test_missing_validation_group():
    with pytest.raises(ValueError, match="denominator"):
        validation_statistics([True], ["cow"], ["grass"], 0.1)
    with pytest.raises(ValueError, match="nonempty"):
        validation_statistics([], [], [], 0.0)


def test_cli_missing_data_and_output_protection(tmp_path):
    output = tmp_path / "failed"
    assert (
        main(
            [
                "--config",
                "config/experiment.yaml",
                "--data-run",
                str(tmp_path / "missing"),
                "--output",
                str(output),
                "--seeds",
                "42",
                "--device",
                "cpu",
            ]
        )
        == 1
    )
    assert json.loads((output / "run_status.json").read_text())["status"] == "failed"
    assert (output / "failure.log").is_file()
    assert main(["--config", "config/experiment.yaml", "--output", str(output), "--smoke"]) == 1
    assert main(["--config", str(tmp_path / "missing.yaml"), "--smoke"]) == 1


def test_incompatible_or_unaudited_data(generated, config):
    with pytest.raises(ValueError):
        AuditedData.open(generated, config, 42)


def test_smoke_lifecycle_and_repeatability(tmp_path, config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.as_dict()))
    first, second = tmp_path / "smoke-a", tmp_path / "smoke-b"
    common = ["--config", str(config_path), "--smoke", "--device", "cpu"]
    assert main([*common, "--output", str(first)]) == 0
    assert main([*common, "--output", str(second), "--data-run", str(first / "datasets")]) == 0
    for variant in ("baseline", "treatment"):
        a, b = [
            json.loads((p / "seed_42" / f"{variant}_metadata.json").read_text())
            for p in (first, second)
        ]
        assert a["run_kind"] == "smoke" and a["epochs_completed"] == 1
        assert a["selected_state_sha256"] == b["selected_state_sha256"]
        assert a["selected_validation"] == b["selected_validation"]
    saved = json.loads((first / "run_config.json").read_text())
    assert saved["effective_config"]["dataset"]["train_size"] == 200
