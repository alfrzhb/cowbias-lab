import hashlib
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cowbias.config import ConfigError, config_from_dict, load_config
from cowbias.data.audit import audit_dataset
from cowbias.data.dataset import MANIFEST_FIELDS, ManifestDataset, read_csv
from cowbias.data.generate import render_foreground
from cowbias.data.pipeline import generate_dataset, main
from cowbias.data.splits import allocate_sources, background_assignments

REPO = Path(__file__).resolve().parents[1]


def test_default_sizes_and_distributions(config):
    records = background_assignments(allocate_sources(config.dataset, 42), config.dataset, 42)
    groups = Counter((f.split, v, f.label, b) for f, b, v, _ in records)
    assert len(records) == 2720
    assert groups["train", "baseline", "cow", "grass"] == 485
    assert groups["train", "baseline", "cow", "beach"] == 15
    assert groups["train", "baseline", "non_cow", "grass"] == 15
    assert groups["train", "baseline", "non_cow", "beach"] == 485
    for label in ("cow", "non_cow"):
        for bg in ("grass", "beach"):
            assert groups["train", "treatment", label, bg] == 250
            assert groups["val", "shared_eval", label, bg] == 60
            assert groups["test", "shared_eval", label, bg] == 120


def test_actual_generation_and_audit(generated, small_config):
    report = audit_dataset(generated, small_config, 42)
    assert report["status"] == "PASS"
    assert report["counts"] == {
        "train/baseline": 200,
        "train/treatment": 200,
        "val/shared_eval": 8,
        "test/shared_eval": 8,
    }
    assert report["groups"]["train/baseline"] == {
        "cow/grass": 97,
        "cow/beach": 3,
        "non_cow/grass": 3,
        "non_cow/beach": 97,
    }
    for classes in report["class_balance"].values():
        assert classes["cow"] == classes["non_cow"]
    assert report["counterfactual"]["pairs"] == 4
    assert report["determinism"]["all_source_pixels_regenerated"] is True
    for name in (
        "audit.json",
        "group_distribution.csv",
        "group_distribution.md",
        "contact_sheet.png",
    ):
        assert (generated / name).is_file()


def test_training_identity_and_test_geometry(generated):
    records = read_csv(generated / "manifest.csv", MANIFEST_FIELDS)
    assert len({r["image_id"] for r in records}) == len(records)
    train = {
        v: {r["foreground_id"]: r for r in records if r["variant"] == v}
        for v in ("baseline", "treatment")
    }
    assert train["baseline"].keys() == train["treatment"].keys()
    for fg, row in train["baseline"].items():
        other = train["treatment"][fg]
        assert (generated / row["mask_path"]).read_bytes() == (
            generated / other["mask_path"]
        ).read_bytes()
    for pair in {r["pair_id"] for r in records if r["split"] == "test"}:
        a, b = [r for r in records if r["pair_id"] == pair]
        assert a["foreground_id"] == b["foreground_id"]
        assert {a["background_label"], b["background_label"]} == {"grass", "beach"}
        with Image.open(generated / a["mask_path"]) as m:
            opaque = np.array(m) == 255
        with (
            Image.open(generated / a["file_path"]) as x,
            Image.open(generated / b["file_path"]) as y,
        ):
            assert np.array_equal(np.array(x)[opaque], np.array(y)[opaque])


def test_dataset_loading_dimensions(generated):
    data = ManifestDataset(generated, split="train", variant="baseline")
    assert len(data) == 200
    item = data[0]
    assert item["image"].shape == (32, 32, 3)
    assert item["mask"].shape == (32, 32)
    assert item["image"].dtype == np.uint8
    assert item["label"] in (0, 1)
    resized = ManifestDataset(generated, split="test", variant="shared_eval", image_size=16)[0]
    assert resized["image"].shape == (16, 16, 3)
    assert resized["mask"].shape == (16, 16)
    with pytest.raises(ValueError, match="No records"):
        ManifestDataset(generated, split="test", variant="baseline")


def test_fixed_seed_reproducibility_and_different_seed(generated, small_config, tmp_path):
    repeat = tmp_path / "repeat"
    generate_dataset(small_config, repeat, 42)

    def digests(root):
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        }

    assert digests(generated) == digests(repeat)
    sources = allocate_sources(small_config.dataset, 42)
    fg = next(s for s in sources if s.kind == "foreground")
    assert not np.array_equal(
        render_foreground(fg, 32), render_foreground(replace(fg, seed=123), 32)
    )
    assert allocate_sources(small_config.dataset, 42) != allocate_sources(small_config.dataset, 123)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("dataset", "image_size", 0, "positive integer"),
        ("dataset", "image_size", 8, "between 24"),
        ("dataset", "model_image_size", 64, "model_image_size"),
        ("dataset", "train_size", 999, "divisible by four"),
        ("dataset", "val_size", -4, "positive integer"),
        ("dataset", "test_size", 6, "divisible by four"),
        ("dataset", "baseline_majority_ratio", -0.1, "between 0 and 1"),
        ("dataset", "baseline_majority_ratio", 1.5, "between 0 and 1"),
        ("dataset", "baseline_majority_ratio", float("nan"), "finite ratio"),
        ("dataset", "baseline_majority_ratio", 1, "< 1"),
        ("dataset", "baseline_majority_ratio", 0.971, "integral counts"),
        ("dataset", "treatment_background_ratio", 0.6, "must be 0.5"),
        ("dataset", "negative_class", "dog", "only 'horse'"),
        ("dataset", "image_size", True, "positive integer"),
        ("training", "seeds", [], "nonempty"),
        ("training", "seeds", [42, 42], "unique"),
        ("training", "seeds", [-1], "integers"),
        ("training", "learning_rate", float("inf"), "finite"),
        ("artifacts", "root", "", "nonempty path"),
    ],
)
def test_invalid_fields(config, section, key, value, message):
    raw = deepcopy(config.as_dict())
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


def test_missing_unknown_and_invalid_yaml(config, tmp_path):
    raw = config.as_dict()
    del raw["dataset"]["train_size"]
    with pytest.raises(ConfigError, match="train_size"):
        config_from_dict(raw)
    raw = config.as_dict()
    raw["dataset"]["unsupported_labels"] = ["dog"]
    with pytest.raises(ConfigError, match="unsupported_labels"):
        config_from_dict(raw)
    path = tmp_path / "invalid.yaml"
    path.write_text("dataset: [unterminated")
    with pytest.raises(ConfigError, match="Cannot read configuration"):
        load_config(path)
    with pytest.raises(ConfigError, match="mapping"):
        config_from_dict([])


def test_cli_help_and_no_overwrite(generated, small_config, tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout and "--audit" in result.stdout
    before = (generated / "manifest.csv").read_bytes()
    assert main(["--config", str(REPO / "config/experiment.yaml"), "--output", str(generated)]) == 1
    assert (generated / "manifest.csv").read_bytes() == before
    with pytest.raises(FileExistsError):
        generate_dataset(small_config, generated, 42)
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("{}")
    output = tmp_path / "should-not-exist"
    assert main(["--config", str(config_path), "--output", str(output)]) == 1
    assert not output.exists()
    report = json.loads((generated / "audit.json").read_text())
    assert report["status"] == "PASS"
