"""Actual small fixture inference is development evidence, never full experiment results."""

import csv
import io
import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cowbias.dashboard.acceptance import check_run
from cowbias.dashboard.data import (
    discover_runs,
    fingerprint,
    image_array,
    load_arrays,
    load_run,
    load_seed,
    table,
)
from cowbias.dashboard.prepare import digest, main, prepare_run, receipt_path, safe_extract
from cowbias.dashboard.state import choose_valid, reset_scope
from cowbias.dashboard.visuals import attribution_text, overlay, selected_png


@pytest.fixture(scope="session")
def dashboard_fixture(trained_run, tmp_path_factory):
    from cowbias.evaluation.evaluator import evaluate

    root = tmp_path_factory.mktemp("dashboard") / "genuine-smoke-fixture"
    evaluate(trained_run, trained_run.parent / "data-run", root, seeds=[42], smoke=True)
    prepare_run(root, fixture=True)
    return root


@pytest.fixture
def data(dashboard_fixture):
    return load_seed(load_run(dashboard_fixture, allow_fixture=True), 42)


def test_discovery_and_fixture_opt_in(dashboard_fixture, tmp_path):
    assert discover_runs(tmp_path / "missing") == []
    assert discover_runs(dashboard_fixture) == [dashboard_fixture]
    assert dashboard_fixture in discover_runs(dashboard_fixture.parent)
    with pytest.raises(ValueError, match="explicitly"):
        load_run(dashboard_fixture)


def test_receipt_preserves_scientific_files(dashboard_fixture):
    receipt = json.loads(receipt_path(dashboard_fixture).read_text())
    assert receipt["verification"]["status"] == "PASS"
    for name, sha in receipt["files"].items():
        assert digest(dashboard_fixture / name) == sha
    with pytest.raises(ValueError, match="Existing"):
        prepare_run(dashboard_fixture, fixture=True)


@pytest.mark.parametrize(
    "name",
    [
        "summary.json",
        "seed_42/predictions.csv",
        "seed_42/test_inputs.npz",
        "run_status.json",
        "source_references.json",
    ],
)
def test_corruption_and_cache_invalidation(dashboard_fixture, tmp_path, name):
    dest = tmp_path / "copy"
    shutil.copytree(dashboard_fixture, dest)
    shutil.copyfile(receipt_path(dashboard_fixture), receipt_path(dest))
    before = fingerprint(dest)
    p = dest / name
    p.write_bytes(p.read_bytes() + b"corrupt")
    assert before != fingerprint(dest)
    with pytest.raises((ValueError, OSError)):
        load_run(dest, allow_fixture=True)


def test_pair_prediction_and_exact_array_identity(data):
    inputs = load_arrays(data)
    for pair in data.members:
        g, b = [data.image_id(pair, bg) for bg in ("grass", "beach")]
        assert data.manifest[g]["foreground_id"] == data.manifest[b]["foreground_id"]
        assert np.array_equal(image_array(inputs, g, "masks"), image_array(inputs, b, "masks"))
        for model in ("baseline", "treatment"):
            p = data.prediction(g, model)
            row = next(
                r
                for r in table(data.run.root / "seed_42/predictions.csv")
                if r["image_id"] == g and r["model"] == model
            )
            assert p["prob_cow"] == float(row["prob_cow"]) and p["seed"] == data.seed
    with pytest.raises(ValueError, match="Missing image"):
        image_array(inputs, "other-seed")
    with pytest.raises(ValueError, match="Missing pair"):
        data.image_id("other-pair", "grass")
    with pytest.raises(ValueError, match="Seed unavailable"):
        load_seed(data.run, 123)


@pytest.mark.parametrize("mutation", ["duplicate", "checkpoint", "missing_pair"])
def test_schema_guards_beyond_hashes(data, tmp_path, mutation):
    dest = tmp_path / "copy"
    shutil.copytree(data.run.root, dest)
    p = dest / "seed_42/predictions.csv"
    if mutation == "duplicate":
        p.write_text(p.read_text() + p.read_text().splitlines()[1] + "\n")
    elif mutation == "checkpoint":
        rows = table(p)
        rows[0]["checkpoint_sha256"] = "incompatible"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    else:
        p = dest / "seed_42/counterfactual_pairs.csv"
        lines = p.read_text().splitlines()
        p.write_text("\n".join([lines[0], *lines[2:]]) + "\n")
    with pytest.raises(ValueError):
        load_seed(replace(data.run, root=dest), 42)


@pytest.mark.parametrize("mutation", ["ids", "dtype", "shape", "range", "nan"])
def test_npz_validation(data, tmp_path, mutation):
    dest = tmp_path / "copy"
    shutil.copytree(data.run.root, dest)
    p = dest / "seed_42/test_inputs.npz"
    with np.load(p, allow_pickle=False) as f:
        values = {k: f[k] for k in f.files}
    if mutation == "ids":
        values["image_ids"] = values["image_ids"][::-1]
    elif mutation == "dtype":
        values["images"] = values["images"].astype(np.float32)
    elif mutation == "shape":
        values["images"] = values["images"][:, :-1]
    elif mutation == "range":
        values["masks"][0, 0, 0] = 2
    else:
        values["masks"][0, 0, 0] = np.nan
    np.savez_compressed(p, **values)
    with pytest.raises(ValueError):
        load_arrays(replace(data, run=replace(data.run, root=dest)))


def test_metrics_and_consistency(data):
    for model in ("baseline", "treatment"):
        m = data.metrics["models"][model]
        assert m["worst_group_accuracy"] == min(g["accuracy"] for g in m["groups"].values())
        assert sum(map(sum, m["confusion_matrix"])) == len(data.test_ids)
    report = check_run(data.run.root, fixture=True)
    assert report["counts"]["predictions"] == 2 * len(data.test_ids)
    assert report["counts"]["gradcam_maps"] == 2 * len(data.test_ids)


def test_state_and_empty_filter(data):
    assert data.pair_options(label="missing") == []
    state = {"scope": ("old", 123), "pair": "old", "background": "beach"}
    reset_scope(state, ("new", 42))
    assert "pair" not in state
    assert choose_valid(state, "pair", ["a", "b"]) == "a"
    state["background"] = "beach"
    reset_scope(state, ("new", 42))
    assert state["pair"] == "a" and state["background"] == "beach"
    assert choose_valid(state, "pair", []) is None


def test_zero_map_and_undefined(data):
    rgb = image_array(load_arrays(data), data.test_ids[0], "images")
    assert np.array_equal(overlay(rgb, np.zeros(rgb.shape[:2])), rgb)
    assert "Undefined" in attribution_text(None, True)
    for model in ("baseline", "treatment"):
        heat = load_arrays(data, model)
        for image in data.test_ids:
            if not image_array(heat, image, "heatmaps").any():
                assert data.attribution[model, image]["foreground_attribution_share"] is None


def test_exports_and_png_metadata(data):
    for name in ("summary.csv", "summary.json", "seed_42/figures/accuracy_by_group.png"):
        assert data.run.export(name) == (data.run.root / name).read_bytes()
    rgb = image_array(load_arrays(data), data.test_ids[0], "images")
    meta = {"seed": data.seed, "image_id": data.test_ids[0], "model": "baseline", "target": "cow"}
    with Image.open(io.BytesIO(selected_png(rgb, np.zeros(rgb.shape[:2]), meta))) as img:
        assert json.loads(img.info["CowBias provenance"]) == meta
        img.verify()
    with pytest.raises(ValueError):
        data.run.export("../outside")


@pytest.mark.parametrize(
    "names", [["../escape"], ["/absolute"], ["a/../../x"], ["a\\x"], ["same", "same"], ["a/./b"]]
)
def test_unsafe_zip(tmp_path, names):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in names:
            z.writestr(name, b"x")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_zip_symlink(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16
        z.writestr(info, "../outside")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


@pytest.mark.parametrize("limit", ["MAX_FILE", "MAX_TOTAL"])
def test_zip_size_limits(tmp_path, monkeypatch, limit):
    from cowbias.dashboard import prepare

    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("file", b"12345")
    monkeypatch.setattr(prepare, limit, 4)
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_cli_no_overwrite(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"bad")
    assert main(["--archive", str(archive), "--output", str(tmp_path / "new")]) == 1
    assert main(["--archive", str(archive), "--output", str(tmp_path)]) == 1


def app(root, monkeypatch, section="Overview"):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COWBIAS_EVAL_ROOT", str(root))
    monkeypatch.setenv("COWBIAS_FIXTURE", "1")
    at = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=30
    )
    at.session_state["section"] = section
    return at.run()


@pytest.mark.parametrize(
    "section",
    ["Overview", "Dataset Explorer", "Model Comparison", "Evaluation", "Explainability & Export"],
)
def test_five_tabs(dashboard_fixture, monkeypatch, section):
    at = app(dashboard_fixture, monkeypatch, section)
    assert not at.exception and not at.error and len(at.tabs) == 5
    assert any("DEVELOPMENT FIXTURE" in w.value for w in at.warning)


def test_ui_background_keeps_foreground(dashboard_fixture, monkeypatch, data):
    at = app(dashboard_fixture, monkeypatch, "Model Comparison")
    pair = at.selectbox(key="pair").value
    for bg in ("beach", "grass"):
        at.radio(key="background").set_value(bg).run()
        assert not at.exception and at.selectbox(key="pair").value == pair
        image = data.image_id(pair, bg)
        assert any(image in c.value for c in at.caption)
        assert [m.value for m in at.metric] == [
            f"{data.prediction(image, m)['prob_cow']:.4%}" for m in ("baseline", "treatment")
        ]


def test_missing_artifact(tmp_path, monkeypatch):
    at = app(tmp_path / "missing", monkeypatch)
    assert not at.exception and any("Experiment not yet run" in i.value for i in at.info)
