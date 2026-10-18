"""Explicit full-artifact acceptance. These skips in routine CI are covered by full acceptance."""

import json
import os
import shutil
from pathlib import Path

import pytest

from cowbias.dashboard.data import load_run, load_seed
from cowbias.dashboard.prepare import receipt_path

ROOT = os.environ.get("COWBIAS_TEST_FULL_RUN")
pytestmark = pytest.mark.skipif(
    not ROOT,
    reason="Opt-in full scientific artifact; separate acceptance job sets COWBIAS_TEST_FULL_RUN",
)


def app(monkeypatch, section):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("COWBIAS_EVAL_ROOT", ROOT)
    monkeypatch.delenv("COWBIAS_FIXTURE", raising=False)
    at = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=30
    )
    at.session_state["section"] = section
    return at.run()


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_real_seed_identity(monkeypatch, seed):
    at = app(monkeypatch, "Model Comparison")
    at.selectbox(key="active_seed").set_value(seed).run()
    data = load_seed(load_run(ROOT), seed)
    pair = at.selectbox(key="pair").value
    for bg in ("beach", "grass"):
        at.radio(key="background").set_value(bg).run()
        assert not at.exception and at.selectbox(key="pair").value == pair
        image = data.image_id(pair, bg)
        assert any(image in c.value for c in at.caption)
        assert [m.value for m in at.metric] == [
            f"{data.prediction(image, m)['prob_cow']:.4%}" for m in ("baseline", "treatment")
        ]


def test_regression_and_empty_case(monkeypatch):
    at = app(monkeypatch, "Model Comparison")
    at.selectbox(key="outcome").set_value("baseline_only").run()
    data = load_seed(load_run(ROOT), 42)
    image = data.image_id(at.selectbox(key="pair").value, at.radio(key="background").value)
    assert data.disagreements[image]["outcome"] == "baseline_only"
    assert any(e.value == "INCORRECT" for e in at.error)
    at.selectbox(key="active_seed").set_value(123).run()
    at.selectbox(key="outcome").set_value("baseline_only").run()
    assert not at.exception and any("No observed pairs" in i.value for i in at.info)


def test_zero_map_and_downloads(monkeypatch):
    at = app(monkeypatch, "Explainability & Export")
    at.selectbox(key="example").set_value("First baseline zero map").run()
    assert not at.exception and not at.error
    assert any("Zero map: Yes" in m.value for m in at.markdown)
    assert any("Foreground share: Undefined" in m.value for m in at.markdown)
    assert len(at.get("download_button")) == 4


def test_metric_display_source(monkeypatch):
    from cowbias.dashboard.ui import metric_value

    at = app(monkeypatch, "Evaluation")
    at.radio(key="evaluation_scope").set_value("Active seed").run()
    run = load_run(ROOT)
    for m in ("baseline", "treatment"):
        for key in ("overall_accuracy", "worst_group_accuracy"):
            expected = run.summary["per_seed"]["42"]["models"][m][key]
            assert metric_value(run, m, key, 42) == f"{expected:.2%}"
    assert not at.exception and not at.error


def test_run_switch_and_corruption(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    for name in ("a", "b"):
        shutil.copytree(ROOT, tmp_path / name)
        shutil.copyfile(receipt_path(ROOT), receipt_path(tmp_path / name))
    monkeypatch.setenv("COWBIAS_EVAL_ROOT", str(tmp_path))
    at = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/streamlit_app.py"), default_timeout=30
    )
    at.session_state["section"] = "Model Comparison"
    at.run()
    at.selectbox(key="active_seed").set_value(2026).run()
    at.selectbox(key="active_run").set_value(str(tmp_path / "b")).run()
    assert not at.exception and at.session_state["scope"][0] == str(tmp_path / "b")
    p = tmp_path / "a/run_status.json"
    p.write_text(json.dumps({"status": "incomplete"}))
    at.selectbox(key="active_run").set_value(str(tmp_path / "a")).run()
    assert not at.exception and any("invalid" in e.value for e in at.error)
