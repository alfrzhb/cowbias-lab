"""Report provenance and publication failures must not produce a plausible final PDF."""

import copy
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cowbias.evidence import (
    ARCHIVES,
    RUN_ID,
    archive_manifest,
    metadata_matches,
    sha256,
    verify_archive,
    verify_extracted,
)
from cowbias.report import generate
from cowbias.report_data import compatible, load_evidence, read_json
from cowbias.report_verification import verify_pdf


@pytest.mark.parametrize("script", ["export_report.py", "verify_report.py", "archive_evidence.py"])
def test_cli_help(script):
    result = subprocess.run(
        [sys.executable, "scripts/" + script, "--help"], capture_output=True, check=False
    )
    assert result.returncode == 0
    assert b"usage:" in result.stdout


def test_wrong_run_id_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match="Incorrect experiment run ID"):
        load_evidence(
            run_id="newest", artifact_root=tmp_path, training_run=tmp_path, archive_dir=tmp_path
        )


@pytest.mark.parametrize("collision", ["pdf", "manifest", "same_path"])
def test_report_output_collision(tmp_path, collision):
    output, manifest = tmp_path / "report.pdf", tmp_path / "manifest.json"
    if collision == "same_path":
        manifest = output
    else:
        (output if collision == "pdf" else manifest).write_bytes(b"KEEP ORIGINAL")
    with pytest.raises(ValueError, match="output path collision"):
        generate(output=output, manifest=manifest)
    if collision != "same_path":
        assert (output if collision == "pdf" else manifest).read_bytes() == b"KEEP ORIGINAL"


def test_cannot_write_report_inside_evidence(tmp_path):
    with pytest.raises(ValueError, match="outside original evidence"):
        generate(
            output=tmp_path / "report.pdf",
            manifest=tmp_path / "manifest.json",
            archive_dir=tmp_path,
        )


def test_corrupted_archive_rejected(tmp_path):
    path = tmp_path / "original.zip"
    path.write_bytes(b"broken archive")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_archive(path, ARCHIVES["stage2-trained-models"]["sha256"])
    with pytest.raises(zipfile.BadZipFile):
        verify_archive(path, sha256(path))


def test_member_hash_and_missing_file_rejected(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("original")
    digest = sha256(path)
    verify_extracted(tmp_path, {"history.csv": digest})
    path.write_text("corruption")
    with pytest.raises(ValueError, match="differs"):
        verify_extracted(tmp_path, {"history.csv": digest})
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_extracted(tmp_path, {"history.csv": digest})


@pytest.mark.parametrize(
    "field,value", [("expired", True), ("id", 1), ("name", "wrong"), ("digest", "sha256:wrong")]
)
def test_remote_archive_metadata(field, value):
    name = "stage2-trained-models"
    spec = ARCHIVES[name]
    info = {
        "id": spec["artifact_id"],
        "name": name,
        "expired": False,
        "digest": "sha256:" + spec["sha256"],
        "workflow_run": {"id": spec["workflow_run_id"], "head_sha": spec["source_commit"]},
    }
    metadata_matches(info, name)
    info[field] = value
    with pytest.raises(ValueError, match="metadata incompatible"):
        metadata_matches(info, name)


@pytest.fixture(scope="module")
def original_sources():
    folder = os.getenv("COWBIAS_TEST_EVIDENCE")
    if not folder:
        pytest.skip("Pinned original archives required; executed in final acceptance")
    root = Path(folder)
    return {
        "run_id": RUN_ID,
        "artifact_root": root / "stage3-scientific-results",
        "training_run": root / "stage2-trained-models",
        "archive_dir": root,
        "stage4_root": root / "stage4-dashboard-review",
    }


@pytest.fixture(scope="module")
def original(original_sources):
    return load_evidence(**original_sources)


def test_full_evidence_and_original_files_unchanged(original, original_sources):
    assert original["verification"]["evaluation"]["prediction_rows"] == 2880
    assert len(original["verification"]["checkpoints"]) == 6
    assert [(r["seed"], r["selected_epoch"]) for r in original["verification"]["checkpoints"]] == [
        (42, 7),
        (42, 4),
        (123, 7),
        (123, 7),
        (2026, 1),
        (2026, 4),
    ]
    verify_extracted(
        original["training"], original["archives"]["archives"]["stage2-trained-models"]["members"]
    )
    assert archive_manifest(original_sources["archive_dir"])["preservation"]["status"] == "PENDING"


def test_incompatible_seed_metadata(original):
    config = copy.deepcopy(original["config"])
    config["seeds"] = [42]
    with pytest.raises(ValueError, match="seed metadata"):
        compatible(
            config, original["evaluation_config"], original["references"], original["metadata"]
        )


def test_incompatible_checkpoint_metadata(original):
    metadata = copy.deepcopy(original["metadata"])
    metadata[42, "baseline"]["selected_epoch"] = 1
    with pytest.raises(ValueError, match="checkpoint reference"):
        compatible(
            original["config"], original["evaluation_config"], original["references"], metadata
        )


@pytest.mark.parametrize("member", ["seed_42/baseline_history.csv", "seed_42/baseline.pt"])
def test_missing_critical_training_source(original_sources, tmp_path, member):
    import shutil

    training = tmp_path / "training"
    shutil.copytree(original_sources["training_run"], training)
    (training / member).unlink()
    with pytest.raises(ValueError, match="missing"):
        load_evidence(**{**original_sources, "training_run": training})


def test_missing_report_source_figure(original_sources, tmp_path):
    import shutil

    artifact_root = tmp_path / "evaluation"
    shutil.copytree(original_sources["artifact_root"], artifact_root)
    (artifact_root / RUN_ID / "seed_42/figures/confusion_matrix.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_evidence(**{**original_sources, "artifact_root": artifact_root})


def test_full_pdf_generation_and_numerical_trace(original_sources, tmp_path):
    pdf, manifest = tmp_path / "report.pdf", tmp_path / "manifest.json"
    result = generate(output=pdf, manifest=manifest, **original_sources)
    assert result["pages"] == 13 and result["tables"] == 10 and result["figures"] == 9
    evidence = read_json(manifest)
    assert verify_pdf(pdf, evidence)["status"] == "PASS"
    table = next(e for e in evidence["entries"] if e["id"] == "Tabel 5")
    summary = read_json(original_sources["artifact_root"] / RUN_ID / "summary.json")
    base = next(
        r
        for r in summary["statistics"]
        if r["metric"] == "overall_accuracy" and r["model"] == "baseline"
    )
    assert table["cells"][1][1] == f"{base['mean'] * 100:.2f} ± {base['sample_sd'] * 100:.2f}"
    assert evidence["scientific_totals"]["new_errors"] == 1
    pdf.write_bytes(pdf.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="PDF hash mismatch"):
        verify_pdf(pdf, evidence)
