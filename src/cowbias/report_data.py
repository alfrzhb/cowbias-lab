"""Fail-closed academic-report inputs, bound to original archive bytes."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cowbias.evidence import ARCHIVES, RUN_ID, archive_manifest, sha256, verify_extracted

SEEDS = (42, 123, 2026)
MODELS = ("baseline", "treatment")
GROUPS = ("cow/grass", "cow/beach", "non_cow/grass", "non_cow/beach")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compatible(training, evaluation, references, metadata):
    if (
        training["run_id"] != "stage2-full"
        or evaluation["run_id"] != RUN_ID
        or training["seeds"] != list(SEEDS)
        or evaluation["seeds"] != list(SEEDS)
        or training["code_revision"] != ARCHIVES["stage2-trained-models"]["source_commit"]
        or evaluation["code_revision"] != ARCHIVES["stage3-scientific-results"]["source_commit"]
        or references["training_config"] != training
        or references["training_run_id"] != training["run_id"]
    ):
        raise ValueError("Incompatible training/evaluation run, source or seed metadata")
    for (seed, model), meta in metadata.items():
        ref = references["checkpoints"][f"{seed}/{model}"]
        if (
            any(meta.get(k) != v for k, v in ref["metadata"].items())
            or set(meta) - set(ref["metadata"])
            != {"checkpoint_file", "checkpoint_sha256", "history_sha256", "reload_verification"}
            or ref["sha256"] != meta["checkpoint_sha256"]
            or meta["seed"] != seed
            or meta["variant"] != model
        ):
            raise ValueError(f"Incompatible checkpoint reference: seed {seed}, {model}")


def load_evidence(*, run_id, artifact_root, training_run, archive_dir, stage4_root=None):
    if run_id != RUN_ID:
        raise ValueError(f"Incorrect experiment run ID; expected {RUN_ID}")
    root, training = Path(artifact_root) / run_id, Path(training_run)
    inventory = archive_manifest(archive_dir)
    archives = inventory["archives"]
    verify_extracted(root, archives["stage3-scientific-results"]["members"], run_id + "/")
    verify_extracted(training, archives["stage2-trained-models"]["members"])
    ec, tc = read_json(root / "evaluation_config.json"), read_json(training / "run_config.json")
    refs = read_json(root / "source_references.json")
    metadata = {
        (s, m): read_json(training / f"seed_{s}/{m}_metadata.json") for s in SEEDS for m in MODELS
    }
    compatible(tc, ec, refs, metadata)
    package = Path(__file__).parent
    source_hashes = {}
    for name, digest in ec["source_sha256"].items():
        if sha256(package / name) != digest:
            raise ValueError(f"Original scientific source compatibility mismatch: {name}")
        source_hashes[name] = digest
    # The historical checkpoint verifier writes its result. Always verify a disposable copy.
    import torch

    from cowbias.evaluation.evaluator import verify_frozen_run
    from cowbias.evaluation.verification import verify_evaluation

    torch.set_num_threads(2)
    checkpoints = verify_frozen_run(training, require_full=True)
    evaluation = verify_evaluation(root, require_full=True)
    histories = {
        (s, m): read_csv(training / f"seed_{s}/{m}_history.csv") for s in SEEDS for m in MODELS
    }
    distributions = {}
    for seed in SEEDS:
        rows = read_csv(root / f"seed_{seed}/manifest.csv")
        distributions[seed] = {
            name: dict(
                Counter(
                    r["object_label"] + "/" + r["background_label"]
                    for r in rows
                    if r["split"] == split and r["variant"] == variant
                )
            )
            for name, split, variant in [
                ("baseline", "train", "baseline"),
                ("treatment", "train", "treatment"),
                ("validation", "val", "shared_eval"),
                ("test", "test", "shared_eval"),
            ]
        }
    if any(distributions[s] != distributions[42] for s in SEEDS):
        raise ValueError("Distribution counts differ across seeds")
    arrays, predictions, disagreements = {}, {}, {}
    for seed in SEEDS:
        folder = root / f"seed_{seed}"
        with np.load(folder / "test_inputs.npz", allow_pickle=False) as data:
            arrays[seed] = {k: data[k].copy() for k in data.files}
        predictions[seed] = {
            (r["model"], r["image_id"]): r for r in read_csv(folder / "predictions.csv")
        }
        disagreements[seed] = read_csv(folder / "model_disagreements.csv")
    # Identity-first deterministic illustrative choices; outcome-conditioned cases are disclosed.
    cases = []
    for outcome in ("treatment_only", "baseline_only"):
        options = [(s, r) for s in SEEDS for r in disagreements[s] if r["outcome"] == outcome]
        if not options:
            raise ValueError(f"Required observed case unavailable: {outcome}")
        seed, row = min(options, key=lambda x: (x[0], x[1]["image_id"]))
        cases.append({**row, "seed": seed})
    screenshot = None
    if stage4_root is not None:
        stage4_root = Path(stage4_root)
        verify_extracted(stage4_root, archives["stage4-dashboard-review"]["members"])
        review = read_json(stage4_root / "dashboard-review/browser-review.json")
        if review["status"] != "PASS" or review["run_id"] != run_id or review["page_errors"]:
            raise ValueError("Stage 4 browser evidence incompatible")
        screenshot = stage4_root / "dashboard-review/1440-model.png"
        from PIL import Image

        with Image.open(screenshot) as im:
            im.verify()
    return {
        "root": root,
        "training": training,
        "stage4": stage4_root,
        "screenshot": screenshot,
        "archives": inventory,
        "config": tc,
        "evaluation_config": ec,
        "references": refs,
        "source_hashes": source_hashes,
        "metadata": metadata,
        "histories": histories,
        "distributions": distributions,
        "summary": read_json(root / "summary.json"),
        "metrics": {s: read_json(root / f"seed_{s}/metrics.json") for s in SEEDS},
        "arrays": arrays,
        "predictions": predictions,
        "cases": cases,
        "verification": {"checkpoints": checkpoints, "evaluation": evaluation},
    }
