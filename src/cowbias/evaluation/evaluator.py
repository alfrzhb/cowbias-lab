"""Frozen-checkpoint evaluation orchestration; no training or optimizer construction."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import traceback
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from cowbias.config import config_from_dict, load_config
from cowbias.data.dataset import ManifestDataset, write_csv
from cowbias.data.pipeline import code_provenance, sha256_file
from cowbias.evaluation.aggregation import aggregate
from cowbias.evaluation.counterfactual import compare_models, pairs_from_predictions
from cowbias.evaluation.data import TestDataset
from cowbias.evaluation.gradcam import attribution, gradcam
from cowbias.evaluation.metrics import MODELS, classification, validate_predictions
from cowbias.training.checkpoint import load_checkpoint, verify_training_run
from cowbias.training.data import PREPROCESSING, AuditedData
from cowbias.training.experiment import environment, write_json
from cowbias.training.reproducibility import resolve_device, seed_everything, state_sha256

STAGE2 = {
    "workflow_run_id": 38827143763,
    "artifact_id": 12015781574,
    "bundle_sha256": "d99ae16c94d57810e7cad11d25f62be061df943202ba4b9a59d10721012597be",
    "source_commit": "881be87ef2d83100b51bfb069e1fa75097eb9688",
    "merged_commit": "b0a3666d1ae382161c2aba230b17c0c516f89a6e",
}


def verify_original_bundle(training, archive):
    if archive is None or sha256_file(Path(archive)) != STAGE2["bundle_sha256"]:
        raise ValueError("Full evaluation requires exact original Stage 2 ZIP via --bundle-archive")
    import hashlib

    with zipfile.ZipFile(archive) as bundle:
        names = []
        for item in bundle.infolist():
            if item.is_dir():
                continue
            name = Path(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe original bundle member")
            actual = Path(training) / name
            if sha256_file(actual) != hashlib.sha256(bundle.read(item)).hexdigest():
                raise ValueError(f"Extracted Stage 2 member differs from original archive: {name}")
            names.append(str(name))
        if len(set(names)) != len(names):
            raise ValueError("Duplicate Stage 2 ZIP members")
        expected_models = {
            f"seed_{seed}/{model}.pt" for seed in (42, 123, 2026) for model in MODELS
        }
        actual_models = {str(p.relative_to(training)) for p in Path(training).rglob("*.pt")}
        if actual_models != expected_models:
            raise ValueError("Expected exactly six original checkpoints")
    return {
        "status": "PASS",
        "sha256": STAGE2["bundle_sha256"],
        "original_members_verified": len(names),
    }


def verify_frozen_run(root, require_full):
    # Stage 2's verifier writes verification.json. Verify a copy, preserving the input bundle.
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "training"
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns("datasets"))
        return verify_training_run(copy, require_full=require_full)


def verify_identity(training, data, seeds, *, strict_source=True):
    cfg = json.loads((training / "run_config.json").read_text())
    if strict_source:
        package = Path(__file__).resolve().parents[1]
        for name, digest in cfg["source_sha256"].items():
            # Stage 1 and Stage 2 implementations must remain source-compatible.
            if (
                name == "config.py" or name.startswith(("data/", "models/", "training/"))
            ) and sha256_file(package / name) != digest:
                raise ValueError(f"Frozen source compatibility mismatch: {name}")
    config = config_from_dict(cfg["effective_config"])
    if load_config(training / "dataset_generation.yaml") != config:
        raise ValueError("Bundled regeneration YAML differs from frozen configuration")
    refs = json.loads((training / "dataset_references.json").read_text())
    audited = {}
    for seed in seeds:
        a = AuditedData.open(data / f"seed_{seed}/data", config, seed)
        ref = refs[str(seed)]
        if a.manifest_sha256 != ref["manifest_sha256"] or a.config_sha256 != ref["config_sha256"]:
            raise ValueError("Regenerated manifest/config identity mismatch")
        for name in ("sources.csv", "source_allocation.json"):
            if sha256_file(a.root / name) != sha256_file(training / ref["snapshot_path"] / name):
                raise ValueError(f"Regenerated source snapshot mismatch: {name}")
        audited[seed] = a
    return cfg, refs, audited


def infer(model, dataset, *, run_id, seed, variant, checkpoint_hash, device):
    model.to(device).eval().requires_grad_(False)
    before = state_sha256(model.state_dict())
    records = []
    with torch.inference_mode():
        for offset in range(0, len(dataset), 64):
            logits = model(dataset.images[offset : offset + 64].to(device)).cpu()
            if logits.ndim != 1 or not torch.isfinite(logits).all():
                raise ValueError("Invalid model logits")
            for index, logit in enumerate(logits.tolist(), offset):
                r = dataset.records[index]
                truth, predicted = int(r["object_label"] == "cow"), int(logit >= 0)
                records.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "model": variant,
                        **{
                            k: r[k]
                            for k in (
                                "image_id",
                                "pair_id",
                                "foreground_id",
                                "object_label",
                                "background_label",
                            )
                        },
                        "y_true": truth,
                        "predicted": predicted,
                        "prob_cow": float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64))),
                        "logit": logit,
                        "correct": int(predicted == truth),
                        "checkpoint_sha256": checkpoint_hash,
                        "manifest_sha256": dataset.audited.manifest_sha256,
                    }
                )
    if before != state_sha256(model.state_dict()) or any(
        p.grad is not None for p in model.parameters()
    ):
        raise ValueError("Ordinary inference changed model state or accumulated gradients")
    return validate_predictions(records)


def select_examples(dataset, disagreements):
    # Fixed first lexicographic foreground identity of each class; both backgrounds retained.
    selected = []
    for label in ("cow", "non_cow"):
        pair = min(r["pair_id"] for r in dataset.records if r["object_label"] == label)
        for r in dataset.records:
            if r["pair_id"] == pair:
                selected.append(
                    {
                        k: r[k]
                        for k in (
                            "image_id",
                            "pair_id",
                            "foreground_id",
                            "object_label",
                            "background_label",
                        )
                    }
                    | {"selection": "fixed_identity"}
                )
    for outcome in ("treatment_only", "baseline_only", "both_incorrect"):
        candidates = sorted(
            (r for r in disagreements if r["outcome"] == outcome), key=lambda r: r["image_id"]
        )
        if candidates:
            r = candidates[0]
            selected.append(
                {
                    k: r[k]
                    for k in (
                        "image_id",
                        "pair_id",
                        "foreground_id",
                        "object_label",
                        "background_label",
                    )
                }
                | {"selection": "exploratory_" + outcome}
            )
    return selected


def evaluate(
    training: Path,
    data: Path,
    output: Path,
    *,
    seeds=None,
    device="cpu",
    smoke=False,
    bundle_archive=None,
):
    training, data, output = (Path(p).resolve() for p in (training, data, output))
    cfg = json.loads((training / "run_config.json").read_text())
    seeds = list(seeds if seeds is not None else cfg["seeds"])
    configured = cfg["effective_config"]["training"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(cfg["seeds"]):
        raise ValueError("Unsupported or duplicate seeds")
    if not smoke and (
        seeds != configured
        or seeds != [42, 123, 2026]
        or cfg["run_kind"] != "full"
        or cfg["effective_config"]["dataset"]["test_size"] != 480
    ):
        raise ValueError(
            "Full evaluation requires original full checkpoints and all three configured seeds"
        )
    if smoke and len(seeds) != 1:
        raise ValueError("Smoke requires exactly one seed")
    dev = resolve_device(device)
    seed_everything(seeds[0], 2)
    bundle_verification = verify_original_bundle(training, bundle_archive) if not smoke else None
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "run_status.json", {"status": "running"})
        verified = verify_frozen_run(training, require_full=not smoke)
        cfg, refs, audited = verify_identity(training, data, seeds, strict_source=not smoke)
        write_json(output / "checkpoint_verification.json", {"status": "PASS", "models": verified})
        protocol = {
            "schema_version": 1,
            "run_id": output.name,
            "run_kind": "smoke" if smoke else "full",
            "seeds": seeds,
            "data_origin": "synthetic",
            "decision_rule": "logit >= 0",
            "probability_threshold": 0.5,
            "target_class": "true_class",
            "sd_ddof": 1,
            "test_images_per_model_seed": cfg["effective_config"]["dataset"]["test_size"],
            "preprocessing": PREPROCESSING,
            "device": str(dev),
            "threads": 2,
            "sample_selection": "lexicographically first pair_id per class, both backgrounds; exploratory outcomes separate",
            "gradcam": {
                "layer": "ReLU after final Conv2d (features.7)",
                "normalization": "ReLU, bilinear upsample, divide by own positive peak; exact zero stays zero",
                "mask": "nearest 48-to-32 resize; soft-alpha values retained",
                "zero_mass": "shares/enrichment null",
            },
            "started_utc": datetime.now(UTC).isoformat(),
            **code_provenance(),
        }
        write_json(output / "evaluation_config.json", protocol)
        env = environment(dev, 2) | {"matplotlib": version("matplotlib")}
        write_json(output / "environment.json", env)
        freeze = subprocess.run(
            [os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        (output / "requirements_freeze.txt").write_text(freeze.stdout)
        source_refs = {
            "bundle_verification": bundle_verification,
            "training_run_id": cfg["run_id"],
            "training_run_kind": cfg["run_kind"],
            "stage2": STAGE2 if not smoke else None,
            "datasets": refs,
            "training_environment": json.loads((training / "environment.json").read_text()),
            "training_source_sha256": cfg["source_sha256"],
            "training_config": cfg,
            "checkpoints": {},
            "dataset_identity": {},
        }
        per_seed = {}
        for seed in seeds:
            print(f"Evaluating seed {seed}", flush=True)
            folder = output / f"seed_{seed}"
            (folder / "figures").mkdir(parents=True)
            a = audited[seed]
            dataset = TestDataset(a)
            for name in (
                "manifest.csv",
                "dataset_config.json",
                "sources.csv",
                "source_allocation.json",
                "audit.json",
            ):
                shutil.copy2(a.root / name, folder / name)
            source_refs["dataset_identity"][str(seed)] = {
                "manifest_sha256": a.manifest_sha256,
                "config_sha256": a.config_sha256,
                "source_catalog_sha256": sha256_file(a.root / "sources.csv"),
                "source_allocation_sha256": sha256_file(a.root / "source_allocation.json"),
                "audit": a.report,
            }
            originals = ManifestDataset(a.root, split="test", variant="shared_eval")
            original_samples = [originals[i] for i in range(len(originals))]
            np.savez_compressed(
                folder / "test_inputs.npz",
                source_images=np.stack([s["image"] for s in original_samples]),
                source_masks=np.stack([s["mask"] for s in original_samples]),
                images=np.stack([dataset.source[i]["image"] for i in range(len(dataset))]),
                masks=dataset.masks,
                image_ids=np.array([r["image_id"] for r in dataset.records]),
            )
            all_rows, models, hashes = [], {}, {}
            for variant in MODELS:
                sidecar = json.loads(
                    (training / f"seed_{seed}/{variant}_metadata.json").read_text()
                )
                digest = sidecar["checkpoint_sha256"]
                model, meta = load_checkpoint(
                    training / f"seed_{seed}/{variant}.pt",
                    expected_sha256=digest,
                    expected_seed=seed,
                    expected_variant=variant,
                    expected_run_kind=cfg["run_kind"],
                )
                if meta["dataset"] != refs[str(seed)] or meta["preprocessing"] != PREPROCESSING:
                    raise ValueError("Checkpoint dataset/preprocessing mismatch")
                models[variant], hashes[variant] = model, digest
                source_refs["checkpoints"][f"{seed}/{variant}"] = {
                    "sha256": digest,
                    "selected_epoch": meta["selected_epoch"],
                    "selected_state_sha256": meta["selected_state_sha256"],
                    "source": f"seed_{seed}/{variant}.pt",
                    "metadata": meta,
                }
                all_rows.extend(
                    infer(
                        model,
                        dataset,
                        run_id=output.name,
                        seed=seed,
                        variant=variant,
                        checkpoint_hash=digest,
                        device=dev,
                    )
                )
            # Persist raw predictions before calculating scientific results.
            write_csv(folder / "predictions.csv", all_rows)
            metric = {"seed": seed, "data_origin": "synthetic", "models": {}}
            all_pairs = []
            for variant in MODELS:
                rows = [r for r in all_rows if r["model"] == variant]
                pairs, cf = pairs_from_predictions(rows)
                all_pairs.extend(pairs)
                metric["models"][variant] = classification(rows) | {"counterfactual": cf}
            disagreements, comparison = compare_models(all_rows)
            write_csv(folder / "counterfactual_pairs.csv", all_pairs)
            write_csv(folder / "model_disagreements.csv", disagreements)
            metric["comparison"] = comparison
            examples = select_examples(dataset, disagreements)
            write_csv(folder / "gradcam_examples.csv", examples)
            attr = []
            for variant in MODELS:
                heatmaps = []
                model = models[variant]
                for index, r in enumerate(dataset.records):
                    target = int(r["object_label"] == "cow")
                    heat, details = gradcam(
                        model, dataset.images[index : index + 1].to(dev), target
                    )
                    heatmaps.append(heat)
                    attr.append(
                        {
                            "run_id": output.name,
                            "seed": seed,
                            "model": variant,
                            "image_id": r["image_id"],
                            "target": target,
                            "checkpoint_sha256": hashes[variant],
                            "manifest_sha256": a.manifest_sha256,
                            **details,
                            **attribution(heat, dataset.masks[index]),
                        }
                    )
                np.savez_compressed(
                    folder / f"{variant}_heatmaps.npz",
                    heatmaps=np.stack(heatmaps),
                    image_ids=np.array([r["image_id"] for r in dataset.records]),
                )
            write_csv(folder / "attribution.csv", attr)
            metric["attribution"] = {}
            for variant in MODELS:
                items = [r for r in attr if r["model"] == variant]
                nonzero = [r for r in items if r["foreground_attribution_share"] is not None]
                metric["attribution"][variant] = {
                    "total": len(items),
                    "zero_mass_count": len(items) - len(nonzero),
                    "nonzero_count": len(nonzero),
                    **{
                        f"mean_{k}": float(np.mean([r[k] for r in nonzero])) if nonzero else None
                        for k in (
                            "foreground_attribution_share",
                            "background_attribution_share",
                            "foreground_attribution_enrichment",
                            "foreground_area_fraction",
                        )
                    },
                }
            write_json(folder / "metrics.json", metric)
            per_seed[str(seed)] = metric
        summary, records = aggregate(per_seed)
        write_json(output / "source_references.json", source_refs)
        write_json(output / "summary.json", summary)
        write_csv(
            output / "summary.csv", [{k: v for k, v in r.items() if k != "values"} for r in records]
        )
        from cowbias.evaluation.plots import cross_seed_figure, seed_figures

        for seed in seeds:
            seed_figures(output / f"seed_{seed}")
        (output / "figures").mkdir()
        cross_seed_figure(output)
        (output / "evaluation.log").write_text(
            f"{protocol['started_utc']} started {output.name}\n"
            + "\n".join(
                f"seed {s}: audit PASS, frozen inference, pairs, complete Grad-CAM PASS"
                for s in seeds
            )
            + f"\n{datetime.now(UTC).isoformat()} all outputs generated; independent verification follows\n"
        )
        write_json(output / "run_status.json", {"status": "verifying"})
        write_json(
            output / "evaluation_manifest.json",
            {
                "schema_version": 1,
                "files": {
                    str(p.relative_to(output)): sha256_file(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                    and p.name
                    not in ("run_status.json", "verification.json", "evaluation_manifest.json")
                },
                "prediction_rows": sum(2 * len(TestDataset(a)) for a in audited.values()),
                "pairs": sum(len(TestDataset(a)) for a in audited.values()),
            },
        )
        figure_sources = {}
        for figure in sorted(output.rglob("*.png")):
            parent = figure.parent.parent
            names = (
                ["summary.json", "summary.csv"]
                if parent == output
                else [
                    "metrics.json",
                    "predictions.csv",
                    "counterfactual_pairs.csv",
                    "model_disagreements.csv",
                    "gradcam_examples.csv",
                    "test_inputs.npz",
                    "baseline_heatmaps.npz",
                    "treatment_heatmaps.npz",
                ]
            )
            figure_sources[str(figure.relative_to(output))] = {
                str((parent / name).relative_to(output)): sha256_file(parent / name)
                for name in names
            }
        write_json(output / "figure_sources.json", figure_sources)
        # Include the figure-source inventory in the finalized integrity manifest.
        inventory = json.loads((output / "evaluation_manifest.json").read_text())
        inventory["files"]["figure_sources.json"] = sha256_file(output / "figure_sources.json")
        write_json(output / "evaluation_manifest.json", inventory)
        from cowbias.evaluation.verification import verify_evaluation

        report = verify_evaluation(output, require_full=not smoke, allow_pending=True)
        write_json(output / "verification.json", report)
        write_json(
            output / "run_status.json",
            {"status": "complete", "completed_utc": datetime.now(UTC).isoformat()},
        )
        print(f"PASS: {protocol['run_kind']} evaluation. Artifacts: {output}", flush=True)
        for path in sorted(output.rglob("*")):
            if path.is_file():
                print(path)
        return summary
    except Exception as exc:
        write_json(output / "run_status.json", {"status": "failed", "error": str(exc)})
        (output / "failure.log").write_text(traceback.format_exc())
        raise


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate explicit frozen Stage 2 checkpoints on audited held-out pairs"
    )
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--data-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory, never overwrite")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    p.add_argument(
        "--smoke", action="store_true", help="One-seed lifecycle check; never scientific findings"
    )
    p.add_argument(
        "--bundle-archive",
        type=Path,
        help="Exact original Stage 2 ZIP; required for full evaluation",
    )
    args = p.parse_args(argv)
    try:
        evaluate(
            args.training_run,
            args.data_run,
            args.output,
            seeds=args.seeds,
            device=args.device,
            smoke=args.smoke,
            bundle_archive=args.bundle_archive,
        )
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
