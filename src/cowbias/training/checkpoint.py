"""Restricted checkpoint loading and self-contained validation-probe verification."""

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from cowbias.data.dataset import MANIFEST_FIELDS, artifact_path, read_csv
from cowbias.data.pipeline import sha256_file
from cowbias.models.cnn import SmallCNN, model_from_metadata
from cowbias.training.reproducibility import state_sha256


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_seed: int | None = None,
    expected_variant: str | None = None,
    expected_run_kind: str | None = None,
) -> tuple[SmallCNN, dict]:
    path = Path(path)
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path.name}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {
            "format_version",
            "state_dict",
            "metadata",
        }:
            raise ValueError("Unsupported checkpoint payload schema")
        if payload["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format version")
        metadata = payload["metadata"]
        if (
            not isinstance(metadata, dict)
            or metadata.get("complete") is not True
            or metadata.get("data_origin") != "synthetic"
        ):
            raise ValueError("Incomplete checkpoint or unsupported data origin")
        for key, expected in (
            ("seed", expected_seed),
            ("variant", expected_variant),
            ("run_kind", expected_run_kind),
        ):
            if expected is not None and metadata.get(key) != expected:
                raise ValueError(f"Checkpoint {key} mismatch")
        if metadata.get("variant") not in ("baseline", "treatment") or metadata.get(
            "run_kind"
        ) not in ("full", "smoke"):
            raise ValueError("Invalid checkpoint variant/run kind")
        model = model_from_metadata(metadata["architecture"])
        state = payload["state_dict"]
        if not isinstance(state, dict):
            raise TypeError("Checkpoint state_dict must be a tensor mapping")
        if any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()
        ):
            raise ValueError("Checkpoint has invalid/non-finite parameters")
        model.load_state_dict(state, strict=True)
        if state_sha256(state) != metadata["selected_state_sha256"]:
            raise ValueError("Checkpoint model-state hash mismatch")
        return model.eval(), metadata
    except (KeyError, TypeError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Incompatible checkpoint {path.name}: {exc}") from exc


def verify_training_run(root: str | Path, *, require_full: bool = False) -> list[dict]:
    from cowbias.training.train import GROUPS, SELECTION_POLICY, selection_key

    root = Path(root)
    config = json.loads((root / "run_config.json").read_text())
    status = json.loads((root / "run_status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Training run is not complete")
    if require_full and (
        config["run_kind"] != "full"
        or set(config["seeds"]) != set(config["requested_config"]["training"]["seeds"])
    ):
        raise ValueError("Full handoff requires every configured seed and full-mode checkpoints")
    summaries = []
    for seed in config["seeds"]:
        seed_dir = root / f"seed_{seed}"
        reference = json.loads((root / "dataset_references.json").read_text())[str(seed)]
        snapshot_root = root / "dataset_manifests" / f"seed_{seed}"
        if sha256_file(snapshot_root / "manifest.csv") != reference["manifest_sha256"]:
            raise ValueError("Archived dataset manifest SHA-256 mismatch")
        if sha256_file(snapshot_root / "dataset_config.json") != reference["config_sha256"]:
            raise ValueError("Archived dataset configuration SHA-256 mismatch")
        records = read_csv(snapshot_root / "manifest.csv", MANIFEST_FIELDS)
        validation_ids = {
            r["image_id"] for r in records if r["split"] == "val" and r["variant"] == "shared_eval"
        }
        probe_path = artifact_path(root, f"seed_{seed}/validation_probe.npz")
        with np.load(probe_path, allow_pickle=False) as probe_file:
            probe = torch.from_numpy(probe_file["images"].copy())
            probe_ids = probe_file["image_ids"].tolist()
        if (
            not set(probe_ids) <= validation_ids
            or not probe_ids
            or len(set(probe_ids)) != len(probe_ids)
            or len(probe_ids) != len(probe)
        ):
            raise ValueError("Checkpoint verification probes must come only from shared validation")
        arm_metadata = []
        for variant in ("baseline", "treatment"):
            metadata = json.loads((seed_dir / f"{variant}_metadata.json").read_text())
            model, embedded = load_checkpoint(
                seed_dir / f"{variant}.pt",
                expected_sha256=metadata["checkpoint_sha256"],
                expected_seed=seed,
                expected_variant=variant,
                expected_run_kind=config["run_kind"],
            )
            if any(metadata.get(k) != value for k, value in embedded.items()):
                raise ValueError("Checkpoint sidecar/embedded metadata mismatch")
            if metadata["run_id"] != config["run_id"] or metadata["dataset"] != reference:
                raise ValueError("Checkpoint run/dataset metadata mismatch")
            if (
                metadata["architecture"] != config["architecture"]
                or metadata["training_config"] != config["effective_config"]["training"]
            ):
                raise ValueError("Checkpoint architecture/training configuration differs from run")
            if metadata["selection_policy"] != SELECTION_POLICY:
                raise ValueError("Unexpected checkpoint selection policy")
            if sha256_file(probe_path) != metadata["validation_probe_sha256"]:
                raise ValueError("Validation probe SHA-256 mismatch")
            if sha256_file(seed_dir / f"{variant}_history.csv") != metadata["history_sha256"]:
                raise ValueError("Training history SHA-256 mismatch")
            with (seed_dir / f"{variant}_history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            if len(history) != config["effective_config"]["training"]["epochs"] or metadata[
                "epochs_completed"
            ] != len(history):
                raise ValueError("Training history epoch count mismatch")
            candidates = []
            for epoch, row in enumerate(history, 1):
                if int(row["epoch"]) != epoch or not all(
                    math.isfinite(float(row[k]))
                    for k in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_wga")
                ):
                    raise ValueError("Invalid epoch training history")
                if any(int(row[f"val_count_{g.replace('/', '_')}"]) <= 0 for g in GROUPS):
                    raise ValueError("Validation history contains an empty group")
                candidates.append(
                    (
                        {
                            "worst_group_accuracy": float(row["val_wga"]),
                            "accuracy": float(row["val_accuracy"]),
                        },
                        epoch,
                    )
                )
            best_metrics, best_epoch = max(candidates, key=lambda c: selection_key(*c))
            if metadata["selected_epoch"] != best_epoch:
                raise ValueError(
                    "Checkpoint was not selected by the declared validation-only policy"
                )
            selected = history[best_epoch - 1]
            for field, history_field in (
                ("loss", "val_loss"),
                ("accuracy", "val_accuracy"),
                ("worst_group_accuracy", "val_wga"),
            ):
                if not math.isclose(
                    metadata["selected_validation"][field],
                    float(selected[history_field]),
                    abs_tol=1e-12,
                ):
                    raise ValueError("Selected validation metrics differ from history")
            with torch.inference_mode():
                logits = model(probe)
            expected_logits = torch.tensor(metadata["validation_probe_logits"], dtype=torch.float32)
            tolerance = metadata["reload_tolerance"]
            if (
                logits.shape != expected_logits.shape
                or not torch.isfinite(logits).all()
                or not torch.allclose(logits, expected_logits, **tolerance)
            ):
                raise ValueError("Validation-probe checkpoint reload verification failed")
            arm_metadata.append(metadata)
            summaries.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "selected_epoch": best_epoch,
                    "validation_loss": metadata["selected_validation"]["loss"],
                    "validation_accuracy": best_metrics["accuracy"],
                    "validation_wga": best_metrics["worst_group_accuracy"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "verification": "PASS",
                }
            )
        a, b = arm_metadata
        for field in (
            "initialization_sha256",
            "architecture",
            "optimizer",
            "loss",
            "batch_size",
            "training_config",
            "foreground_order_sha256",
            "validation_order_sha256",
            "foreground_permutation_sha256",
            "preprocessing",
            "epochs_completed",
        ):
            if a[field] != b[field]:
                raise ValueError(f"Baseline/treatment fairness mismatch: {field}")
    (root / "verification.json").write_text(
        json.dumps({"status": "PASS", "models": summaries}, indent=2) + "\n"
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify saved CNN checkpoints using archived validation probes"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--require-full", action="store_true", help="Require all configured seeds and full runs"
    )
    parser.add_argument(
        "--github-summary", action="store_true", help="Write job summary and a results notice"
    )
    args = parser.parse_args(argv)
    try:
        torch.set_num_threads(2)
        results = verify_training_run(args.run, require_full=args.require_full)
        print(json.dumps(results, indent=2))
        if args.github_summary:
            payload = (
                json.dumps(results).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            )
            print(f"::notice title=Verified Stage 2 training::{payload}")
            if os.environ.get("GITHUB_STEP_SUMMARY"):
                lines = [
                    "## Stage 2 — validation-only training results",
                    "",
                    "| Seed | Variant | Epoch | Validation loss | Validation accuracy | Validation WGA | Verified |",
                    "|---|---|---:|---:|---:|---:|---|",
                ]
                lines.extend(
                    f"| {r['seed']} | {r['variant']} | {r['selected_epoch']} | {r['validation_loss']:.6f} | {r['validation_accuracy']:.6f} | {r['validation_wga']:.6f} | PASS |"
                    for r in results
                )
                Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text("\n".join(lines) + "\n")
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Checkpoint verification failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
