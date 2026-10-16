"""Audited, paired CNN experiments with portable artifact provenance."""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from cowbias.config import ExperimentConfig, load_config
from cowbias.data.generate import GENERATOR_VERSION
from cowbias.data.pipeline import code_provenance, sha256_file
from cowbias.data.pipeline import main as prepare_data
from cowbias.models.cnn import SmallCNN
from cowbias.training.checkpoint import verify_training_run
from cowbias.training.data import PREPROCESSING, AuditedData, TorchManifestDataset
from cowbias.training.reproducibility import clone_state, resolve_device, seed_everything
from cowbias.training.train import GROUPS, SELECTION_POLICY, train_arm
from cowbias.training.treatment import verify_fair_inputs

LOGGER = logging.getLogger("cowbias.training")


def write_json(path: Path, content: dict | list) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def environment(device: torch.device, threads: int) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {p: version(p) for p in ("torch", "numpy", "Pillow", "PyYAML")},
        "device": str(device),
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "threads": threads,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "remaining_limitations": [
            (
                "Bitwise equivalence across dependency versions, devices, "
                "and hardware is not guaranteed."
            )
        ],
    }


def run_experiment(
    config: ExperimentConfig,
    output: Path,
    *,
    seeds: list[int] | None = None,
    smoke: bool = False,
    data_run: Path | None = None,
    device: str | None = None,
    threads: int = 2,
) -> list[dict]:
    seeds = list(
        seeds
        if seeds is not None
        else config.training.seeds[:1]
        if smoke
        else config.training.seeds
    )
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(config.training.seeds):
        raise ValueError("seeds must be a nonempty unique subset of configured training.seeds")
    if smoke and len(seeds) != 1:
        raise ValueError("Smoke mode requires exactly one seed")
    effective = (
        replace(
            config,
            dataset=replace(config.dataset, train_size=200, val_size=16, test_size=16),
            training=replace(config.training, epochs=1),
        )
        if smoke
        else config
    )
    selected_device = resolve_device(device or ("cpu" if smoke else effective.training.device))
    seed_everything(seeds[0], threads)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "training.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    run_kind = "smoke" if smoke else "full"
    try:
        write_json(output / "run_status.json", {"status": "running"})
        provenance = code_provenance()
        env = environment(selected_device, threads)
        write_json(output / "environment.json", env)
        freeze = subprocess.run(
            [os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        (output / "requirements_freeze.txt").write_text(freeze.stdout)
        run_config = {
            "run_id": output.name,
            "run_kind": run_kind,
            "data_origin": "synthetic",
            "seeds": seeds,
            "requested_config": config.as_dict(),
            "effective_config": effective.as_dict(),
            "architecture": SmallCNN(effective.dataset.model_image_size).architecture(),
            "selection_policy": SELECTION_POLICY,
            "preprocessing": PREPROCESSING,
            "device": str(selected_device),
            "threads": threads,
            **provenance,
        }
        write_json(output / "run_config.json", run_config)
        config_path = output / "dataset_generation.yaml"
        config_path.write_text(yaml.safe_dump(effective.as_dict()))
        if data_run is None:
            data_run = output / "datasets"
            result = prepare_data(
                [
                    "--config",
                    str(config_path),
                    "--output",
                    str(data_run),
                    "--seeds",
                    *map(str, seeds),
                ]
            )
            if result != 0:
                raise ValueError("Stage 1 dataset generation failed; training is forbidden")
        data_run = Path(data_run).resolve()
        references = {}
        for seed in seeds:
            audited = AuditedData.open(data_run / f"seed_{seed}" / "data", effective, seed)
            fair = verify_fair_inputs(audited)
            reference = {
                "seed": seed,
                "dataset_run_id": data_run.name,
                "source_path_relative_to_training_run": os.path.relpath(audited.root, output),
                "manifest_sha256": audited.manifest_sha256,
                "config_sha256": audited.config_sha256,
                "generator_version": GENERATOR_VERSION,
                "data_origin": "synthetic",
                "snapshot_path": f"dataset_manifests/seed_{seed}",
                "regeneration": "python scripts/prepare_data.py --config dataset_generation.yaml "
                f"--seeds {seed} --output artifacts/runs/regenerated-data",
            }
            references[str(seed)] = reference
            write_json(output / "dataset_references.json", references)
            snapshot = output / reference["snapshot_path"]
            snapshot.mkdir(parents=True)
            for name in (
                "manifest.csv",
                "dataset_config.json",
                "sources.csv",
                "source_allocation.json",
                "audit.json",
                "contact_sheet.png",
                "group_distribution.csv",
                "group_distribution.md",
            ):
                shutil.copy2(audited.root / name, snapshot / name)
            seed_dir = output / f"seed_{seed}"
            seed_dir.mkdir()
            write_json(seed_dir / "fairness.json", fair)
            baseline = TorchManifestDataset(audited, split="train", variant="baseline")
            treatment = TorchManifestDataset(audited, split="train", variant="treatment")
            validation = TorchManifestDataset(audited, split="val", variant="shared_eval")
            probe_indices = [
                next(
                    i
                    for i, r in enumerate(validation.records)
                    if f"{r['object_label']}/{r['background_label']}" == group
                )
                for group in GROUPS
            ]
            probe = validation.images[probe_indices].clone()
            probe_path = seed_dir / "validation_probe.npz"
            np.savez_compressed(
                probe_path,
                images=probe.numpy(),
                image_ids=np.array([validation.records[i]["image_id"] for i in probe_indices]),
            )
            seed_everything(seed, threads)
            initial_state = clone_state(SmallCNN(effective.dataset.model_image_size))
            context = {
                "run_id": output.name,
                "run_kind": run_kind,
                "data_origin": "synthetic",
                "dataset": reference,
                "training_config": effective.as_dict()["training"],
                "environment": env,
                **provenance,
                "preprocessing": PREPROCESSING,
                "foreground_order_sha256": fair["foreground_order_sha256"],
                "validation_order_sha256": fair["validation_order_sha256"],
                "validation_probe_sha256": sha256_file(probe_path),
            }
            arms = []
            for train_data in (baseline, treatment):
                arms.append(
                    train_arm(
                        train_data,
                        validation,
                        initial_state,
                        seed_dir,
                        context,
                        epochs=effective.training.epochs,
                        batch_size=effective.training.batch_size,
                        learning_rate=effective.training.learning_rate,
                        seed=seed,
                        threads=threads,
                        device=selected_device,
                        probe=probe,
                    )
                )
            if arms[0]["foreground_permutation_sha256"] != arms[1]["foreground_permutation_sha256"]:
                raise ValueError("Foreground batch permutations differ between arms")
        write_json(output / "run_status.json", {"status": "complete"})
        summaries = verify_training_run(output)
        write_json(output / "training_summary.json", summaries)
        print(f"PASS: {run_kind} run, {len(summaries)} verified checkpoints. {output}")
        for row in summaries:
            prefix = output / f"seed_{row['seed']}" / row["variant"]
            print(f"Checkpoint: {prefix}.pt | History: {prefix}_history.csv")
        return summaries
    except Exception as exc:
        write_json(output / "run_status.json", {"status": "failed", "error": str(exc)})
        (output / "failure.log").write_text(traceback.format_exc())
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train fair baseline/treatment CNNs on audited synthetic data"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One seed, 200 training images/arm, 16 validation, one epoch",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--data-run",
        type=Path,
        help="Existing Stage 1 run containing seed_N/data; otherwise regenerate",
    )
    parser.add_argument("--output", type=Path, help="New training run directory; never overwrite")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        run_id = (
            "stage2-"
            + ("smoke-" if args.smoke else "full-")
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
        )
        run_experiment(
            config,
            args.output or Path(config.artifacts.root) / run_id,
            seeds=args.seeds,
            smoke=args.smoke,
            data_run=args.data_run,
            device=args.device,
            threads=args.threads,
        )
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        LOGGER.error(json.dumps({"event": "training_failed", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
