"""Real binary optimization and validation-only model selection."""

import csv
import json
import logging
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cowbias.data.pipeline import sha256_file
from cowbias.models.cnn import SmallCNN
from cowbias.training.data import TorchManifestDataset, identity_sha256, make_loader
from cowbias.training.reproducibility import clone_state, seed_everything, state_sha256

LOGGER = logging.getLogger("cowbias.training")
GROUPS = ("cow/grass", "cow/beach", "non_cow/grass", "non_cow/beach")
SELECTION_POLICY = "highest validation WGA, then highest validation accuracy, then earliest epoch"
RELOAD_ATOL = 1e-6
RELOAD_RTOL = 1e-5


def selection_key(metrics: dict, epoch: int) -> tuple[float, float, int]:
    scores = (metrics["worst_group_accuracy"], metrics["accuracy"])
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores) or epoch < 1:
        raise ValueError("Invalid validation selection metrics or epoch")
    return scores[0], scores[1], -epoch


def validation_statistics(
    correct: list[bool], labels: list[str], backgrounds: list[str], loss_sum: float
) -> dict:
    if not correct or not len(correct) == len(labels) == len(backgrounds):
        raise ValueError("Validation records must be nonempty and aligned")
    counts = {group: 0 for group in GROUPS}
    hits = {group: 0 for group in GROUPS}
    for hit, label, bg in zip(correct, labels, backgrounds, strict=True):
        group = f"{label}/{bg}"
        if group not in counts:
            raise ValueError(f"Unsupported validation group: {group}")
        counts[group] += 1
        hits[group] += int(hit)
    if any(n == 0 for n in counts.values()):
        raise ValueError(f"Validation group denominator is zero: {counts}")
    if not math.isfinite(loss_sum):
        raise ValueError("Validation loss is not finite")
    accuracies = {group: hits[group] / counts[group] for group in GROUPS}
    return {
        "loss": loss_sum / len(correct),
        "accuracy": sum(correct) / len(correct),
        "group_accuracy": accuracies,
        "group_counts": counts,
        "worst_group_accuracy": min(accuracies.values()),
    }


def _batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = batch["image"].to(device), batch["target"].to(device)
    if targets.ndim != 1 or targets.shape[0] != images.shape[0] or targets.numel() == 0:
        raise ValueError("Invalid or empty target batch shape")
    if targets.dtype != torch.float32 or not torch.isfinite(targets).all():
        raise ValueError("Targets must be finite float32 binary values")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must map non_cow=0 and cow=1")
    return images, targets


def validate(model: SmallCNN, loader: DataLoader, device: torch.device) -> dict:
    if loader.dataset.split != "val" or loader.dataset.variant != "shared_eval":
        raise ValueError("Model selection may only access shared validation data")
    model.eval()
    correct, labels, backgrounds = [], [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            images, targets = _batch(batch, device)
            logits = model(images)
            if logits.shape != targets.shape or not torch.isfinite(logits).all():
                raise ValueError("Invalid validation logits")
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            loss_sum += float(loss) * targets.numel()
            correct.extend(((logits >= 0) == targets.bool()).cpu().tolist())
            labels.extend(batch["object_label"])
            backgrounds.extend(batch["background_label"])
    return validation_statistics(correct, labels, backgrounds, loss_sum)


def train_epoch(
    model: SmallCNN, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> dict:
    if loader.dataset.split != "train" or loader.dataset.variant not in ("baseline", "treatment"):
        raise ValueError("Optimization may only consume a training variant")
    model.train()
    loss_sum, hits, count = 0.0, 0, 0
    foreground_order = []
    for batch in loader:
        images, targets = _batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != targets.shape or not torch.isfinite(logits).all():
            raise ValueError("Invalid training logits")
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Non-finite parameter gradients")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Non-finite model parameters after optimizer step")
        loss_sum += float(loss.detach()) * targets.numel()
        hits += int(((logits.detach() >= 0) == targets.bool()).sum())
        count += targets.numel()
        foreground_order.extend(batch["foreground_id"])
    if count != len(loader.dataset) or count == 0:
        raise ValueError("Training did not consume every foreground exactly once")
    return {
        "loss": loss_sum / count,
        "accuracy": hits / count,
        "foreground_permutation_sha256": identity_sha256(foreground_order),
    }


def train_arm(
    train_data: TorchManifestDataset,
    validation_data: TorchManifestDataset,
    initial_state: dict[str, torch.Tensor],
    output: Path,
    context: dict,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    threads: int,
    device: torch.device,
    probe: torch.Tensor,
) -> dict:
    """Fresh optimizer/RNG/loader per arm; retain only validation-selected weights."""
    from cowbias.training.checkpoint import load_checkpoint

    if epochs < 1 or not 0 < learning_rate < 1:
        raise ValueError("Invalid training epochs or learning rate")
    if train_data.split != "train" or validation_data.split != "val":
        raise ValueError("Training/validation split boundary violated")
    if train_data.audited != validation_data.audited or train_data.audited.seed != seed:
        raise ValueError("Training and validation must share one audited seed dataset")
    train_data.audited.assert_unchanged()
    seed_everything(seed, threads)
    model = SmallCNN(train_data.images.shape[-1]).to(device)
    model.load_state_dict(initial_state, strict=True)
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), initial_state[key]):
            raise ValueError(f"Initial parameter mismatch: {key}")
    initial_hash = state_sha256(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = make_loader(train_data, seed, batch_size, shuffle=True)
    val_loader = make_loader(validation_data, seed + 1, batch_size, shuffle=False)
    variant = train_data.variant
    history_path = output / f"{variant}_history.csv"
    checkpoint_path = output / f"{variant}.pt"
    if history_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite training outputs for {variant}")
    history, best_state, best_metrics, best_epoch = [], None, None, 0
    with history_path.open("x", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            training = train_epoch(model, train_loader, optimizer, device)
            metrics = validate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": training["loss"],
                "train_accuracy": training["accuracy"],
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_wga": metrics["worst_group_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": time.perf_counter() - started,
                "foreground_permutation_sha256": training["foreground_permutation_sha256"],
            }
            for group in GROUPS:
                suffix = group.replace("/", "_")
                row[f"val_accuracy_{suffix}"] = metrics["group_accuracy"][group]
                row[f"val_count_{suffix}"] = metrics["group_counts"][group]
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            history.append(row)
            LOGGER.info(
                json.dumps({"event": "epoch_complete", "seed": seed, "variant": variant, **row})
            )
            if best_metrics is None or selection_key(metrics, epoch) > selection_key(
                best_metrics, best_epoch
            ):
                best_state, best_metrics, best_epoch = clone_state(model), metrics, epoch
                # Clearly incomplete diagnostics survive a later training failure.
                torch.save(
                    {
                        "state_dict": best_state,
                        "seed": seed,
                        "variant": variant,
                        "selected_epoch": epoch,
                        "complete": False,
                    },
                    output / f"{variant}_incomplete.pt",
                )
    if best_state is None or state_sha256(best_state) == initial_hash:
        raise ValueError("No optimized model state was produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        reference = model(probe.to(device)).cpu()
    metadata = {
        **context,
        "seed": seed,
        "variant": variant,
        "architecture": model.architecture(),
        "initialization_sha256": initial_hash,
        "initial_parameters_equal": True,
        "selected_state_sha256": state_sha256(best_state),
        "epochs_completed": epochs,
        "selected_epoch": best_epoch,
        "selection_policy": SELECTION_POLICY,
        "selected_validation": best_metrics,
        "optimizer": {"class": "AdamW", "settings": optimizer.defaults},
        "loss": "BCEWithLogitsLoss",
        "batch_size": batch_size,
        "foreground_permutation_sha256": [r["foreground_permutation_sha256"] for r in history],
        "validation_probe_logits": reference.tolist(),
        "reload_tolerance": {"atol": RELOAD_ATOL, "rtol": RELOAD_RTOL},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "complete": True,
    }
    # Restrict metadata to JSON primitives, avoiding TorchVersion/Path/custom classes.
    metadata = json.loads(json.dumps(metadata))
    torch.save(
        {"format_version": 1, "state_dict": best_state, "metadata": metadata}, checkpoint_path
    )
    checksum = sha256_file(checkpoint_path)
    reloaded, saved = load_checkpoint(
        checkpoint_path,
        expected_sha256=checksum,
        expected_seed=seed,
        expected_variant=variant,
        expected_run_kind=context["run_kind"],
    )
    reloaded.to(device).eval()
    with torch.inference_mode():
        replay = reloaded(probe.to(device)).cpu()
    if not torch.allclose(reference, replay, atol=RELOAD_ATOL, rtol=RELOAD_RTOL):
        raise ValueError("Reloaded checkpoint predictions differ from selected model")
    if saved != metadata:
        raise ValueError("Reloaded checkpoint metadata differs")
    metadata = {
        **metadata,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checksum,
        "history_sha256": sha256_file(history_path),
        "reload_verification": {
            "status": "PASS",
            "device": str(device),
            "max_abs_difference": float((reference - replay).abs().max()),
        },
    }
    (output / f"{variant}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / f"{variant}_incomplete.pt").unlink()
    return metadata
