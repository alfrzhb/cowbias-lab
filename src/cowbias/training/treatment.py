"""Verify the existing manifest intervention; never create another treatment generator."""

from cowbias.data.dataset import ManifestDataset, artifact_path
from cowbias.training.data import AuditedData, identity_sha256


def verify_fair_inputs(audited: AuditedData) -> dict:
    audited.assert_unchanged()
    baseline = ManifestDataset(audited.root, split="train", variant="baseline").records
    treatment = ManifestDataset(audited.root, split="train", variant="treatment").records
    expected = audited.config.dataset.train_size
    if len(baseline) != expected or len(treatment) != expected:
        raise ValueError("Both training variants must contain the configured number of records")
    for a, b in zip(baseline, treatment, strict=True):
        if any(
            a[key] != b[key] for key in ("foreground_id", "object_label", "foreground_family_id")
        ):
            raise ValueError("Baseline/treatment foreground identity, label or order differs")
        if (
            artifact_path(audited.root, a["mask_path"]).read_bytes()
            != artifact_path(audited.root, b["mask_path"]).read_bytes()
        ):
            raise ValueError("Baseline/treatment foreground geometry differs")
    groups = audited.report["groups"]
    for label in ("cow", "non_cow"):
        for background in ("grass", "beach"):
            if groups["train/treatment"][f"{label}/{background}"] != expected // 4:
                raise ValueError("Treatment background balance is invalid")
    validation = ManifestDataset(audited.root, split="val", variant="shared_eval").records
    return {
        "foreground_ids": [r["foreground_id"] for r in baseline],
        "foreground_order_sha256": identity_sha256([r["foreground_id"] for r in baseline]),
        "validation_image_ids": [r["image_id"] for r in validation],
        "validation_order_sha256": identity_sha256([r["image_id"] for r in validation]),
        "training_records_per_variant": expected,
        "source_leakage": audited.report["leakage"]["cross_split_overlaps"],
        "identical_masks_and_geometry": True,
        "groups": groups,
    }
