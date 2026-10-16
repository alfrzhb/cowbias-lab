"""Independent artifact verifier. Rebuilds arithmetic from CSV, not evaluator objects.

Intentionally does not import metrics/counterfactual/aggregation/attribution routines.
Hashes detect corruption, not a malicious party rewriting an entire unsigned bundle.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

GROUPS = ("cow/grass", "cow/beach", "non_cow/grass", "non_cow/beach")
MODELS = ("baseline", "treatment")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def read_table(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(bool(rows), f"Empty table {path.name}")
    return rows


def equal(actual, expected, context="value"):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(), f"{context}: keys differ"
        )
        for k, v in expected.items():
            equal(actual[k], v, f"{context}/{k}")
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected), f"{context}: length differs"
        )
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal(a, b, f"{context}/{i}")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(
            isinstance(actual, (float, int))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"{context}: numeric mismatch",
        )
    else:
        require(actual == expected, f"{context}: mismatch")


def check_table(actual, expected, context):
    require(len(actual) == len(expected), f"{context}: row count mismatch")
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        require(set(a) == set(b), f"{context}: columns mismatch")
        for k, v in b.items():
            if isinstance(v, (int, float)):
                equal(float(a[k]), v, f"{context}/{i}/{k}")
            else:
                require(a[k] == str(v), f"{context}/{i}/{k}: mismatch")


def avg(values):
    return sum(values) / len(values)


def sd(values):
    return (
        math.sqrt(sum((v - avg(values)) ** 2 for v in values) / (len(values) - 1))
        if len(values) > 1
        else None
    )


def independently_classify(rows):
    groups = {}
    for g in GROUPS:
        items = [r for r in rows if r["object_label"] + "/" + r["background_label"] == g]
        require(bool(items), "Missing group")
        hits = sum(r["predicted"] == r["y_true"] for r in items)
        groups[g] = {"correct": hits, "total": len(items), "accuracy": hits / len(items)}
    matrix = [[0, 0], [0, 0]]
    for r in rows:
        matrix[r["y_true"]][r["predicted"]] += 1
    correct = matrix[0][0] + matrix[1][1]
    return {
        "total": len(rows),
        "correct": correct,
        "overall_accuracy": correct / len(rows),
        "groups": groups,
        "worst_group_accuracy": min(v["accuracy"] for v in groups.values()),
        "confusion_matrix": matrix,
        "class_order": ["non_cow", "cow"],
        "confusion_convention": "rows actual, columns predicted",
    }


def independently_pair(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pair_id"]].append(r)
    pairs, foregrounds = [], set()
    for pair, members in grouped.items():
        require(len(members) == 2, "Pair member count")
        by_bg = {r["background_label"]: r for r in members}
        require(set(by_bg) == {"grass", "beach"}, "Pair backgrounds")
        g, b = by_bg["grass"], by_bg["beach"]
        require(
            g["foreground_id"] == b["foreground_id"] and g["y_true"] == b["y_true"], "Pair identity"
        )
        require(g["foreground_id"] not in foregrounds, "Repeated foreground pair")
        foregrounds.add(g["foreground_id"])
        gc, bc = int(g["predicted"] == g["y_true"]), int(b["predicted"] == b["y_true"])
        outcome = {
            (1, 1): "both_correct",
            (1, 0): "grass_correct_only",
            (0, 1): "beach_correct_only",
            (0, 0): "both_incorrect",
        }[gc, bc]
        pairs.append(
            {k: g[k] for k in ("run_id", "seed", "model", "foreground_id", "y_true")}
            | {
                "pair_id": pair,
                "grass_image_id": g["image_id"],
                "beach_image_id": b["image_id"],
                "grass_predicted": g["predicted"],
                "beach_predicted": b["predicted"],
                "grass_prob_cow": g["prob_cow"],
                "beach_prob_cow": b["prob_cow"],
                "flip": int(g["predicted"] != b["predicted"]),
                "both_correct": gc * bc,
                "grass_correct": gc,
                "beach_correct": bc,
                "delta_prob_cow": abs(g["prob_cow"] - b["prob_cow"]),
                "outcome": outcome,
            }
        )
    deltas = sorted(p["delta_prob_cow"] for p in pairs)
    n = len(pairs)
    median = (deltas[(n - 1) // 2] + deltas[n // 2]) / 2
    return pairs, {
        "total_pairs": n,
        "flipped_pairs": sum(p["flip"] for p in pairs),
        "background_flip_rate": sum(p["flip"] for p in pairs) / n,
        "both_correct_pairs": sum(p["both_correct"] for p in pairs),
        "paired_correct_rate": sum(p["both_correct"] for p in pairs) / n,
        "outcomes": {
            k: sum(p["outcome"] == k for p in pairs)
            for k in ("both_correct", "grass_correct_only", "beach_correct_only", "both_incorrect")
        },
        "probability_sensitivity": {
            "mean": avg(deltas),
            "median": median,
            "sample_sd": sd(deltas) or 0.0,
            "maximum": max(deltas),
            "minimum": min(deltas),
        },
    }


def independently_compare(rows):
    base = {r["image_id"]: r for r in rows if r["model"] == "baseline"}
    treatment = {r["image_id"]: r for r in rows if r["model"] == "treatment"}
    require(base.keys() == treatment.keys(), "Model IDs mismatch")
    output = []
    for image, a in base.items():
        b = treatment[image]
        ac, bc = int(a["predicted"] == a["y_true"]), int(b["predicted"] == b["y_true"])
        outcome = {
            (1, 1): "both_correct",
            (1, 0): "baseline_only",
            (0, 1): "treatment_only",
            (0, 0): "both_incorrect",
        }[ac, bc]
        output.append(
            {
                k: a[k]
                for k in (
                    "run_id",
                    "seed",
                    "image_id",
                    "pair_id",
                    "foreground_id",
                    "y_true",
                    "object_label",
                    "background_label",
                )
            }
            | {
                "baseline_predicted": a["predicted"],
                "treatment_predicted": b["predicted"],
                "baseline_prob_cow": a["prob_cow"],
                "treatment_prob_cow": b["prob_cow"],
                "baseline_correct": ac,
                "treatment_correct": bc,
                "probability_change": b["prob_cow"] - a["prob_cow"],
                "disagree": int(a["predicted"] != b["predicted"]),
                "outcome": outcome,
            }
        )

    def summarize(items):
        counts = {
            k: sum(r["outcome"] == k for r in items)
            for k in ("both_correct", "baseline_only", "treatment_only", "both_incorrect")
        }
        n = len(items)
        return {
            "total": n,
            "outcomes": counts,
            "outcome_rates": {k: v / n for k, v in counts.items()},
            "model_disagreement_rate": sum(r["disagree"] for r in items) / n,
            "treatment_corrected": counts["treatment_only"],
            "new_errors": counts["baseline_only"],
            "net_correct_change": counts["treatment_only"] - counts["baseline_only"],
            "mean_probability_change": avg([r["probability_change"] for r in items]),
        }

    result = summarize(output)
    result["by_class"] = {
        label: summarize([r for r in output if r["object_label"] == label])
        for label in ("cow", "non_cow")
    }
    result["by_background"] = {
        bg: summarize([r for r in output if r["background_label"] == bg])
        for bg in ("grass", "beach")
    }
    return output, result


def verify_evaluation(root, *, require_full=False, allow_pending=False):
    root = Path(root).resolve()
    cfg = load_json(root / "evaluation_config.json")
    status = load_json(root / "run_status.json")
    require(
        status["status"] == "complete" or (allow_pending and status["status"] == "verifying"),
        "Incomplete evaluation run",
    )
    seeds = cfg["seeds"]
    require(bool(seeds) and len(set(seeds)) == len(seeds), "Invalid seeds")
    if require_full:
        require(
            cfg["run_kind"] == "full"
            and seeds == [42, 123, 2026]
            and cfg["test_images_per_model_seed"] == 480,
            "Full verification requires original three seeds and 480 test images/model",
        )
    require(
        cfg["data_origin"] == "synthetic"
        and cfg["decision_rule"] == "logit >= 0"
        and cfg["probability_threshold"] == 0.5
        and cfg["target_class"] == "true_class"
        and cfg["sd_ddof"] == 1,
        "Evaluation protocol mismatch",
    )
    manifest = load_json(root / "evaluation_manifest.json")
    excluded = {"run_status.json", "verification.json", "evaluation_manifest.json"}
    actual_files = {
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name not in excluded
    }
    require(actual_files == set(manifest["files"]), "Artifact inventory mismatch")
    for name, digest in manifest["files"].items():
        p = (root / name).resolve()
        require(p.is_relative_to(root) and p.is_file(), "Unsafe or missing artifact")
        require(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest, f"Artifact hash mismatch: {name}"
        )
    required = (
        "environment.json",
        "source_references.json",
        "summary.json",
        "summary.csv",
        "evaluation.log",
        "checkpoint_verification.json",
        "requirements_freeze.txt",
        "figures/cross_seed_summary.png",
    )
    require(all((root / p).is_file() for p in required), "Required artifact missing")
    refs = load_json(root / "source_references.json")
    checkpoint_report = load_json(root / "checkpoint_verification.json")
    require(checkpoint_report["status"] == "PASS", "Checkpoint verification missing")
    require(
        refs["training_run_kind"] == "full" or cfg["run_kind"] == "smoke",
        "Smoke checkpoints in full run",
    )
    if require_full:
        from cowbias.evaluation.evaluator import STAGE2

        require(
            refs["stage2"] == STAGE2
            and refs["bundle_verification"]["sha256"] == STAGE2["bundle_sha256"]
            and refs["bundle_verification"]["status"] == "PASS"
            and refs["bundle_verification"]["original_members_verified"] == 57,
            "Original Stage 2 bundle provenance missing",
        )
    figure_sources = load_json(root / "figure_sources.json")
    require(
        set(figure_sources) == {str(p.relative_to(root)) for p in root.rglob("*.png")},
        "Figure-source coverage mismatch",
    )
    for sources in figure_sources.values():
        require(
            bool(sources)
            and all(manifest["files"].get(name) == digest for name, digest in sources.items()),
            "Figure source hash mismatch",
        )
    per_seed, total_rows, total_pairs, total_attr = {}, 0, 0, 0
    n = cfg["test_images_per_model_seed"]
    for seed in seeds:
        folder = root / f"seed_{seed}"
        dsref = refs["datasets"][str(seed)]
        for file, key in [
            ("manifest.csv", "manifest_sha256"),
            ("dataset_config.json", "config_sha256"),
        ]:
            require(
                hashlib.sha256((folder / file).read_bytes()).hexdigest() == dsref[key],
                "Dataset provenance mismatch",
            )
        dsconfig = load_json(folder / "dataset_config.json")
        require(
            dsconfig["seed"] == seed
            and dsconfig["generator_version"] == dsref["generator_version"],
            "Dataset seed/generator mismatch",
        )
        audit = load_json(folder / "audit.json")
        require(
            audit["status"] == "PASS"
            and audit["leakage"]["cross_split_overlaps"] == 0
            and audit["determinism"]["record_assignments_replayed"]
            and audit["counterfactual"]["pairs"] == n // 2,
            "Dataset audit invariant failed",
        )
        snap = refs["dataset_identity"][str(seed)]
        equal(audit, snap["audit"], "audit snapshot")
        for name, key in [
            ("sources.csv", "source_catalog_sha256"),
            ("source_allocation.json", "source_allocation_sha256"),
        ]:
            require(
                hashlib.sha256((folder / name).read_bytes()).hexdigest() == snap[key],
                "Source identity mismatch",
            )
        expected = [
            r
            for r in read_table(folder / "manifest.csv")
            if r["split"] == "test" and r["variant"] == "shared_eval"
        ]
        ids = [r["image_id"] for r in expected]
        require(len(ids) == n and len(set(ids)) == n, "Test manifest count")
        by_id = {r["image_id"]: r for r in expected}
        rows = read_table(folder / "predictions.csv")
        require(len(rows) == 2 * n, "Prediction row count")
        keys = set()
        for r in rows:
            for k in ("seed", "y_true", "predicted", "correct"):
                require(str(int(r[k])) == r[k], f"Invalid integer {k}")
                r[k] = int(r[k])
            for k in ("logit", "prob_cow"):
                r[k] = float(r[k])
                require(math.isfinite(r[k]), "Nonfinite prediction")
            require(
                r["seed"] == seed and r["run_id"] == cfg["run_id"], "Prediction run/seed mismatch"
            )
            require(
                r["model"] in MODELS and r["image_id"] in by_id, "Unexpected prediction model/ID"
            )
            require(
                r["y_true"] in (0, 1) and r["predicted"] in (0, 1) and r["correct"] in (0, 1),
                "Invalid binary label",
            )
            require(
                0 <= r["prob_cow"] <= 1
                and r["predicted"] == int(r["logit"] >= 0)
                and r["correct"] == int(r["y_true"] == r["predicted"]),
                "Threshold/correctness mismatch",
            )
            sigmoid = 1 / (1 + math.exp(-abs(r["logit"])))
            sigmoid = sigmoid if r["logit"] >= 0 else 1 - sigmoid
            require(math.isclose(sigmoid, r["prob_cow"], abs_tol=1e-7), "Sigmoid mismatch")
            source = by_id[r["image_id"]]
            require(
                all(
                    r[k] == source[k]
                    for k in ("pair_id", "foreground_id", "background_label", "object_label")
                ),
                "Prediction manifest mismatch",
            )
            require(r["y_true"] == int(source["object_label"] == "cow"), "Class mapping mismatch")
            require(
                r["manifest_sha256"] == dsref["manifest_sha256"],
                "Prediction manifest hash mismatch",
            )
            checkpoint = refs["checkpoints"][f"{seed}/{r['model']}"]
            require(
                r["checkpoint_sha256"] == checkpoint["sha256"],
                "Prediction checkpoint hash mismatch",
            )
            require(
                checkpoint["metadata"]["seed"] == seed
                and checkpoint["metadata"]["variant"] == r["model"]
                and checkpoint["metadata"]["selected_state_sha256"]
                == checkpoint["selected_state_sha256"],
                "Checkpoint metadata mismatch",
            )
            key = r["model"], r["image_id"]
            require(key not in keys, "Duplicate prediction ID")
            keys.add(key)
        require(
            [(r["model"], r["image_id"]) for r in rows] == [(m, i) for m in MODELS for i in ids],
            "Prediction ordering/coverage mismatch",
        )
        metric = {"seed": seed, "data_origin": "synthetic", "models": {}}
        all_pairs = []
        for model in MODELS:
            arm = [r for r in rows if r["model"] == model]
            pairs, cf = independently_pair(arm)
            require(len(pairs) == n // 2, "Counterfactual count")
            all_pairs += pairs
            metric["models"][model] = independently_classify(arm) | {"counterfactual": cf}
            require(
                all(g["total"] == n // 4 for g in metric["models"][model]["groups"].values()),
                "Group denominator mismatch",
            )
        check_table(
            read_table(folder / "counterfactual_pairs.csv"), all_pairs, "counterfactual pairs"
        )
        comparisons, comp_summary = independently_compare(rows)
        check_table(
            read_table(folder / "model_disagreements.csv"), comparisons, "model disagreements"
        )
        metric["comparison"] = comp_summary
        with np.load(folder / "test_inputs.npz", allow_pickle=False) as f:
            require(f["image_ids"].tolist() == ids, "Input identity mismatch")
            masks, images = f["masks"].copy(), f["images"].copy()
        with np.load(folder / "test_inputs.npz", allow_pickle=False) as f:
            original_images, original_masks = f["source_images"], f["source_masks"]
        size = refs["training_config"]["architecture"]["input_size"]
        source_size = dsconfig["config"]["dataset"]["image_size"]
        require(
            original_images.shape == (n, source_size, source_size, 3)
            and original_masks.shape == (n, source_size, source_size),
            "Original input dimensions",
        )
        for i in range(n):
            resized_image = np.array(
                Image.fromarray(original_images[i]).resize((size, size), Image.Resampling.BILINEAR)
            )
            resized_mask = (
                np.array(
                    Image.fromarray(original_masks[i]).resize(
                        (size, size), Image.Resampling.NEAREST
                    )
                ).astype(np.float32)
                / 255
            )
            require(
                np.array_equal(images[i], resized_image) and np.array_equal(masks[i], resized_mask),
                "Input preprocessing/mask alignment mismatch",
            )
        require(
            images.shape == (n, size, size, 3)
            and images.dtype == np.uint8
            and masks.shape == (n, size, size)
            and np.isfinite(masks).all()
            and (masks >= 0).all()
            and (masks <= 1).all(),
            "Invalid saved input/mask",
        )
        # Replay held-out source rendering directly from the archived allocation, so
        # correct-shaped masks cannot silently be shifted or exchanged between images.
        from cowbias.data.generate import compose, render_background, render_foreground
        from cowbias.data.splits import Source

        catalog = {r["source_id"]: r for r in read_table(folder / "sources.csv")}
        source_cache = {}

        def source_pixels(
            source_id, source_cache=source_cache, catalog=catalog, source_size=source_size
        ):
            if source_id not in source_cache:
                r = catalog[source_id]
                source = Source(
                    **{
                        k: int(r[k]) if k == "seed" else r[k]
                        for k in (
                            "source_id",
                            "family_id",
                            "kind",
                            "label",
                            "split",
                            "seed",
                            "foreground_id",
                        )
                    }
                )
                source_cache[source_id] = (
                    render_foreground(source, source_size)
                    if source.kind == "foreground"
                    else render_background(source, source_size)
                )
            return source_cache[source_id]

        for i, record in enumerate(expected):
            fg = source_pixels(record["foreground_id"])
            bg = source_pixels(record["background_asset_id"])
            require(
                np.array_equal(original_images[i], np.array(compose(fg, bg)))
                and np.array_equal(original_masks[i], np.array(fg.getchannel("A"))),
                "Archived held-out image/mask source replay mismatch",
            )
        attributes = read_table(folder / "attribution.csv")
        require(len(attributes) == 2 * n, "Attribution row count")
        metric["attribution"] = {}
        for m in MODELS:
            with np.load(folder / f"{m}_heatmaps.npz", allow_pickle=False) as f:
                require(f["image_ids"].tolist() == ids, "Heatmap identity mismatch")
                heat = f["heatmaps"].copy()
            require(
                heat.shape == (n, size, size)
                and np.isfinite(heat).all()
                and (heat >= 0).all()
                and (heat <= 1).all(),
                "Invalid heatmaps",
            )
            arm = [r for r in attributes if r["model"] == m]
            require([r["image_id"] for r in arm] == ids, "Attribution IDs/order mismatch")
            valid = []
            cp = refs["checkpoints"][f"{seed}/{m}"]
            for i, r in enumerate(arm):
                require(
                    int(r["seed"]) == seed
                    and r["run_id"] == cfg["run_id"]
                    and r["checkpoint_sha256"] == cp["sha256"]
                    and r["manifest_sha256"] == dsref["manifest_sha256"],
                    "Attribution provenance mismatch",
                )
                target = int(by_id[r["image_id"]]["object_label"] == "cow")
                require(
                    int(r["target"]) == target and r["score"] == ("logit" if target else "-logit"),
                    "Attribution target mismatch",
                )
                require(
                    r["state_before"] == r["state_after"] == cp["selected_state_sha256"],
                    "Attribution model mutation",
                )
                require(
                    r["layer"] == "features.7"
                    and math.isfinite(float(r["gradient_l1"]))
                    and float(r["gradient_l1"]) >= 0,
                    "Invalid Grad-CAM layer/gradient",
                )
                mass = float(heat[i].astype(np.float64).sum())
                area = float(masks[i].astype(np.float64).mean())
                equal(float(r["heatmap_mass"]), mass, "attribution mass")
                equal(float(r["foreground_area_fraction"]), area, "mask area")
                require((r["zero_mass"] == "True") == (mass == 0), "Zero heatmap flag")
                if mass == 0:
                    require(
                        all(
                            r[k] == ""
                            for k in (
                                "foreground_attribution_share",
                                "background_attribution_share",
                                "foreground_attribution_enrichment",
                            )
                        ),
                        "Undefined attribution must be empty",
                    )
                else:
                    fg = float((heat[i].astype(np.float64) * masks[i]).sum() / mass)
                    bg = float(
                        (heat[i].astype(np.float64) * (1 - masks[i].astype(np.float64))).sum()
                        / mass
                    )
                    equal(float(r["foreground_attribution_share"]), fg, "foreground share")
                    equal(float(r["background_attribution_share"]), bg, "background share")
                    equal(fg + bg, 1.0, "share sum")
                    equal(float(r["foreground_attribution_enrichment"]), fg / area, "enrichment")
                    valid.append(
                        {
                            "foreground_attribution_share": fg,
                            "background_attribution_share": bg,
                            "foreground_attribution_enrichment": fg / area,
                            "foreground_area_fraction": area,
                        }
                    )
            metric["attribution"][m] = {
                "total": len(arm),
                "zero_mass_count": len(arm) - len(valid),
                "nonzero_count": len(valid),
                **{
                    f"mean_{k}": avg([v[k] for v in valid]) if valid else None
                    for k in (
                        "foreground_attribution_share",
                        "background_attribution_share",
                        "foreground_attribution_enrichment",
                        "foreground_area_fraction",
                    )
                },
            }
        equal(load_json(folder / "metrics.json"), metric, f"seed {seed} metrics")
        examples = read_table(folder / "gradcam_examples.csv")
        fixed = [r for r in examples if r["selection"] == "fixed_identity"]
        first_pairs = {
            min(r["pair_id"] for r in expected if r["object_label"] == label)
            for label in ("cow", "non_cow")
        }
        require(
            [r["image_id"] for r in fixed]
            == [
                r["image_id"]
                for label in ("cow", "non_cow")
                for r in expected
                if r["object_label"] == label and r["pair_id"] in first_pairs
            ],
            "Representative sampling mismatch",
        )
        for r in examples:
            require(
                r["image_id"] in ids
                and all(
                    r[k] == by_id[r["image_id"]][k]
                    for k in ("pair_id", "foreground_id", "object_label", "background_label")
                ),
                "Example identity mismatch",
            )
        for figure in (
            "accuracy_by_group",
            "confusion_matrix",
            "counterfactual_examples",
            "gradcam_comparison",
            "background_sensitivity",
        ):
            with Image.open(folder / f"figures/{figure}.png") as image:
                image.verify()
        per_seed[str(seed)] = metric
        total_rows += len(rows)
        total_pairs += len(all_pairs)
        total_attr += len(attributes)
    summary = load_json(root / "summary.json")
    equal(summary["per_seed"], per_seed, "summary per-seed")
    require(
        summary["seeds"] == sorted(seeds)
        and summary["sd_ddof"] == 1
        and summary["data_origin"] == "synthetic",
        "Summary scope mismatch",
    )

    def flat(m):
        return {
            "overall_accuracy": m["overall_accuracy"],
            "worst_group_accuracy": m["worst_group_accuracy"],
            **{f"group_accuracy/{g}": m["groups"][g]["accuracy"] for g in GROUPS},
            "background_flip_rate": m["counterfactual"]["background_flip_rate"],
            "paired_correct_rate": m["counterfactual"]["paired_correct_rate"],
            "mean_absolute_probability_difference": m["counterfactual"]["probability_sensitivity"][
                "mean"
            ],
        }

    flat_seed = {s: {m: flat(per_seed[str(s)]["models"][m]) for m in MODELS} for s in sorted(seeds)}
    stats = []
    for key in flat_seed[min(seeds)]["baseline"]:
        for model in (*MODELS, "treatment_minus_baseline"):
            values = {
                str(s): flat_seed[s]["treatment"][key] - flat_seed[s]["baseline"][key]
                if model == "treatment_minus_baseline"
                else flat_seed[s][model][key]
                for s in sorted(seeds)
            }
            stats.append(
                {
                    "model": model,
                    "metric": key,
                    "n_seeds": len(seeds),
                    "mean": avg(list(values.values())),
                    "sample_sd": sd(list(values.values())),
                    "values": values,
                }
            )
    equal(summary["statistics"], stats, "summary statistics")
    csv_stats = read_table(root / "summary.csv")
    for r, e in zip(csv_stats, stats, strict=True):
        require(
            r["model"] == e["model"]
            and r["metric"] == e["metric"]
            and int(r["n_seeds"]) == len(seeds),
            "Summary CSV identity",
        )
        equal(float(r["mean"]), e["mean"], "summary CSV mean")
        equal(float(r["sample_sd"]) if r["sample_sd"] else None, e["sample_sd"], "summary CSV SD")
    require(
        total_rows == manifest["prediction_rows"] == 2 * n * len(seeds)
        and total_pairs == manifest["pairs"] == n * len(seeds),
        "Global totals mismatch",
    )
    return {
        "status": "PASS",
        "verified_utc": datetime.now(UTC).isoformat(),
        "method": "independent saved CSV arithmetic, paired identities, source/input hashes, raw heatmap/mask recomputation",
        "run_id": cfg["run_id"],
        "seeds": seeds,
        "prediction_rows": total_rows,
        "model_pair_evaluations": total_pairs,
        "attribution_rows": total_attr,
        "files_verified": len(manifest["files"]),
        "full": cfg["run_kind"] == "full",
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Independently verify saved Stage 3 scientific artifacts"
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--require-full", action="store_true")
    args = p.parse_args(argv)
    try:
        report = verify_evaluation(args.run, require_full=args.require_full)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f"Evaluation verification failed: {exc}")
        return 1
