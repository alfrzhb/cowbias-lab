"""True foreground pairs and same-image model comparisons; no arbitrary matching."""

import statistics
from collections import defaultdict

from cowbias.evaluation.metrics import validate_predictions

OUTCOMES = ("both_correct", "grass_correct_only", "beach_correct_only", "both_incorrect")
MODEL_OUTCOMES = ("both_correct", "baseline_only", "treatment_only", "both_incorrect")


def pairs_from_predictions(rows: list[dict]) -> tuple[list[dict], dict]:
    rows = validate_predictions(rows)
    if len({(r["seed"], r["model"]) for r in rows}) != 1:
        raise ValueError("Pairs require one seed/model")
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pair_id"]].append(r)
    pairs, foregrounds = [], set()
    for pair_id, members in grouped.items():
        if len(members) != 2 or {r["background_label"] for r in members} != {"grass", "beach"}:
            raise ValueError("Pair must contain exactly grass and beach")
        g, b = sorted(members, key=lambda r: r["background_label"] != "grass")
        if g["foreground_id"] != b["foreground_id"] or g["y_true"] != b["y_true"]:
            raise ValueError("Pair foreground/class mismatch")
        if g["foreground_id"] in foregrounds:
            raise ValueError("Foreground used in multiple pairs")
        foregrounds.add(g["foreground_id"])
        outcome = (
            "both_correct"
            if g["correct"] and b["correct"]
            else "grass_correct_only"
            if g["correct"]
            else "beach_correct_only"
            if b["correct"]
            else "both_incorrect"
        )
        pairs.append(
            {k: g[k] for k in ("run_id", "seed", "model", "foreground_id", "y_true")}
            | {
                "pair_id": pair_id,
                "grass_image_id": g["image_id"],
                "beach_image_id": b["image_id"],
                "grass_predicted": g["predicted"],
                "beach_predicted": b["predicted"],
                "grass_prob_cow": g["prob_cow"],
                "beach_prob_cow": b["prob_cow"],
                "flip": int(g["predicted"] != b["predicted"]),
                "both_correct": int(g["correct"] and b["correct"]),
                "grass_correct": g["correct"],
                "beach_correct": b["correct"],
                "delta_prob_cow": abs(g["prob_cow"] - b["prob_cow"]),
                "outcome": outcome,
            }
        )
    deltas = [p["delta_prob_cow"] for p in pairs]
    return pairs, {
        "total_pairs": len(pairs),
        "flipped_pairs": sum(p["flip"] for p in pairs),
        "background_flip_rate": sum(p["flip"] for p in pairs) / len(pairs),
        "both_correct_pairs": sum(p["both_correct"] for p in pairs),
        "paired_correct_rate": sum(p["both_correct"] for p in pairs) / len(pairs),
        "outcomes": {k: sum(p["outcome"] == k for p in pairs) for k in OUTCOMES},
        "probability_sensitivity": {
            "mean": statistics.mean(deltas),
            "median": statistics.median(deltas),
            "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "maximum": max(deltas),
            "minimum": min(deltas),
        },
    }


def compare_models(rows: list[dict]) -> tuple[list[dict], dict]:
    rows = validate_predictions(rows)
    if len({r["seed"] for r in rows}) != 1:
        raise ValueError("Comparison requires one seed")
    arms = {
        m: {r["image_id"]: r for r in rows if r["model"] == m} for m in ("baseline", "treatment")
    }
    if not arms["baseline"] or arms["baseline"].keys() != arms["treatment"].keys():
        raise ValueError("Model image IDs differ")
    output = []
    for image_id, a in arms["baseline"].items():
        b = arms["treatment"][image_id]
        for k in (
            "seed",
            "run_id",
            "foreground_id",
            "pair_id",
            "y_true",
            "background_label",
            "object_label",
            "manifest_sha256",
        ):
            if a[k] != b[k]:
                raise ValueError(f"Same-image metadata mismatch: {k}")
        outcome = (
            "both_correct"
            if a["correct"] and b["correct"]
            else "baseline_only"
            if a["correct"]
            else "treatment_only"
            if b["correct"]
            else "both_incorrect"
        )
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
                "baseline_correct": a["correct"],
                "treatment_correct": b["correct"],
                "probability_change": b["prob_cow"] - a["prob_cow"],
                "disagree": int(a["predicted"] != b["predicted"]),
                "outcome": outcome,
            }
        )

    def summarize(items):
        counts = {k: sum(r["outcome"] == k for r in items) for k in MODEL_OUTCOMES}
        n = len(items)
        return {
            "total": n,
            "outcomes": counts,
            "outcome_rates": {k: v / n for k, v in counts.items()},
            "model_disagreement_rate": sum(r["disagree"] for r in items) / n,
            "treatment_corrected": counts["treatment_only"],
            "new_errors": counts["baseline_only"],
            "net_correct_change": counts["treatment_only"] - counts["baseline_only"],
            "mean_probability_change": statistics.mean(r["probability_change"] for r in items),
        }

    summary = summarize(output)
    summary["by_class"] = {
        label: summarize([r for r in output if r["object_label"] == label])
        for label in ("cow", "non_cow")
    }
    summary["by_background"] = {
        bg: summarize([r for r in output if r["background_label"] == bg])
        for bg in ("grass", "beach")
    }
    return output, summary
