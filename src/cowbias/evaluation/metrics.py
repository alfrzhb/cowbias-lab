"""Strict prediction schema and held-out classification arithmetic."""

import math
from collections import Counter

GROUPS = ("cow/grass", "cow/beach", "non_cow/grass", "non_cow/beach")
MODELS = ("baseline", "treatment")
FIELDS = (
    "run_id",
    "seed",
    "model",
    "image_id",
    "pair_id",
    "y_true",
    "background_label",
    "predicted",
    "prob_cow",
    "correct",
    "foreground_id",
    "object_label",
    "logit",
    "checkpoint_sha256",
    "manifest_sha256",
)


def validate_predictions(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("Empty predictions")
    result, keys = [], set()
    for original in rows:
        if not set(FIELDS) <= original.keys():
            raise ValueError("Missing prediction columns")
        r = original.copy()
        for k in ("seed", "y_true", "predicted", "correct"):
            if str(r[k]) not in ("0", "1") and k != "seed":
                raise ValueError(f"Invalid binary {k}")
            if isinstance(r[k], float) and not r[k].is_integer():
                raise ValueError(f"Invalid integer {k}")
            r[k] = int(r[k])
        for k in ("logit", "prob_cow"):
            r[k] = float(r[k])
            if not math.isfinite(r[k]):
                raise ValueError(f"Nonfinite {k}")
        if not 0 <= r["prob_cow"] <= 1:
            raise ValueError("Probability outside [0,1]")
        expected_prob = (
            1 / (1 + math.exp(-r["logit"]))
            if r["logit"] >= 0
            else math.exp(r["logit"]) / (1 + math.exp(r["logit"]))
        )
        if not math.isclose(expected_prob, r["prob_cow"], abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError("Probability/logit mismatch")
        if r["predicted"] != int(r["logit"] >= 0):
            raise ValueError("Fixed threshold mismatch")
        if r["correct"] != int(r["predicted"] == r["y_true"]):
            raise ValueError("Incorrect correctness flag")
        if r["object_label"] not in ("cow", "non_cow") or r["y_true"] != int(
            r["object_label"] == "cow"
        ):
            raise ValueError("Object/label mismatch")
        if r["background_label"] not in ("grass", "beach") or r["model"] not in MODELS:
            raise ValueError("Unsupported background/model")
        if any(not r[k] for k in ("run_id", "image_id", "pair_id", "foreground_id")):
            raise ValueError("Missing identity")
        for k in ("checkpoint_sha256", "manifest_sha256"):
            if len(r[k]) != 64 or any(c not in "0123456789abcdef" for c in r[k]):
                raise ValueError(f"Invalid {k}")
        key = r["seed"], r["model"], r["image_id"]
        if key in keys:
            raise ValueError("Duplicate prediction key")
        keys.add(key)
        result.append(r)
    return result


def classification(rows: list[dict]) -> dict:
    rows = validate_predictions(rows)
    if len({(r["seed"], r["model"]) for r in rows}) != 1:
        raise ValueError("Classification requires one seed/model")
    counts = Counter(f"{r['object_label']}/{r['background_label']}" for r in rows)
    hits = Counter(f"{r['object_label']}/{r['background_label']}" for r in rows if r["correct"])
    if any(counts[g] == 0 for g in GROUPS):
        raise ValueError("Missing group: zero denominator")
    groups = {
        g: {"correct": hits[g], "total": counts[g], "accuracy": hits[g] / counts[g]} for g in GROUPS
    }
    matrix = [
        [sum(r["y_true"] == y and r["predicted"] == p for r in rows) for p in (0, 1)]
        for y in (0, 1)
    ]
    return {
        "total": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "overall_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "groups": groups,
        "worst_group_accuracy": min(g["accuracy"] for g in groups.values()),
        "confusion_matrix": matrix,
        "class_order": ["non_cow", "cow"],
        "confusion_convention": "rows actual, columns predicted",
    }
