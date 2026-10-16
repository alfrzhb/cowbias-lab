"""Seed-level arithmetic; the unit of cross-seed replication is a complete run."""

import statistics

from cowbias.evaluation.metrics import GROUPS, MODELS

LIMITATIONS = [
    "Three seeds provide limited evidence; no confirmatory significance claim.",
    "Seeds vary source allocation/rendering and initialization jointly.",
    "480 images per seed represent 240 paired foreground identities, not 480 independent animals.",
    "No row-independent confidence intervals; no bootstrap or p-values are reported.",
    "Procedural synthetic illustrations; real-photo generalization is untested.",
]


def flatten(metrics: dict) -> dict:
    return {
        "overall_accuracy": metrics["overall_accuracy"],
        "worst_group_accuracy": metrics["worst_group_accuracy"],
        **{f"group_accuracy/{g}": metrics["groups"][g]["accuracy"] for g in GROUPS},
        "background_flip_rate": metrics["counterfactual"]["background_flip_rate"],
        "paired_correct_rate": metrics["counterfactual"]["paired_correct_rate"],
        "mean_absolute_probability_difference": metrics["counterfactual"][
            "probability_sensitivity"
        ]["mean"],
    }


def aggregate(per_seed: dict) -> tuple[dict, list[dict]]:
    if not per_seed:
        raise ValueError("No seeds to aggregate")
    seeds = sorted(map(int, per_seed))
    flat = {s: {m: flatten(per_seed[str(s)]["models"][m]) for m in MODELS} for s in seeds}
    records = []
    for metric in flat[seeds[0]]["baseline"]:
        for model in (*MODELS, "treatment_minus_baseline"):
            values = [
                (flat[s]["treatment"][metric] - flat[s]["baseline"][metric])
                if model == "treatment_minus_baseline"
                else flat[s][model][metric]
                for s in seeds
            ]
            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_seeds": len(seeds),
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                    "values": dict(zip(map(str, seeds), values, strict=True)),
                }
            )
    return {
        "schema_version": 1,
        "data_origin": "synthetic",
        "seeds": seeds,
        "sd_ddof": 1,
        "statistics": records,
        "per_seed": per_seed,
        "limitations": LIMITATIONS,
    }, records
