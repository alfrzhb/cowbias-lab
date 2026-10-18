"""Focused dashboard/source consistency check. Never runs model inference."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cowbias.dashboard.data import (
    MODELS,
    image_array,
    load_arrays,
    load_run,
    load_seed,
    require,
    table,
)


def check_run(root, fixture=False):
    run = load_run(root, allow_fixture=fixture)
    counts = {"predictions": 0, "model_pair_records": 0, "gradcam_maps": 0, "exports": 0}
    for seed in run.seeds:
        data = load_seed(run, seed)
        inputs = load_arrays(data)
        for row in table(run.root / f"seed_{seed}/predictions.csv"):
            p = data.prediction(row["image_id"], row["model"])
            require(
                p["prob_cow"] == float(row["prob_cow"]) and p["logit"] == float(row["logit"]),
                "Prediction mismatch",
            )
            require(
                data.image_id(row["pair_id"], row["background_label"]) == row["image_id"],
                "Pair identity mismatch",
            )
            counts["predictions"] += 1
        counts["model_pair_records"] += len(data.pairs)
        for model in MODELS:
            metric = data.metrics["models"][model]
            matrix = np.zeros((2, 2), dtype=int)
            heat = load_arrays(data, model)
            for image in data.test_ids:
                p = data.prediction(image, model)
                matrix[p["y_true"], p["predicted"]] += 1
                a = data.attribution[model, image]
                h = image_array(heat, image, "heatmaps").astype(np.float64)
                mask = image_array(inputs, image, "masks").astype(np.float64)
                require(float(h.sum()) == a["heatmap_mass"], "Attribution mass mismatch")
                if h.sum():
                    require(
                        np.isclose(
                            (h * mask).sum() / h.sum(),
                            a["foreground_attribution_share"],
                            rtol=1e-12,
                        ),
                        "Foreground attribution mismatch",
                    )
                else:
                    require(
                        a["foreground_attribution_share"] is None,
                        "Zero attribution must be undefined",
                    )
                counts["gradcam_maps"] += 1
            require(matrix.tolist() == metric["confusion_matrix"], "Confusion mismatch")
            require(
                matrix.trace() / matrix.sum() == metric["overall_accuracy"], "Accuracy mismatch"
            )
        comparison = data.metrics["comparison"]
        require(
            sum(r["outcome"] == "baseline_only" for r in data.disagreements.values())
            == comparison["new_errors"],
            "Regression mismatch",
        )
        require(
            sum(r["outcome"] == "treatment_only" for r in data.disagreements.values())
            == comparison["treatment_corrected"],
            "Correction mismatch",
        )
    for name in run.files:
        if not name.endswith((".csv", ".json", ".png")):
            continue
        raw = run.export(name)
        require(raw == (run.root / name).read_bytes(), "Export bytes mismatch")
        if name.endswith(".json"):
            json.loads(raw)
        elif name.endswith(".csv"):
            table(run.root / name)
        else:
            with Image.open(io.BytesIO(raw)) as img:
                img.verify()
        counts["exports"] += 1
    return {
        "status": "PASS",
        "run_id": run.config["run_id"],
        "context": run.receipt["context"],
        "archive_sha256": run.receipt["original_archive_sha256"],
        "counts": counts,
        "evaluation_source": run.config["code_revision"],
        "seeds": run.seeds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = check_run(args.run, args.fixture)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        require(
            not args.output.resolve().is_relative_to(args.run.resolve()),
            "Output must be outside scientific run",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0
