"""Render reviewable held-out result tables directly from independently verified artifacts."""

import argparse
import json
import os
from pathlib import Path

from cowbias.evaluation.verification import verify_evaluation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--github-summary", action="store_true")
a = p.parse_args()
report = verify_evaluation(a.run, require_full=True)
s = json.loads((a.run / "summary.json").read_text())
lines = [
    f"## CowBias Lab — SYNTHETIC held-out evaluation `{a.run.name}`",
    "",
    f"Independent verification: PASS; {report['prediction_rows']} predictions; {report['model_pair_evaluations']} model-pairs; {report['attribution_rows']} genuine Grad-CAM maps.",
    "",
    "| Seed | Model | Accuracy | WGA | Flip rate | Paired correct | Mean abs ΔP(cow) |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for seed in s["seeds"]:
    for model in ("baseline", "treatment"):
        m = s["per_seed"][str(seed)]["models"][model]
        cf = m["counterfactual"]
        lines.append(
            f"| {seed} | {model} | {m['overall_accuracy']:.6f} | {m['worst_group_accuracy']:.6f} | {cf['background_flip_rate']:.6f} | {cf['paired_correct_rate']:.6f} | {cf['probability_sensitivity']['mean']:.6f} |"
        )
lines += ["", "| Metric | Model | Mean | Sample SD |", "|---|---|---:|---:|"]
for r in s["statistics"]:
    lines.append(f"| {r['metric']} | {r['model']} | {r['mean']:.6f} | {r['sample_sd']:.6f} |")
lines += [
    "",
    "Only three seeds, varying dataset allocation/rendering and initialization jointly. Pairs are foreground clusters. No significance claim. Grad-CAM is descriptive. No real-photo generalization established.",
    "",
    f"Source: `{json.loads((a.run / 'evaluation_config.json').read_text())['code_revision']}`.",
    "Stage 2: https://github.com/alfrzhb/cowbias-lab/actions/runs/38827143763/artifacts/12015781574",
]
text = "\n".join(lines) + "\n"
print(text)
if a.github_summary:
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as f:
        f.write(text)
