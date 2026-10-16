"""Re-audit regenerated datasets and match archived Stage 2 source identities."""

import argparse
from pathlib import Path

from cowbias.evaluation.evaluator import verify_identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--training-run", type=Path, required=True)
p.add_argument("--data-run", type=Path, required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
a = p.parse_args()
try:
    _, refs, audits = verify_identity(a.training_run, a.data_run, a.seeds)
    for seed in a.seeds:
        print(
            f"PASS seed {seed}: {refs[str(seed)]['manifest_sha256']}; zero source leakage; "
            f"{audits[seed].report['counterfactual']['pairs']} genuine pairs"
        )
except (ValueError, OSError, KeyError) as exc:
    p.exit(1, f"Dataset identity verification failed: {exc}\n")
