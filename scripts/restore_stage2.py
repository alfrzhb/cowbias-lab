"""Retrieve the original authorized Actions ZIP and validate before safe extraction."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from cowbias.evaluation.evaluator import STAGE2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.output.exists() or args.archive.exists():
            raise ValueError("Refusing existing archive or output")
        repo = "alfrzhb/cowbias-lab"
        endpoint = f"repos/{repo}/actions/artifacts/{STAGE2['artifact_id']}"
        info = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))
        if (
            info["workflow_run"]["id"] != STAGE2["workflow_run_id"]
            or info["name"] != "stage2-trained-models"
            or info["expired"]
            or info["digest"] != "sha256:" + STAGE2["bundle_sha256"]
        ):
            raise ValueError("Stage 2 artifact metadata mismatch")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with args.archive.open("xb") as f:
            subprocess.run(["gh", "api", endpoint + "/zip"], stdout=f, check=True)
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        if digest != STAGE2["bundle_sha256"]:
            raise ValueError("Original Stage 2 ZIP SHA-256 mismatch")
        with zipfile.ZipFile(args.archive) as z:
            for entry in z.infolist():
                path = Path(entry.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe ZIP entry")
            args.output.mkdir(parents=True, exist_ok=False)
            z.extractall(args.output)
        args.archive.with_suffix(".retrieval.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"PASS: original ZIP {digest}; extracted {args.output}")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"Stage 2 retrieval failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
