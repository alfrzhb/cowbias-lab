"""Restore and verify the exact original Stage 3 ZIP without changing scientific files."""

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

RUN_ID = "stage3-full-38874819983-1"
SOURCE = "f4c517e64c20ed3498a656ff31dcb67bdbc5e69d"
ARCHIVE_SHA256 = "75b42453cde76bb6b5e7f5c1db903908b592439949a324e79f8ff17186dbb0d4"
REPO = "alfrzhb/cowbias-lab"
WORKFLOW = 38874819983
ARTIFACT = 12035480745
MAX_FILE = 128 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def receipt_path(root):
    root = Path(root)
    return root.parent / (root.name + ".dashboard-receipt.json")


def safe_extract(archive, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing existing extraction directory")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > 5000 or sum(e.file_size for e in entries) > MAX_TOTAL:
            raise ValueError("ZIP exceeds extraction limits")
        seen = set()
        for e in entries:
            name = e.filename.rstrip("/") if e.is_dir() else e.filename
            p = PurePosixPath(name)
            mode = stat.S_IFMT(e.external_attr >> 16)
            if (
                not name
                or "\\" in name
                or ":" in name
                or p.is_absolute()
                or any(x in ("", ".", "..") for x in name.split("/"))
                or str(p) in seen
                or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                or e.flag_bits & 1
                or e.file_size > MAX_FILE
                or e.file_size > 1024 * max(e.compress_size, 1)
            ):
                raise ValueError(f"Unsafe or duplicate ZIP member: {e.filename!r}")
            seen.add(str(p))
        output.mkdir(parents=True, exist_ok=False)
        for e in entries:
            dest = output / e.filename
            if e.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(e) as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)


def prepare_run(root, *, fixture=False, archive_hash=None):
    from cowbias.evaluation.verification import verify_evaluation

    root = Path(root).resolve()
    receipt = receipt_path(root)
    if receipt.exists():
        raise ValueError("Existing preparation receipt; choose a new destination")
    cfg = json.loads((root / "evaluation_config.json").read_text())
    if fixture:
        if cfg["run_kind"] != "smoke":
            raise ValueError("Development fixtures must be genuine smoke evaluations")
    elif (
        archive_hash != ARCHIVE_SHA256 or cfg["run_id"] != RUN_ID or cfg["code_revision"] != SOURCE
    ):
        raise ValueError("Full demo requires pinned original Stage 3 evidence")
    report = verify_evaluation(root, require_full=not fixture)
    result = {
        "schema_version": 1,
        "run_id": cfg["run_id"],
        "context": "development_fixture" if fixture else "verified_full",
        "original_archive_sha256": archive_hash,
        "workflow_run_id": None if fixture else WORKFLOW,
        "artifact_id": None if fixture else ARTIFACT,
        "verification": report,
        "prepared_utc": datetime.now(UTC).isoformat(),
        "files": {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    }
    with receipt.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Existing original ZIP; otherwise use authorized gh"
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/downloaded-stage3"))
    parser.add_argument("--fixture-run", type=Path, help="Explicit genuine smoke fixture")
    args = parser.parse_args(argv)
    try:
        if args.fixture_run:
            if args.archive:
                raise ValueError("--fixture-run and --archive are mutually exclusive")
            result = prepare_run(args.fixture_run, fixture=True)
        else:
            if args.output.exists():
                raise ValueError("Refusing existing output; choose a new directory")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
                temp = Path(temp)
                archive = args.archive
                if archive is None:
                    endpoint = f"repos/{REPO}/actions/artifacts/{ARTIFACT}"
                    info = json.loads(
                        subprocess.check_output(
                            ["gh", "api", endpoint], text=True, stderr=subprocess.PIPE
                        )
                    )
                    if (
                        info["workflow_run"]["id"] != WORKFLOW
                        or info["name"] != "stage3-scientific-results"
                        or info["expired"]
                        or info["digest"] != "sha256:" + ARCHIVE_SHA256
                    ):
                        raise ValueError("Actions workflow/artifact identity mismatch")
                    archive = temp / "original.zip"
                    with archive.open("xb") as f:
                        subprocess.run(
                            ["gh", "api", endpoint + "/zip"],
                            stdout=f,
                            stderr=subprocess.PIPE,
                            check=True,
                        )
                elif not archive.is_file():
                    raise ValueError(f"Original archive not found: {archive}")
                archive_hash = digest(archive)
                if archive_hash != ARCHIVE_SHA256:
                    raise ValueError("Original ZIP SHA-256 mismatch")
                staging = temp / "extracted"
                safe_extract(archive, staging)
                result = prepare_run(staging / RUN_ID, archive_hash=archive_hash)
                staging.rename(args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
        EOFError,
        ImportError,
    ) as exc:
        if isinstance(exc, (subprocess.SubprocessError, FileNotFoundError)):
            message = "GitHub retrieval unavailable. Install/authenticate gh with repository/actions read access, or supply --archive."
        elif isinstance(exc, ImportError):
            message = "Install the eval extra for the existing Stage 3 scientific verifier."
        else:
            message = str(exc)
        print(f"Dashboard preparation failed: {message}")
        return 1
