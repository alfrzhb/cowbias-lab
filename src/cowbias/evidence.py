"""Pinned original archives: byte preservation, recovery and member-level traceability."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from cowbias.dashboard.prepare import safe_extract

REPOSITORY = "alfrzhb/cowbias-lab"
RUN_ID = "stage3-full-38874819983-1"
ARCHIVES = {
    "stage2-trained-models": {
        "workflow_run_id": 38827143763,
        "artifact_id": 12015781574,
        "sha256": "d99ae16c94d57810e7cad11d25f62be061df943202ba4b9a59d10721012597be",
        "source_commit": "881be87ef2d83100b51bfb069e1fa75097eb9688",
        "expires_at": "2027-01-14T00:02:25Z",
    },
    "stage3-scientific-results": {
        "workflow_run_id": 38874819983,
        "artifact_id": 12035480745,
        "sha256": "75b42453cde76bb6b5e7f5c1db903908b592439949a324e79f8ff17186dbb0d4",
        "source_commit": "f4c517e64c20ed3498a656ff31dcb67bdbc5e69d",
        "expires_at": "2027-01-14T08:36:30Z",
    },
    "stage4-dashboard-review": {
        "workflow_run_id": 39140400975,
        "artifact_id": 12141178453,
        "sha256": "78b5f33e9741808dcbcb55c8d014b00acfc84e5bf43b39ec2019d0bd6e2dd18b",
        "source_commit": "4fc877ae5350a0af9d70f6f6c9533e97276bd254",
        "expires_at": "2027-01-16T15:25:18Z",
    },
}
PRESERVATION = {
    "status": "PENDING",
    "reason": "Repository is public; no authorized durable access-controlled archive destination "
    "is available. Original verified ZIP copies and recovery tooling exist; Actions retention "
    "and a temporary workspace are not permanent preservation.",
    "durable_location": None,
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_archive(path, expected_sha256):
    path = Path(path)
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"Original archive SHA-256 mismatch or missing: {path.name}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Archive CRC failure: {path.name}")
        names = [e.filename for e in archive.infolist() if not e.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def verify_extracted(root, members, prefix=""):
    root = Path(root).resolve()
    for name, digest in members.items():
        if prefix and not name.startswith(prefix):
            continue
        relative = name[len(prefix) :]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Extracted evidence missing or differs from original: {relative}")


def metadata_matches(info, name):
    expected = ARCHIVES[name]
    if (
        info.get("id") != expected["artifact_id"]
        or info.get("name") != name
        or info.get("expired") is not False
        or info.get("digest") != "sha256:" + expected["sha256"]
        or info.get("workflow_run", {}).get("id") != expected["workflow_run_id"]
        or info.get("workflow_run", {}).get("head_sha") != expected["source_commit"]
    ):
        raise ValueError(f"Actions metadata incompatible: {name}")


def archive_manifest(folder):
    result = {"repository": REPOSITORY, "preservation": PRESERVATION, "archives": {}}
    for name, spec in ARCHIVES.items():
        path = Path(folder) / (name + ".zip")
        members = verify_archive(path, spec["sha256"])
        result["archives"][name] = {
            **spec,
            "archive_name": path.name,
            "bytes": path.stat().st_size,
            "workflow_url": f"https://github.com/{REPOSITORY}/actions/runs/"
            f"{spec['workflow_run_id']}",
            "members": members,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/evidence"))
    parser.add_argument("--download", action="store_true", help="Use authenticated gh API")
    parser.add_argument("--extract", action="store_true", help="Extract to NEW named directories")
    parser.add_argument("--manifest", type=Path, required=True, help="NEW inventory JSON")
    args = parser.parse_args(argv)
    try:
        if args.manifest.exists():
            raise ValueError("Archive manifest output collision")
        args.directory.mkdir(parents=True, exist_ok=True)
        for name, spec in ARCHIVES.items():
            path = args.directory / (name + ".zip")
            if args.download:
                endpoint = f"repos/{REPOSITORY}/actions/artifacts/{spec['artifact_id']}"
                info = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))
                metadata_matches(info, name)
                if path.exists():
                    raise ValueError(f"Archive output collision: {path.name}")
                with path.open("xb") as handle:
                    subprocess.run(["gh", "api", endpoint + "/zip"], stdout=handle, check=True)
                path.with_suffix(".retrieval.json").write_text(json.dumps(info, indent=2) + "\n")
            members = verify_archive(path, spec["sha256"])
            if args.extract:
                destination = args.directory / name
                safe_extract(path, destination)
                verify_extracted(destination, members)
        result = archive_manifest(args.directory)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print("PASS: three original ZIPs and member inventories; durable preservation PENDING")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print(f"Evidence recovery failed: {exc}")
        return 1
