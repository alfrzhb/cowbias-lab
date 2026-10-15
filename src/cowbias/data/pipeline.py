"""Offline source allocation, rendering, compositing, auditing and CLI."""

import argparse
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from cowbias.config import ExperimentConfig, load_config
from cowbias.data.dataset import MANIFEST_FIELDS, read_csv, write_csv
from cowbias.data.generate import GENERATOR_VERSION, compose, render_background, render_foreground
from cowbias.data.splits import allocate_sources, background_assignments

LOGGER = logging.getLogger("cowbias")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_dataset(config: ExperimentConfig, root: Path, seed: int) -> dict:
    """Create a new dataset directory. Existing paths always fail, even when empty."""
    from cowbias.data.audit import audit_dataset

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    LOGGER.info("dataset_started", extra={"seed": seed, "path": str(root)})
    random.seed(seed)
    sources = allocate_sources(config.dataset, seed)
    # Freeze the split allocation on disk before any source is rendered.
    (root / "source_allocation.json").write_text(
        json.dumps([s.as_dict() for s in sources], indent=2) + "\n", encoding="utf-8"
    )
    foregrounds, backgrounds, catalog = {}, {}, []
    for folder in ("sources/foreground", "sources/background", "images", "masks"):
        (root / folder).mkdir(parents=True)
    for source in sources:
        image = (
            render_foreground(source, config.dataset.image_size)
            if source.kind == "foreground"
            else render_background(source, config.dataset.image_size)
        )
        relative = f"sources/{source.kind}/{source.source_id}.png"
        image.save(root / relative)
        entry = {**source.as_dict(), "file_path": relative, "sha256": sha256_file(root / relative)}
        catalog.append(entry)
        if source.kind == "foreground":
            foregrounds[source.source_id] = image
        else:
            backgrounds[source.foreground_id, source.label] = (source, image)
    write_csv(root / "sources.csv", catalog)
    records = []
    for i, (fg, bg_label, variant, pair_id) in enumerate(
        background_assignments(sources, config.dataset, seed)
    ):
        bg, bg_image = backgrounds[fg.source_id, bg_label]
        image_id = f"s{seed}-image-{i:06d}"
        image_path, mask_path = f"images/{image_id}.png", f"masks/{image_id}.png"
        compose(foregrounds[fg.source_id], bg_image).save(root / image_path)
        foregrounds[fg.source_id].getchannel("A").save(root / mask_path)
        records.append(
            dict(
                zip(
                    MANIFEST_FIELDS,
                    (
                        image_id,
                        fg.source_id,
                        bg.source_id,
                        fg.label,
                        bg.label,
                        fg.split,
                        variant,
                        pair_id,
                        seed,
                        image_path,
                        mask_path,
                        fg.family_id,
                        bg.family_id,
                    ),
                    strict=True,
                )
            )
        )
    write_csv(root / "manifest.csv", records, MANIFEST_FIELDS)
    snapshot = {
        "seed": seed,
        "config": config.as_dict(),
        "generator_version": GENERATOR_VERSION,
        "data_origin": "synthetic",
    }
    (root / "dataset_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    report = audit_dataset(root, config, seed)
    create_contact_sheet(root)
    LOGGER.info(
        "dataset_complete",
        extra={
            "seed": seed,
            "records": len(records),
            "path": str(root),
            "status": report["status"],
        },
    )
    return report


def code_provenance() -> dict:
    """Git is optional; source hashes identify executed code even in a dirty tree."""
    package_root = Path(__file__).resolve().parents[1]
    result = {
        "code_revision": None,
        "working_tree_dirty": None,
        "source_sha256": {
            str(p.relative_to(package_root)): sha256_file(p)
            for p in sorted(package_root.rglob("*.py"))
        },
    }
    try:
        revision = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if revision.returncode == 0:
            dirty = subprocess.run(
                ["git", "-C", str(package_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            result["code_revision"] = revision.stdout.strip()
            result["working_tree_dirty"] = bool(dirty.stdout.strip())
    except OSError:
        pass  # Source hashes are still available when Git is not installed.
    return result


def create_contact_sheet(root: Path) -> None:
    records = read_csv(root / "manifest.csv", MANIFEST_FIELDS)
    # Four groups x six independent examples, then four true counterfactual pairs.
    selected: list[tuple[str, list[dict]]] = []
    for label in ("cow", "non_cow"):
        for bg in ("grass", "beach"):
            group = [
                r
                for r in records
                if r["split"] == "val"
                and r["object_label"] == label
                and r["background_label"] == bg
            ]
            selected.append((f"val: {label} / {bg}", group[:6]))
    for label in ("cow", "non_cow"):
        pairs = sorted(
            {r["pair_id"] for r in records if r["split"] == "test" and r["object_label"] == label}
        )[:2]
        for pair in pairs:
            selected.append((f"test pair: {label}", [r for r in records if r["pair_id"] == pair]))
    sheet = Image.new("RGB", (6 * 144, len(selected) * 166 + 35), "#f2efe7")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 10), "CowBias Lab | SYNTHETIC | labels outside exported sample pixels", fill="black"
    )
    for y, (title, samples) in enumerate(selected):
        top = 35 + y * 166
        draw.text((8, top), title, fill="black")
        for x, row in enumerate(samples):
            with Image.open(root / row["file_path"]) as img:
                sheet.paste(
                    img.resize((136, 136), Image.Resampling.NEAREST), (x * 144 + 4, top + 18)
                )
    sheet.save(root / "contact_sheet.png")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"level": record.levelname, "event": record.getMessage()}
        for name in ("seed", "records", "path", "status"):
            if hasattr(record, name):
                data[name] = getattr(record, name)
        return json.dumps(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and audit CowBias Lab synthetic datasets"
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML configuration")
    parser.add_argument("--output", type=Path, help="New run directory; refuses existing paths")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Subset of configured seeds (default: all)"
    )
    parser.add_argument("--audit", type=Path, help="Re-audit an existing seed_N/data directory")
    args = parser.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    LOGGER.handlers = [handler]
    LOGGER.setLevel(logging.INFO)
    try:
        config = load_config(args.config)
        if args.audit is not None:
            if args.output is not None or args.seeds is not None:
                raise ValueError("--audit cannot be combined with --output or --seeds")
            from cowbias.data.audit import audit_dataset

            report = audit_dataset(args.audit, config)
            print(f"{report['status']}: audit verified {args.audit.resolve()}")
            return 0
        seeds = args.seeds if args.seeds is not None else config.training.seeds
        if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config.training.seeds):
            raise ValueError("--seeds must be a unique subset of training.seeds")
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        root = args.output or Path(config.artifacts.root) / run_id
        root.mkdir(parents=True, exist_ok=False)
        metadata = {
            "config": config.as_dict(),
            "seeds": list(seeds),
            **code_provenance(),
            "python": platform.python_version(),
            "dependencies": {p: version(p) for p in ("numpy", "Pillow", "PyYAML")},
            "config_sha256": sha256_file(args.config),
            "data_origin": "synthetic",
            "generator_version": GENERATOR_VERSION,
            "run_id": root.name,
        }
        (root / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for seed in seeds:
            generate_dataset(config, root / f"seed_{seed}" / "data", seed)
        print(
            f"PASS: {len(seeds)} audited seed(s), {2 * config.dataset.train_size + config.dataset.val_size + config.dataset.test_size} images/seed. Artifacts: {root.resolve()}"
        )
        return 0
    except (ValueError, OSError) as exc:
        LOGGER.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
