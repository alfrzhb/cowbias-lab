"""Identity-based, read-only Stage 3 access; no Torch or serialized Python objects."""

import csv
import json
import math
import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cowbias.dashboard.prepare import ARCHIVE_SHA256, RUN_ID, SOURCE, digest, receipt_path

MODELS = ("baseline", "treatment")
GROUPS = ("cow/grass", "cow/beach", "non_cow/grass", "non_cow/beach")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def json_file(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")),
    )


def table(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        require(
            reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
            f"Invalid CSV header: {Path(path).name}",
        )
        rows = list(reader)
    require(
        rows and all(None not in r and None not in r.values() for r in rows),
        f"Empty or malformed CSV: {Path(path).name}",
    )
    return rows


def indexed(rows, keys):
    result = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        k = k[0] if len(keys) == 1 else k
        require(k not in result, f"Duplicate ID: {k}")
        result[k] = r
    return result


def discover_runs(root=None):
    root = Path(root or os.environ.get("COWBIAS_EVAL_ROOT", "artifacts/runs"))
    if not root.is_dir():
        return []
    if (root / "evaluation_config.json").is_file():
        return [root.resolve()]
    return sorted(
        {
            p.parent.resolve()
            for pattern in ("*/evaluation_config.json", "*/*/evaluation_config.json")
            for p in root.glob(pattern)
        }
    )


def fingerprint(root):
    root = Path(root).resolve()
    paths = [p for p in root.rglob("*") if p.is_file()] + [receipt_path(root)]
    return tuple(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_ino)
        for p in sorted(paths)
    )


def safe_file(root, name):
    p = (root / name).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), f"Unsafe or missing file: {name}")
    return p


@dataclass(frozen=True)
class Run:
    root: Path
    config: dict
    references: dict
    summary: dict
    receipt: dict

    @property
    def seeds(self):
        return self.config["seeds"]

    @property
    def fixture(self):
        return self.receipt["context"] == "development_fixture"

    @property
    def files(self):
        return self.receipt["files"]

    def export(self, name):
        require(name in self.files, "Export not in verified inventory")
        path = safe_file(self.root, name)
        require(digest(path) == self.files[name], f"Export integrity changed: {name}")
        return path.read_bytes()


def load_run(root, *, allow_fixture=False):
    root = Path(root).resolve()
    receipt = json_file(receipt_path(root))
    cfg = json_file(root / "evaluation_config.json")
    verification = json_file(root / "verification.json")
    require(receipt["schema_version"] == cfg["schema_version"] == 1, "Unsupported schema")
    require(
        json_file(root / "run_status.json")["status"] == "complete"
        and verification["status"] == receipt["verification"]["status"] == "PASS",
        "Incomplete or unverified run",
    )
    require(
        cfg["run_id"]
        == receipt["run_id"]
        == verification["run_id"]
        == receipt["verification"]["run_id"],
        "Receipt/run mismatch",
    )
    require(
        cfg["data_origin"] == "synthetic"
        and cfg["decision_rule"] == "logit >= 0"
        and cfg["probability_threshold"] == 0.5
        and cfg["target_class"] == "true_class"
        and cfg["sd_ddof"] == 1,
        "Scientific protocol mismatch",
    )
    if receipt["context"] == "development_fixture":
        require(
            allow_fixture and cfg["run_kind"] == "smoke", "Fixture mode must be explicitly enabled"
        )
    else:
        require(
            receipt["context"] == "verified_full"
            and cfg["run_kind"] == "full"
            and cfg["run_id"] == RUN_ID
            and cfg["code_revision"] == SOURCE
            and receipt["original_archive_sha256"] == ARCHIVE_SHA256
            and receipt["verification"]["full"],
            "Unrecognized full scientific provenance",
        )
    files = receipt["files"]
    require(
        set(files) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()},
        "Prepared inventory changed",
    )
    for name, sha in files.items():
        require(digest(safe_file(root, name)) == sha, f"Artifact integrity mismatch: {name}")
    manifest = json_file(root / "evaluation_manifest.json")
    require(
        all(files.get(k) == v for k, v in manifest["files"].items()), "Receipt/manifest mismatch"
    )
    summary = json_file(root / "summary.json")
    refs = json_file(root / "source_references.json")
    require(
        cfg["seeds"]
        and len(set(cfg["seeds"])) == len(cfg["seeds"])
        and summary["seeds"] == sorted(cfg["seeds"]),
        "Seed scope mismatch",
    )
    for seed in cfg["seeds"]:
        require(
            json_file(root / f"seed_{seed}/metrics.json") == summary["per_seed"][str(seed)],
            "Summary/per-seed mismatch",
        )
        for model in MODELS:
            cp = refs["checkpoints"][f"{seed}/{model}"]
            require(
                cp["metadata"]["seed"] == seed
                and cp["metadata"]["variant"] == model
                and cp["metadata"]["selected_state_sha256"] == cp["selected_state_sha256"],
                "Incompatible checkpoint metadata",
            )
    return Run(root, cfg, refs, summary, receipt)


@dataclass(frozen=True)
class SeedData:
    run: Run
    seed: int
    manifest: dict
    test_ids: tuple
    predictions: dict
    pairs: dict
    members: dict
    disagreements: dict
    attribution: dict
    examples: tuple
    metrics: dict

    def image_id(self, pair, background):
        require(pair in self.members, f"Missing pair for seed {self.seed}: {pair}")
        require(background in self.members[pair], f"Missing pair background: {background}")
        return self.members[pair][background]

    def prediction(self, image, model):
        require((model, image) in self.predictions, f"Missing prediction: {image}/{model}")
        return self.predictions[model, image]

    def pair_options(self, label="All", outcome="All", flips="All"):
        return sorted(
            pair
            for pair, members in self.members.items()
            if (label == "All" or self.manifest[members["grass"]]["object_label"] == label)
            and (
                outcome == "All"
                or any(self.disagreements[i]["outcome"] == outcome for i in members.values())
            )
            and (flips == "All" or self.pairs[flips, pair]["flip"])
        )

    def distributions(self):
        counts = Counter(
            (r["split"], r["variant"], r["object_label"] + "/" + r["background_label"])
            for r in self.manifest.values()
        )
        return [
            {"split": s, "variant": v, "group": g, "count": n}
            for (s, v, g), n in sorted(counts.items())
        ]


def load_seed(run, seed):
    require(seed in run.seeds, f"Seed unavailable: {seed}")
    folder = run.root / f"seed_{seed}"
    manifest = indexed(table(folder / "manifest.csv"), ["image_id"])
    require(all(int(r["seed"]) == seed for r in manifest.values()), "Manifest seed mismatch")
    ids = tuple(
        r["image_id"]
        for r in manifest.values()
        if r["split"] == "test" and r["variant"] == "shared_eval"
    )
    require(len(ids) == run.config["test_images_per_model_seed"], "Test count mismatch")
    members = {}
    for image in ids:
        r = manifest[image]
        require(
            r["object_label"] in ("cow", "non_cow") and r["background_label"] in ("grass", "beach"),
            "Invalid group",
        )
        group = members.setdefault(r["pair_id"], {})
        require(r["background_label"] not in group, "Duplicate pair background")
        group[r["background_label"]] = image
    for pair, group in members.items():
        require(set(group) == {"grass", "beach"}, f"Incomplete pair: {pair}")
        g, b = (manifest[group[k]] for k in ("grass", "beach"))
        require(
            g["foreground_id"] == b["foreground_id"] and g["object_label"] == b["object_label"],
            "Foreground mismatch",
        )
    pred = indexed(table(folder / "predictions.csv"), ["model", "image_id"])
    attr = indexed(table(folder / "attribution.csv"), ["model", "image_id"])
    require(
        set(pred) == set(attr) == {(m, i) for m in MODELS for i in ids},
        "Prediction/attribution coverage",
    )
    dsref = run.references["datasets"][str(seed)]
    require(
        digest(folder / "manifest.csv") == dsref["manifest_sha256"], "Manifest provenance mismatch"
    )
    for (model, image), r in pred.items():
        for k in ("seed", "y_true", "predicted", "correct"):
            r[k] = int(r[k])
        for k in ("prob_cow", "logit"):
            r[k] = float(r[k])
            require(math.isfinite(r[k]), "Nonfinite prediction")
        cp = run.references["checkpoints"][f"{seed}/{model}"]
        require(
            r["run_id"] == run.config["run_id"]
            and r["seed"] == seed
            and r["checkpoint_sha256"] == cp["sha256"]
            and r["manifest_sha256"] == dsref["manifest_sha256"],
            "Prediction provenance",
        )
        require(
            all(
                r[k] == manifest[image][k]
                for k in ("pair_id", "foreground_id", "object_label", "background_label")
            ),
            "Prediction identity",
        )
        require(
            r["y_true"] == int(manifest[image]["object_label"] == "cow")
            and r["predicted"] == int(r["logit"] >= 0)
            and 0 <= r["prob_cow"] <= 1
            and r["correct"] == int(r["predicted"] == r["y_true"]),
            "Prediction labels",
        )
        a = attr[model, image]
        require(
            a["run_id"] == r["run_id"]
            and int(a["seed"]) == seed
            and a["checkpoint_sha256"] == cp["sha256"]
            and a["manifest_sha256"] == r["manifest_sha256"]
            and int(a["target"]) == r["y_true"]
            and a["score"] == ("logit" if r["y_true"] else "-logit")
            and a["state_before"] == a["state_after"] == cp["selected_state_sha256"],
            "Attribution target/provenance",
        )
        require(a["zero_mass"] in ("True", "False"), "Invalid zero-map flag")
        a["zero_mass"] = a["zero_mass"] == "True"
        for k in (
            "heatmap_mass",
            "foreground_attribution_share",
            "background_attribution_share",
            "foreground_area_fraction",
            "foreground_attribution_enrichment",
        ):
            a[k] = None if a[k] == "" else float(a[k])
            require(a[k] is None or math.isfinite(a[k]), "Nonfinite attribution")
        if a["zero_mass"]:
            require(
                all(
                    a[k] is None
                    for k in (
                        "foreground_attribution_share",
                        "background_attribution_share",
                        "foreground_attribution_enrichment",
                    )
                ),
                "Zero shares must be undefined",
            )
    pairs = indexed(table(folder / "counterfactual_pairs.csv"), ["model", "pair_id"])
    require(set(pairs) == {(m, p) for m in MODELS for p in members}, "Pair coverage mismatch")
    for (model, pair), r in pairs.items():
        require(r["run_id"] == run.config["run_id"] and int(r["seed"]) == seed, "Pair run/seed")
        for bg in ("grass", "beach"):
            require(r[f"{bg}_image_id"] == members[pair][bg], "Pair image mismatch")
            p = pred[model, members[pair][bg]]
            require(
                r["foreground_id"] == p["foreground_id"]
                and int(r[f"{bg}_predicted"]) == p["predicted"]
                and float(r[f"{bg}_prob_cow"]) == p["prob_cow"],
                "Pair prediction mismatch",
            )
        for k in ("flip", "both_correct", "grass_correct", "beach_correct"):
            r[k] = int(r[k])
        r["delta_prob_cow"] = float(r["delta_prob_cow"])
    disagreements = indexed(table(folder / "model_disagreements.csv"), ["image_id"])
    require(set(disagreements) == set(ids), "Disagreement coverage")
    for image, r in disagreements.items():
        require(
            r["run_id"] == run.config["run_id"] and int(r["seed"]) == seed, "Disagreement run/seed"
        )
        for model in MODELS:
            require(
                int(r[f"{model}_predicted"]) == pred[model, image]["predicted"]
                and float(r[f"{model}_prob_cow"]) == pred[model, image]["prob_cow"],
                "Disagreement prediction",
            )
    metrics = json_file(folder / "metrics.json")
    for model in MODELS:
        m = metrics["models"][model]
        require(
            m["worst_group_accuracy"] == min(g["accuracy"] for g in m["groups"].values()),
            "WGA mismatch",
        )
        for g in GROUPS:
            require(
                m["groups"][g]["total"]
                == sum(
                    manifest[i]["object_label"] + "/" + manifest[i]["background_label"] == g
                    for i in ids
                ),
                "Group denominator mismatch",
            )
    examples = tuple(table(folder / "gradcam_examples.csv"))
    require(
        all(
            e["image_id"] in ids and e["pair_id"] == manifest[e["image_id"]]["pair_id"]
            for e in examples
        ),
        "Example identity",
    )
    return SeedData(
        run, seed, manifest, ids, pred, pairs, members, disagreements, attr, examples, metrics
    )


def load_arrays(data, model=None):
    require(model is None or model in MODELS, "Invalid model")
    path = (
        data.run.root
        / f"seed_{data.seed}"
        / (f"{model}_heatmaps.npz" if model else "test_inputs.npz")
    )
    with zipfile.ZipFile(path) as z:
        require(sum(e.file_size for e in z.infolist()) <= 128 * 1024 * 1024, "NPZ too large")
    with np.load(path, allow_pickle=False) as f:
        ids = f["image_ids"]
        require(
            ids.dtype.kind in "US" and ids.tolist() == list(data.test_ids), "NPZ ID order mismatch"
        )
        keys = ["heatmaps"] if model else ["images", "masks", "source_images", "source_masks"]
        result = {k: f[k] for k in keys}
    n = len(ids)
    model_size = data.run.references["training_config"]["architecture"]["input_size"]
    source_size = json_file(path.parent / "dataset_config.json")["config"]["dataset"]["image_size"]
    for k, a in result.items():
        require(a.dtype.kind in "uf" and np.isfinite(a).all(), f"Invalid {k} dtype/values")
        size = source_size if k.startswith("source_") else model_size
        shape = (n, size, size, 3) if "images" in k else (n, size, size)
        require(a.shape == shape, f"Invalid {k} shape")
        byte = "images" in k or k == "source_masks"
        require((a >= 0).all() and (a <= (255 if byte else 1)).all(), f"Invalid {k} range")
        require(a.dtype == np.uint8 if byte else a.dtype.kind == "f", f"Invalid {k} dtype")
        a.setflags(write=False)
    if model:
        for i, image in enumerate(data.test_ids):
            require(
                (not result["heatmaps"][i].any()) == data.attribution[model, image]["zero_mass"],
                "Zero-map mismatch",
            )
    result["index"] = {image: i for i, image in enumerate(data.test_ids)}
    return result


def image_array(arrays, image, key="source_images"):
    require(image in arrays["index"], f"Missing image: {image}")
    return arrays[key][arrays["index"][image]]
