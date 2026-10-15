# CowBias Lab

A reproducible synthetic experiment for studying spurious correlations between animal identity and background. The motivating case is a cow classifier that relies on grass and fails on a beach.

**Status: Stage 1 implemented.** The package generates and audits actual images, masks, source catalogs and counterfactual pairs. Training, model evaluation, Grad-CAM and Streamlit remain future stages; no model accuracy is claimed.

Requirements: [docs/PRD.md](docs/PRD.md). Implementation sequence: [docs/PLAN.md](docs/PLAN.md).

## Install and generate

Python **3.11+** is required; verification used Python 3.12.14. Run from the repository root:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your OS.
python -m pip install -e '.[dev]'
python scripts/prepare_data.py --config config/experiment.yaml
python -m pytest -q
ruff check .
```

Generation runs all configured seeds (`42`, `123`, `2026`), producing **2,720 composites per seed** and 8,160 in total. Core dependencies are NumPy, Pillow and PyYAML; later-stage dependencies are retained as optional `ml` and `app` extras. Stage 1 requires no GPU or PyTorch.

```bash
python scripts/prepare_data.py --help

# Optional: one configured seed and an explicit, NEW output directory.
python scripts/prepare_data.py --config config/experiment.yaml \
  --seeds 42 --output artifacts/runs/my-stage1-run
```

Default output is `artifacts/runs/<UTC-time>-<unique-suffix>/`. Relative paths, including `artifacts.root` and `--output`, resolve from the current working directory. Every generation gets a new directory. An existing explicit output directory is rejected without modifying its contents. Failed partial runs remain available for diagnosis; rerun into a new directory.

## Dataset and intervention

| Dataset | Cow + grass | Cow + beach | Non-cow + grass | Non-cow + beach | Total |
|---|---:|---:|---:|---:|---:|
| Baseline train | 485 | 15 | 15 | 485 | 1,000 |
| Treatment train | 250 | 250 | 250 | 250 | 1,000 |
| Shared validation | 60 | 60 | 60 | 60 | 240 |
| Shared test | 120 | 120 | 120 | 120 | 480 |

`non_cow` is a horse surrogate. Every dataset is class-balanced. Baseline has a 97:3 majority/minority background ratio for each class. Treatment uses the **exact same 1,000 training foreground instances**, with the same pixels, pose, geometry, scale and placement; foreground record order is also identical, so matching future DataLoader shuffle seeds can preserve minibatch membership. Only background assignment changes. Assignment is randomized with exact 50:50 background quotas per label.

There is one shared validation set and one shared test set per seed for both future models. Test has 240 unique foregrounds (120 per class), each on both grass and beach with one `pair_id`. Validation has 240 independent, unpaired foregrounds.

The renderer varies coat palette, patches, body proportions, legs/pose, scale, placement, orientation and rotation. Both species share nuisance-parameter distributions and the same rendering/compositing method. Landscapes vary texture, color, lighting, horizon, clouds and shoreline or vegetation. Image files contain no text, watermarks or embedded class metadata. Contact sheet captions are outside the sample pixels.

## Source isolation and audit

Original foreground and background IDs are allocated to train/val/test **before rendering or compositing**. Each original is generated from an independent stream keyed by seed and source ID. Each original has a family ID; future derivatives must inherit that family and split. The current renderer uses continuous procedural variations rather than a finite library of shared animal or texture originals.

Each foreground has a grass asset and a beach asset allocated inside its split. Training variants can intentionally reuse these assets within train. Source IDs, family IDs and identical source-file hashes may never cross splits. Unused background originals stay in the source catalog and are audited too.

Generation runs the audit automatically and fails nonzero on violations. Re-audit an existing seed directory:

```bash
python scripts/prepare_data.py --config config/experiment.yaml \
  --audit artifacts/runs/my-stage1-run/seed_42/data
```

Audit checks exact counts and all four groups, class balance, unique composite IDs and paths, source/family/hash overlap, matching training foreground order, catalog/manifest agreement, file readability and dimensions, source checksums, and source regeneration from the recorded seed. It independently replays `allocate_sources()` and `background_assignments()` from the saved configuration and seed, checking every record's identity/order, foreground, label, background category and asset, split, variant and pair ID. A background swap that preserves quotas and is correctly recomposited still fails if it disagrees with the seeded assignments.

The audit reconstructs **every composite** from the original RGBA foreground and RGB background and compares its full pixels. Test masks and opaque foreground pixels must be identical within each pair, while visible background pixels must change. Soft alpha boundary pixels naturally blend with the new background; equality is verified against exact alpha compositing rather than requiring those blended pixels to stay unchanged.

`audit.json` records PASS or FAIL. A failing re-audit replaces a stale PASS with the actual failure and raises an informative error. Invalid datasets are never silently repaired. Inspect `contact_sheet.png` for representative held-out groups and true test pairs; automated integrity checks do not replace visual inspection.

## Output artifacts

```text
artifacts/runs/<run_id>/
  run_config.json
  seed_42/data/
    dataset_config.json
    source_allocation.json
    sources.csv
    sources/foreground/*.png
    sources/background/*.png
    images/*.png
    masks/*.png
    manifest.csv
    audit.json
    group_distribution.csv
    group_distribution.md
    contact_sheet.png
  seed_123/data/...
  seed_2026/data/...
```

| Artifact | Contents |
|---|---|
| `run_config.json` | Config, seeds, dependency/Python versions, code revision, dirty-tree flag, source-code hashes and input-config SHA-256 |
| `dataset_config.json` | Per-seed configuration, seed, generator version and synthetic origin |
| `source_allocation.json` | Frozen source identities and splits, exported before any image is rendered |
| `sources.csv` | Original source ID, family, type, label, split, seed, asset association, relative path and SHA-256 |
| `images/`, `masks/` | 48×48 RGB composites and matching 8-bit alpha masks; RGBA originals remain in `sources/foreground/` |
| `manifest.csv` | Required image/source/label/split/variant/pair/seed/path fields, plus `mask_path` and both family IDs |
| `audit.json`, distribution tables | Machine-readable invariant results and four-group counts |
| `contact_sheet.png` | Enlarged validation examples and unchanged-foreground test pairs |

All manifest and source paths are relative to the corresponding `data/` directory and are validated against missing files, absolute paths and directory escapes. Generated outputs under `artifacts/` are Git-ignored; keep custom outputs there too.

## Package layout and loading

```text
config/experiment.yaml       # Validated central configuration
scripts/prepare_data.py      # CLI wrapper
src/cowbias/config.py        # Typed immutable config, validation and YAML loader
src/cowbias/data/
  splits.py                 # Source allocation and exact background quotas
  generate.py               # NumPy/Pillow foregrounds, backgrounds, compositing
  dataset.py                # CSV I/O, safe paths and inspection dataset
  pipeline.py               # Generation, provenance, contact sheets and CLI
  audit.py                  # Distribution, leakage, replay and pair verification
tests/                      # Real generation, reproducibility and corruption tests
```

Existing later-stage directories remain reserved. `ManifestDataset` provides framework-independent inspection, including optional model-size resizing:

```python
from cowbias.data.dataset import ManifestDataset

data = ManifestDataset(
    "artifacts/runs/my-stage1-run/seed_42/data",
    split="train",
    variant="baseline",
    image_size=32,
)
sample = data[0]  # image: uint8 HWC; mask: uint8 HW; label: cow=1, non_cow=0
```

Held-out loading uses `variant="shared_eval"`. Training augmentation, normalization, tensor conversion and DataLoader integration belong to Stage 2.

## Fresh remote session and Stage 2 dataset handoff

Generated artifacts are disposable remote-session outputs, not files delivered by a Git clone. Regenerate them when the previous run directory is unavailable. In a fresh remote checkout, select the intended source revision and install the package:

```bash
git clone --branch feat/stage-1-dataset-foundation https://github.com/alfrzhb/cowbias-lab.git
cd cowbias-lab
python -m pip install -e '.[dev]'

# Use a new run directory. Omit --seeds to generate all three configured seeds.
python scripts/prepare_data.py --config config/experiment.yaml \
  --seeds 42 --output artifacts/runs/stage2-data
python scripts/prepare_data.py --config config/experiment.yaml \
  --audit artifacts/runs/stage2-data/seed_42/data
```

After this feature branch is merged, use `main` or the intended dataset-producing commit instead. To reproduce an earlier run, restore its configuration and source revision and install the NumPy, Pillow and PyYAML versions recorded in its `run_config.json` (for example, the initial verified environment used NumPy 2.3.5, Pillow 12.3.0 and PyYAML 6.0.3). Record the Python version too. Pinning a code commit alone does not pin dependencies. The generator version and source identities are unchanged by audit hardening.

Select the run directory **and seed explicitly**; do not silently select the newest folder or mix seeds across the two models. Validate before loading. This example selects data only and performs no training:

```python
from pathlib import Path

from cowbias.config import load_config
from cowbias.data.audit import audit_dataset
from cowbias.data.dataset import ManifestDataset

config = load_config("config/experiment.yaml")
seed = 42
data_root = Path("artifacts/runs/stage2-data") / f"seed_{seed}" / "data"
audit_dataset(data_root, config, seed=seed)

baseline = ManifestDataset(data_root, split="train", variant="baseline",
                           image_size=config.dataset.model_image_size)
treatment = ManifestDataset(data_root, split="train", variant="treatment",
                            image_size=config.dataset.model_image_size)
validation = ManifestDataset(data_root, split="val", variant="shared_eval",
                             image_size=config.dataset.model_image_size)
test = ManifestDataset(data_root, split="test", variant="shared_eval",
                       image_size=config.dataset.model_image_size)
```

An absent/incomplete directory, wrong requested seed, mismatched saved configuration, unsupported generator version/origin, invalid relative path, source/pixel mismatch or seeded assignment mismatch fails the audit. `ManifestDataset` is a lightweight reader; it does not replace this explicit pre-load audit. Preserve the parent `run_config.json` together with each seed's complete `data/` directory if moving artifacts. Keep manifests, allocation, original sources and masks together, rather than copying only composite images. Paths remain relative to `data/`.

The existing output-directory refusal also applies to remote reruns: choose another run directory instead of overwriting an earlier experiment. Dataset regeneration and loading require no cloud storage, Git LFS or previous developer-machine files.

## Continuous integration and visual review

`.github/workflows/ci.yml` runs on PRs targeting `main`, pushes to `main`, and manual dispatch. CPU-only Python 3.11 and 3.12 jobs install `.[dev]`, run pytest (including real dataset generation/audit), and run Ruff. Permissions are limited to `contents: read`; checkout does not persist credentials. Actions are pinned to stable release commits.

CI uses the existing reduced integration fixture rather than generating all three full datasets per job. Successful jobs upload `dataset-review-python-3.11` and `dataset-review-python-3.12` artifacts, retained for seven days. Each contains the fixture contact sheet, audit JSON, group table and dataset configuration. Download these from the workflow run's **Artifacts** section. These are small synthetic previews with test-fixture counts, not the full experiment datasets and not a Stage 2 handoff.

For full runs, review `artifacts/runs/<run_id>/seed_<seed>/data/contact_sheet.png` and `audit.json`. The existing sheet shows all four validation groups and actual paired test images, marks the experiment **SYNTHETIC**, and leaves exported sample pixels unchanged. Bulk images, masks, sources and checkpoints remain excluded from Git.

## Reproducibility and limitations

Use the same configuration, seed, code and dependency versions in a new output directory. Randomness uses stable SHA-256-derived independent NumPy streams; Python's randomized `hash()` is never a seed. Within the same environment, generated per-seed artifacts are byte-identical across output directories. Run IDs and run-level provenance intentionally describe each execution separately. Dependency upgrades, particularly Pillow resampling/encoding, can change pixels or PNG bytes; compare the versions recorded in `run_config.json` before reproducing an older run.

Configuration rejects missing/unsupported fields, invalid labels, invalid integer sizes, non-integral quotas, missing/duplicate seeds and invalid ratios. Sizes are configurable but must preserve balanced groups; treatment remains 50:50 by design.

These are small **synthetic illustrations**, not photographs. Shared drawing grammar, coat patterns and anatomy can be easier to learn than real animal identity. Good visual separation or a future successful synthetic experiment cannot establish real-world cow recognition, and source isolation does not establish generalization to unseen rendering styles. No training or model effectiveness claim is part of Stage 1.

Academic context: Artificial Intelligence, Graduate Informatics, UIN Sunan Kalijaga — Task 3 (14 October 2026).
