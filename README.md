# CowBias Lab

A reproducible synthetic experiment for studying spurious correlations between animal identity and background. The motivating case is a cow classifier that relies on grass and fails on a beach.

**Status: Stages 1–3 implement audited data, frozen CNN training, held-out scientific evaluation, counterfactual analysis, genuine Grad-CAM and independent verification.** See [methodology](docs/methodology.md) and [held-out execution report](docs/stage3-results.md). Streamlit and the final academic PDF remain Stages 4–5. Validation scores below describe checkpoint selection, separately from held-out results.

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

Generation runs all configured seeds (`42`, `123`, `2026`), producing **2,720 composites per seed** and 8,160 in total. Core dependencies are NumPy, Pillow and PyYAML. Stage 2 adds the `ml` extra (PyTorch); `app` remains optional for a future stage. Stage 1 requires no GPU or PyTorch.

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

`non_cow` is a horse surrogate. Every dataset is class-balanced. Baseline has a 97:3 majority/minority background ratio for each class. Treatment uses the **exact same 1,000 training foreground instances**, with the same pixels, pose, geometry, scale and placement; foreground record order is also identical, so matching DataLoader shuffle seeds can preserve minibatch membership. Only background assignment changes. Assignment is randomized with exact 50:50 background quotas per label.

There is one shared validation set and one shared test set per seed for both models. Test has 240 unique foregrounds (120 per class), each on both grass and beach with one `pair_id`. Validation has 240 independent, unpaired foregrounds.

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

`ManifestDataset` provides framework-independent inspection, including optional model-size resizing:

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

Held-out loading uses `variant="shared_eval"`. Stage 2 wraps this reader with deterministic normalization, tensor conversion and DataLoaders; it forbids model access to the test split.

## Fresh remote session and Stage 2 dataset handoff

Generated artifacts are disposable remote-session outputs, not files delivered by a Git clone. Regenerate them when the previous run directory is unavailable. In a fresh remote checkout, select the intended source revision and install the package:

```bash
git clone https://github.com/alfrzhb/cowbias-lab.git
cd cowbias-lab
python -m pip install -e '.[dev]'

# Use a new run directory. Omit --seeds to generate all three configured seeds.
python scripts/prepare_data.py --config config/experiment.yaml \
  --seeds 42 --output artifacts/runs/stage2-data
python scripts/prepare_data.py --config config/experiment.yaml \
  --audit artifacts/runs/stage2-data/seed_42/data
```

For an implementation under review, check out its PR branch; for reproduction, use the artifact's exact source revision. Restore its configuration and the NumPy, Pillow and PyYAML versions recorded in its `run_config.json` (for example, Stage 1 verification used NumPy 2.3.5, Pillow 12.3.0 and PyYAML 6.0.3). Record the Python version too. Pinning a code commit alone does not pin dependencies. The generator version and source identities are unchanged by Stage 2.

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

`.github/workflows/ci.yml` runs on PRs targeting `main`, pushes to `main`, and manual dispatch. CPU-only Python 3.11 and 3.12 jobs install CPU PyTorch and `.[ml,dev]`, run pytest (including actual generation, audit and paired optimization), run Ruff, then run the CPU smoke CLI and independent checkpoint verifier. Permissions are limited to `contents: read`; checkout does not persist credentials. Actions are pinned to stable release commits.

CI uses the existing reduced integration fixture rather than generating all three full datasets per job. Successful jobs upload `dataset-review-python-3.11` and `dataset-review-python-3.12` artifacts, retained for seven days. Each contains the fixture contact sheet, audit JSON, group table and dataset configuration. Download these from the workflow run's **Artifacts** section. These are small synthetic previews with test-fixture counts, not the full experiment datasets and not a Stage 2 handoff.

For full runs, review `artifacts/runs/<run_id>/seed_<seed>/data/contact_sheet.png` and `audit.json`. The existing sheet shows all four validation groups and actual paired test images, marks the experiment **SYNTHETIC**, and leaves exported sample pixels unchanged. Bulk images, masks, sources and checkpoints remain excluded from Git.

## Reproducibility and limitations

Use the same configuration, seed, code and dependency versions in a new output directory. Randomness uses stable SHA-256-derived independent NumPy streams; Python's randomized `hash()` is never a seed. Within the same environment, generated per-seed artifacts are byte-identical across output directories. Run IDs and run-level provenance intentionally describe each execution separately. Dependency upgrades, particularly Pillow resampling/encoding, can change pixels or PNG bytes; compare the versions recorded in `run_config.json` before reproducing an older run.

Configuration rejects missing/unsupported fields, invalid labels, invalid integer sizes, non-integral quotas, missing/duplicate seeds and invalid ratios. Sizes are configurable but must preserve balanced groups; treatment remains 50:50 by design.

These are small **synthetic illustrations**, not photographs. Shared drawing grammar, coat patterns and anatomy can be easier to learn than real animal identity. Good visual separation or a future successful synthetic experiment cannot establish real-world cow recognition, and source isolation does not establish generalization to unseen rendering styles. No training or model effectiveness claim is part of Stage 1.

Academic context: Artificial Intelligence, Graduate Informatics, UIN Sunan Kalijaga — Task 3 (14 October 2026).

## Stage 2: train and verify CNNs

Install the ML extra. CPU-only wheels avoid downloading GPU dependencies:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[ml,dev]'
python -m pytest -q
ruff check .

python scripts/run_experiment.py --config config/experiment.yaml --smoke
python scripts/run_experiment.py --config config/experiment.yaml \
  --seeds 42 123 2026 --device cpu --output artifacts/runs/my-training-run
python scripts/verify_checkpoints.py --run artifacts/runs/my-training-run --require-full
```

Without `--data-run`, the training runner invokes the existing Stage 1 pipeline into `<training_run>/datasets/seed_<seed>/data`. With `--data-run artifacts/runs/my-stage1-run`, it uses that explicit run and re-audits every requested seed before training. It never guesses a latest directory. Missing files, incompatible configuration/seed/generator version, leakage, altered sources, or failed deterministic assignment replay stop training. Both dataset and training outputs refuse existing output directories. Keep all outputs under the ignored `artifacts/` tree.

Smoke mode generates actual images: **200 examples per training arm, 16 shared validation and 16 test composites**, one configured seed and one epoch, on CPU by default. The test composites are audited only. Smoke artifacts carry `run_kind: smoke` and fail `--require-full`; they are not six-model experimental evidence. Full mode uses the central configuration unchanged: 1,000 examples per arm, seven epochs, batch 64, AdamW at 0.001, seeds 42/123/2026. `--device` and `--threads` (default 2) control resources. Use `--help` for the CLI contract. Failures return nonzero and preserve `failure.log`, partial histories and clearly marked incomplete diagnostic states.

`models/cnn.py` defines one compact network: three 3×3 Conv/ReLU/2×2 MaxPool blocks with 16/32/64 channels, adaptive 4×4 pooling, a 64-unit ReLU classifier and **one binary logit**. It uses BCEWithLogitsLoss, `cow=1`, `non_cow=0`, and threshold logit ≥ 0. No pretrained weights, external downloads, dropout or batch normalization are used. The architecture is reconstructed from primitive metadata.

`training/data.py` reads only audited train/validation manifests. Pillow bilinear resizing produces RGB 32×32; float32 CHW tensors are scaled by `(pixel/255 - 0.5)/0.5` per channel. There are no flips, crops, jitter, augmentation, class reweighting or oversampling. Background labels remain metadata for validation group accounting; the CNN receives only the three image channels.

For each seed, `training/experiment.py` creates one initial state and loads its cloned tensors into independent baseline and treatment instances. Every initial tensor is checked for equality and hashed. Python, NumPy, PyTorch CPU/CUDA RNGs are reset before each arm; deterministic algorithms are enabled, cuDNN benchmarking and TF32 are disabled. Independent identically seeded DataLoader generators preserve foreground minibatch membership and order. Every consumed epoch permutation is hashed and compared. Both arms use the same optimizer defaults, loss, schedule, preprocessing, record count and shared validation order. The existing audit plus `training/treatment.py` verifies identical foreground identities/order/labels and masks/geometry. The Stage 1 background assignments are the only intervention.

All seven epochs run. The policy declared before training selects **highest validation worst-group accuracy**, then highest overall validation accuracy, then earliest epoch. WGA is the minimum of all four group accuracies, with every denominator required nonzero. Histories include train/validation losses and accuracy, four validation group accuracies/counts, WGA, learning rate, epoch duration and consumed foreground order hash. Validation runs in eval/inference mode and cannot update weights. The Stage 2 training module performs no test inference, test-based selection or final test analysis. Stage 3 uses a separate evaluation-only test adapter.

## Training artifacts and safe reloading

```text
artifacts/runs/<training_run>/
  run_config.json              # Requested/effective config, seed list, code hashes
  environment.json            # Python/dependency versions, hardware, determinism
  requirements_freeze.txt      # Environment inventory; editable path is session-specific
  dataset_generation.yaml     # Exact configuration for regenerating source datasets
  dataset_references.json     # Per-seed IDs, manifest/config hashes, relative references
  training.log
  run_status.json
  training_summary.json
  verification.json
  dataset_manifests/seed_42/   # Manifest, config, source catalog/allocation, audit, sheet
  seed_42/
    baseline.pt
    treatment.pt
    baseline_history.csv
    treatment_history.csv
    baseline_metadata.json
    treatment_metadata.json
    fairness.json             # Ordered foreground/validation IDs and invariant results
    validation_probe.npz      # Four actual validation inputs, one per group, for reload
  seed_123/...                # Full runs only
  seed_2026/...               # Full runs only
  datasets/...                # Only when generated here; excluded from upload bundle
```

Metadata records run kind, seed/variant, architecture, synthetic origin, dataset run and manifest/config SHA-256, generator version, initialization hash, training/optimizer settings, selected epoch/metrics/policy, dependency/hardware versions, Git revision/dirty flag and package source hashes. Sidecars add checkpoint/history SHA-256 and the reload verification result. Plain `state_dict` tensors and JSON primitives are serialized; loading uses `weights_only=True` and strict architecture/state validation. Only load trusted bundles and verify the recorded checksum:

```python
import json
from pathlib import Path
from cowbias.training.checkpoint import load_checkpoint

folder = Path("artifacts/runs/my-training-run/seed_42")
meta = json.loads((folder / "baseline_metadata.json").read_text())
model, metadata = load_checkpoint(
    folder / "baseline.pt", expected_sha256=meta["checkpoint_sha256"],
    expected_seed=42, expected_variant="baseline", expected_run_kind="full",
)
# model expects the exact documented float32 [N, 3, 32, 32] preprocessing.
```

`verify_checkpoints.py --require-full` requires completed full-mode checkpoints for every configured seed, verifies all six checksums and histories, reconstructs fresh models, checks the declared epoch-selection policy and fairness metadata, and compares finite logits against saved validation probes (`atol=1e-6`, `rtol=1e-5`). These probes let an artifact recipient verify loading without the original filesystem or any test inference.

## Durable remote handoff to Stage 3

`.github/workflows/full-training.yml` generates and audits all three datasets, trains all six models on CPU and verifies them independently. It runs on explicitly scoped changes pushed to `feat/stage-2-cnn-training`, enabling the initial run before the workflow exists on `main`. After merging, **Actions → Full Stage 2 training → Run workflow** supports controlled manual runs. Standard PR CI only trains small fixtures and smoke runs.

Successful full runs upload **`stage2-trained-models`**, retained for **90 days**. Open [Full Stage 2 training runs](https://github.com/alfrzhb/cowbias-lab/actions/workflows/full-training.yml), select a successful run at the intended code revision, and download the artifact from **Artifacts**. The run summary contains actual validation results; check `run_status.json` and `verification.json` after downloading. A failed run can also upload diagnostics, so an artifact's existence alone does not establish success. Archive the downloaded ZIP before retention expiry if longer preservation is needed.

With GitHub CLI authentication in a future remote session:

```bash
gh run download <verified-run-id> --repo alfrzhb/cowbias-lab \
  --name stage2-trained-models --dir artifacts/runs/downloaded-stage2
python scripts/verify_checkpoints.py --run artifacts/runs/downloaded-stage2 --require-full

# Restore the recorded source commit and dependency versions first, then regenerate data.
python scripts/prepare_data.py \
  --config artifacts/runs/downloaded-stage2/dataset_generation.yaml \
  --output artifacts/runs/regenerated-stage1
```

The bundle contains all six checkpoints, histories, metadata, validation probes and small Stage 1 audit/reference snapshots; bulk datasets and model binaries stay out of ordinary Git commits. A source-path reference may point to a prior remote session, so regenerate instead of assuming that path still exists. Match regenerated manifest/config hashes to `dataset_references.json` and run the Stage 1 audit before Stage 3 inference. Snapshots retain source-level hashes and allocation for independent comparison.

Reproduction requires matching code, configuration, seeds and dependency environment. Dataset identities are deterministic; bitwise image encoding and trained weights can change across Pillow/NumPy/PyTorch versions or hardware, even with deterministic algorithms enabled. `requirements_freeze.txt` is an inventory, not a portable editable-project path: install this repository separately and restore dependency versions from `environment.json`. Stage 3 consumes these frozen inputs for held-out group evaluation, counterfactual predictions and explainability. Stage 2 validation evidence alone makes no final claim that the treatment mitigates background bias.

## Verified Stage 2 execution

The CPU experiment at source commit `881be87ef2d83100b51bfb069e1fa75097eb9688`
completed all six seven-epoch training runs. The retained epochs below follow the
predeclared WGA → overall accuracy → earliest epoch policy. These are **validation**
results used for selection, not held-out test results or a final mitigation claim.

- [Full training run 38827143763](https://github.com/alfrzhb/cowbias-lab/actions/runs/38827143763): SUCCESS.
- [Download stage2-trained-models](https://github.com/alfrzhb/cowbias-lab/actions/runs/38827143763/artifacts/12015781574): 3,631,083-byte ZIP; upload and fresh download verified.
- Artifact retention: 90 days; GitHub reports expiry **14 January 2027 at 00:02 UTC**.
- ZIP SHA-256: `d99ae16c94d57810e7cad11d25f62be061df943202ba4b9a59d10721012597be`.
- [Python 3.11 and 3.12 CI](https://github.com/alfrzhb/cowbias-lab/actions/runs/38827190926): **82 tests passed per job**, Ruff passed, real CPU smoke and checkpoint verification passed.
- Downloaded bundle reverified independently: all six checkpoint/history checksums, metadata, validation-only selection, fairness controls and fresh-model validation probe predictions PASS.
- Dataset audits: all three seeds PASS, with deterministic source/assignment replay and zero cross-split source leakage.
- Training environment: Python 3.12.15, PyTorch 2.14.1+cpu, NumPy 2.5.4, Pillow 12.3.0, PyYAML 6.0.3; two CPU threads. Full versions and source hashes are in the bundle.

| Seed | Model | Selected epoch | Validation loss | Validation accuracy | Validation WGA | Reload |
|---|---|---:|---:|---:|---:|---|
| 42 | baseline | 7 | 0.608923 | 0.820833 | 0.350000 | PASS |
| 42 | treatment | 4 | 0.025483 | 0.995833 | 0.983333 | PASS |
| 123 | baseline | 7 | 0.909820 | 0.545833 | 0.050000 | PASS |
| 123 | treatment | 7 | 0.003250 | 1.000000 | 1.000000 | PASS |
| 2026 | baseline | 1 | 1.840064 | 0.500000 | 0.000000 | PASS |
| 2026 | treatment | 4 | 0.046593 | 1.000000 | 1.000000 | PASS |

Seed 2026 baseline selected epoch 1 because all seven epochs tied on both WGA
and overall accuracy; lower validation loss is deliberately not a tie-breaker.
The treatment's observed validation scores do not establish held-out robustness.

Retrieve the exact verified bundle in a new remote session:

```bash
gh run download 38827143763 --repo alfrzhb/cowbias-lab \
  --name stage2-trained-models --dir artifacts/runs/downloaded-stage2
python scripts/verify_checkpoints.py --run artifacts/runs/downloaded-stage2 --require-full
```

The bundle's training run ID is `stage2-full`; it is disambiguated by the Actions
run ID and source commit above. Checkpoint files are under
`seed_<seed>/{baseline,treatment}.pt`. Exact checkpoint SHA-256 values are in
`verification.json`, `training_summary.json` and each model metadata sidecar.
For Stage 3, restore the recorded source/dependencies, regenerate with
`dataset_generation.yaml`, match manifest/config hashes, and re-audit the data
before any held-out inference. Download before expiry for preservation beyond
the configured Actions retention period.


## Stage 3: scientific held-out evaluation

Use Python 3.12 and the recorded CPU dependency versions. The original environment
used Python 3.12.15; patch/hardware differences are recorded and checked with saved
validation-probe numerical tolerances, not described as universal bitwise reproduction.

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.4 Pillow==12.3.0 PyYAML==6.0.3 matplotlib==3.10.8
python -m pip install -e '.[eval,dev]'
python -m pytest -q
python -m ruff check .

# Authorized gh authentication is needed for this private repository.
# Checks workflow/artifact identity, expiry and original ZIP SHA-256 before extraction.
python scripts/restore_stage2.py --archive artifacts/runs/stage2.zip \
  --output artifacts/runs/downloaded-stage2

# The existing Stage 2 verifier writes verification.json: verify a disposable copy
# to preserve the exact input bundle. The evaluator does this automatically too.
python -c "from pathlib import Path; from cowbias.evaluation.evaluator import verify_frozen_run; print(verify_frozen_run(Path('artifacts/runs/downloaded-stage2'), True))"

python scripts/prepare_data.py \
  --config artifacts/runs/downloaded-stage2/dataset_generation.yaml \
  --seeds 42 123 2026 --output artifacts/runs/stage3-regenerated-data
python scripts/verify_dataset_identity.py \
  --training-run artifacts/runs/downloaded-stage2 \
  --data-run artifacts/runs/stage3-regenerated-data --seeds 42 123 2026

python scripts/evaluate_experiment.py \
  --training-run artifacts/runs/downloaded-stage2 \
  --bundle-archive artifacts/runs/stage2.zip \
  --data-run artifacts/runs/stage3-regenerated-data \
  --seeds 42 123 2026 --device cpu --output artifacts/runs/stage3-full
python scripts/verify_evaluation.py --run artifacts/runs/stage3-full --require-full
python scripts/summarize_evaluation.py --run artifacts/runs/stage3-full
```

`--bundle-archive` additionally verifies every extracted input file against the
original archive, preserving the exact six models. Do not edit archived sidecars,
verification files or configuration. Use a new output directory on every execution;
existing output paths fail. `--help` documents explicit paths, seeds, device and
smoke mode. Smoke mode requires one seed and uses actual fixture inference, with
artifacts labeled smoke and rejected by `--require-full`. It never replaces full
scientific findings. Routine CI runs independent fixtures and a training/evaluation
smoke; it never downloads the full bundle.

The dedicated [Full Stage 3 evaluation workflow](.github/workflows/full-evaluation.yml)
uses read-only contents/actions permissions, restores the original models, regenerates
and checks data identity, evaluates all six models, computes Grad-CAM for all 2,880
predictions, independently verifies results, and uploads `stage3-scientific-results`
for 90 days. A narrowly scoped push trigger on the Stage 3 feature branch supports
its first execution before default-branch manual dispatch is available. After merge,
manual Actions execution is supported. Failures preserve logs and incomplete status.

The held-out distribution is 120 samples in each of four groups per seed. WGA is the
minimum group accuracy. Confusion matrices use actual rows and predicted columns,
class order `[non_cow,cow]`. Flip rate counts genuine foreground pairs whose predicted
class changes across backgrounds. Read it together with **paired correctness** (both
images correct), WGA and overall accuracy: a classifier can be stable and wrong.
Disagreements explicitly distinguish corrected errors from new treatment errors.
Cross-seed summaries use arithmetic mean and sample SD (`ddof=1`), with paired
within-seed treatment-minus-baseline differences. Only three jointly varying data
and initialization seeds are available; no significance claim is made.

Grad-CAM targets the true class, using cow score `logit` and horse score `-logit` on
the final convolutional ReLU. Baseline and treatment use identical images/targets.
Every map is saved and independently peak-normalized; zero maps remain zero.
Foreground mask shares and area-adjusted enrichment are descriptive localization
measures. Zero-mass maps have undefined shares and explicit zero-map counts.
Neither heatmap appearance nor foreground share alone proves causal independence.

Each seed exports `predictions.csv`, `counterfactual_pairs.csv`,
`model_disagreements.csv`, `metrics.json`, `attribution.csv`, `gradcam_examples.csv`,
`test_inputs.npz`, model heatmap NPZs and figures A–F. Root `summary.json`/CSV contain
per-seed results, mean/SD, paired differences and limitations. Original test RGB/mask
arrays and model-resolution inputs are included without bulk training images.
`source_references.json` identifies checkpoints, epochs, original artifact/source,
data hashes, audits and training environment. `evaluation_config.json` freezes
protocol and executed source. `evaluation_manifest.json` hashes immutable files;
`figure_sources.json` traces plots to their saved scientific inputs.

Stage 4 consumes this stable schema version 1 directly after verification. It can
show predictions, real counterfactuals, figures and saved Grad-CAM without training
or recomputing metrics. For live Grad-CAM, restore the referenced original checkpoint
and the identified input. See [methodology and limitations](docs/methodology.md).

Download the exact successful Stage 3 artifact listed in the execution report:

```bash
gh run download <verified-stage3-actions-run-id> --repo alfrzhb/cowbias-lab \
  --name stage3-scientific-results --dir artifacts/runs/downloaded-stage3
# Upload contains a run directory plus the external execution/retrieval logs.
python scripts/verify_evaluation.py \
  --run artifacts/runs/downloaded-stage3/stage3-full-<run-id>-<attempt> --require-full
```

Preserve the ZIP before its recorded expiry for longer retention. These are synthetic
horse/cow illustrations with two controlled backgrounds; performance on photographs
and other species or styles remains untested. The academic PDF is intentionally not
produced in Stage 3.
