# Reproducibility and Stage 5 research handoff

## Scope and authoritative evidence

The starting `main` was `962aab7c9887d156c1460e454b05bf307bc0d9ba`, the Stage 4 squash merge.
Stages 1-4 scientific source, model, preprocessing, selection and evaluation protocol are unchanged.
The Stage 5 branch is `feat/stage-5-final-qa-report`; it must be independently reviewed before merge.

The original DOCX is absent in the execution workspace. The report uses the exact assignment
text supplied by the user and `docs/PRD.md`; it does not claim direct inspection of the DOCX.
Earlier explanation/prototype PDFs are not sources of experimental numbers or images.

Three distinct activities must not be conflated:

1. **Original full experiment:** six Stage 2 models, followed by Stage 3 held-out evaluation.
2. **Stage 5 QA:** original-byte checks, validation-probe checkpoint reload, independent arithmetic,
   deterministic dataset replay, genuine small training/evaluation smoke, and report/UI checks.
3. **Future replication:** new full training/evaluation under a new run identity, if desired.
   No Stage 5 full retraining or full held-out inference rerun was needed or performed.

## Exact original archives and availability

Availability and GitHub metadata were checked on 19 October 2026; all three were unexpired.
The ZIP bytes downloaded through authorized GitHub access matched the following hashes and passed
ZIP CRC validation. `docs/evidence-manifest.json` includes full per-member SHA-256 inventories.

| Original archive name | Workflow | Artifact | Bytes | Expires (UTC) |
|---|---:|---:|---:|---|
| `stage2-trained-models.zip` | 38827143763 | 12015781574 | 3,631,083 | 2027-01-14 00:02:25 |
| `stage3-scientific-results.zip` | 38874819983 | 12035480745 | 18,190,613 | 2027-01-14 08:36:30 |
| `stage4-dashboard-review.zip` | 39140400975 | 12141178453 | 1,848,412 | 2027-01-16 15:25:18 |

```text
stage2-trained-models.zip
  d99ae16c94d57810e7cad11d25f62be061df943202ba4b9a59d10721012597be
  source 881be87ef2d83100b51bfb069e1fa75097eb9688
stage3-scientific-results.zip
  75b42453cde76bb6b5e7f5c1db903908b592439949a324e79f8ff17186dbb0d4
  source f4c517e64c20ed3498a656ff31dcb67bdbc5e69d
stage4-dashboard-review.zip
  78b5f33e9741808dcbcb55c8d014b00acfc84e5bf43b39ec2019d0bd6e2dd18b
  source 4fc877ae5350a0af9d70f6f6c9533e97276bd254
```

Dataset run identity: `datasets`; training run: `stage2-full`; evaluation run:
`stage3-full-38874819983-1`. Executed source commits are retained separately from squash merges.
The Stage 2 archive has 57 files; Stage 3 has 75 archive files including 70 inventoried scientific
files plus run metadata/diagnostics; Stage 4 has 33 files. Verification never modifies originals:
the historical Stage 2 checkpoint verifier runs on a disposable copy.

## Durable archival status: PENDING

Repository metadata confirms **public** visibility. There is no authorized durable access-controlled
GitHub destination or applicable release-upload capability for private research archives in this
session. Public release assets would not preserve the requested access restrictions. Repository
visibility was not changed, and no archives were sent to another service or committed to Git.

Verified original local ZIP copies and a reproducible archival script exist, but this is **not**
a claim of durable preservation. The scientific handoff does not depend on an undocumented workspace:
the repository records exact identifiers, retrieval, hashes and verification. After Actions expiry,
retrieval still requires a separately preserved byte-identical copy. Before the earliest expiry,
an authorized custodian must supply a durable access-controlled destination, transfer the original
ZIPs without recompression, round-trip download them, check these hashes, and record the destination.
No such round-trip archival destination is currently recorded. Report and PR disclose this risk.

Recovery while Actions artifacts remain available:

```bash
# Requires an already-authorized gh session or GH_TOKEN with repository/actions read.
python scripts/archive_evidence.py --download --extract \
  --directory artifacts/evidence --manifest artifacts/archive-manifest.json
```

For separately obtained originals, put the three named ZIPs in `artifacts/evidence` and omit
`--download`. `--extract` requires new extraction directories. Archive names/bytes are retained;
SHA-256, CRC, source workflow metadata, safe extraction and member hashes are checked. The script
never publishes assets. Existing ZIP/download, extraction and manifest destinations are rejected.
`*.retrieval.json` records API metadata for authenticated downloads without credentials.

## Fresh-clone procedure

The acceptance workflow clones the reviewed commit into a separate directory and creates a new
venv without system site-packages. It installs declared dependencies from package indexes; wheel
caches may accelerate installation but do not provide hidden project files. Local Stage 5 QA uses
the same process, explicitly supplying and re-verifying the three original ZIPs.

```bash
git clone https://github.com/alfrzhb/cowbias-lab.git cowbias-clean
cd cowbias-clean
git checkout feat/stage-5-final-qa-report  # For exact reproduction, use the reviewed commit SHA.
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.4 Pillow==12.3.0 PyYAML==6.0.3 matplotlib==3.10.8
python -m pip install -e '.[eval,ui,dev,browser,report]'
python -m playwright install --with-deps chromium --only-shell

# Recover three originals using the command above, or provide the exact ZIPs explicitly.
python scripts/prepare_data.py --config artifacts/evidence/stage2-trained-models/dataset_generation.yaml \
  --output artifacts/runs/data-replay
python scripts/verify_dataset_identity.py --training-run artifacts/evidence/stage2-trained-models \
  --data-run artifacts/runs/data-replay

python scripts/run_experiment.py --config config/experiment.yaml --smoke --device cpu --output artifacts/runs/fresh-smoke
python scripts/verify_checkpoints.py --run artifacts/runs/fresh-smoke
python scripts/evaluate_experiment.py --training-run artifacts/runs/fresh-smoke \
  --data-run artifacts/runs/fresh-smoke/datasets --seeds 42 --smoke --device cpu \
  --output artifacts/runs/fresh-evaluation
python scripts/verify_evaluation.py --run artifacts/runs/fresh-evaluation

python scripts/prepare_dashboard.py --archive artifacts/evidence/stage3-scientific-results.zip
python scripts/verify_dashboard.py \
  --run artifacts/runs/downloaded-stage3/stage3-full-38874819983-1
python scripts/review_dashboard.py \
  --run artifacts/runs/downloaded-stage3/stage3-full-38874819983-1 --output artifacts/review
```

The browser review starts an actual Streamlit server, waits for health/readiness, checks all five
tabs, every seed and outcome, exact paired identity under background switching, responsive
viewports, missing-artifact state and reopened original CSV/JSON/PNG exports. It shuts down the
server. For manual demonstration use the `COWBIAS_EVAL_ROOT` command in README.

## Reproduce and validate the PDF

```bash
python scripts/export_report.py --run-id stage3-full-38874819983-1 \
  --artifact-root artifacts/evidence/stage3-scientific-results \
  --training-run artifacts/evidence/stage2-trained-models --archive-dir artifacts/evidence \
  --stage4-root artifacts/evidence/stage4-dashboard-review \
  --output artifacts/report/task-3-report.pdf --manifest artifacts/report/evidence-manifest.json
python scripts/verify_report.py --pdf artifacts/report/task-3-report.pdf \
  --manifest artifacts/report/evidence-manifest.json --compare-manifest docs/evidence-manifest.json \
  --render-dir artifacts/report/pages
```

Output paths must be new and outside the evidence directories. The exporter never discovers the
newest run. It rejects an incorrect run ID, missing/corrupt archive/member, incompatible seed,
checkpoint/history/source hash, missing figure or invalid full scientific evidence before export.
Stage 4 screenshot source is optional; absent screenshots are omitted, not replaced by mockups.
All three ZIPs remain required for final evidence validation.

The report has **13 A4 pages** with the screenshot and original scientific summary figure; the
extra page keeps the dashboard readable. Without optional Stage 4 screenshot input it has 12 pages.
The document uses selectable Times-family text, numbered figures/tables and clickable evidence links.
Student identity remains unassigned. Ten tables and nine figures trace to original files or clearly
marked source-derived diagrams. Every numeric result is read or derived from verified CSV/JSON;
the six training curves come from original Stage 2 history. Case selection is disclosed and includes
both a correction and the new treatment error. Raw genuine Stage 3 maps and an original scientific
PNG are used, alongside actual Stage 4 Chromium capture.

`verify_report.py` checks declared page count, all table-cell text on the correct page, exact RGB
pixels of every embedded figure, link presence/scheme, selectable text, glyph errors and page bounds.
It cannot infer visual readability or semantic non-overlap from those checks. Review every rendered
PNG before approval. Local visual QA results and acceptance renders are recorded with the PR.
PDF byte identity is not the cross-environment contract: comparison uses extracted-text hash,
figure pixels/sources, numerical tables, pinned archive manifests and report-source hashes.

## Negative and edge-case coverage

| Failure | Evidence in tests |
|---|---|
| Invalid ratios | `test_dataset.py::test_invalid_fields` |
| Foreground split duplication / background family leakage | `test_no_leakage.py`, `test_assignment_replay.py` |
| Missing counterfactual member / empty group | `test_evaluation_metrics.py`, `test_training.py::test_missing_validation_group` |
| Duplicate or missing predictions | `test_evaluation_metrics.py`, `test_evaluator.py`, `test_dashboard.py` |
| Wrong checkpoint hash / incompatible seed | `test_checkpoint.py`, `test_report.py` |
| Missing training history | `test_report.py::test_missing_critical_training_source` |
| Corrupt image arrays / wrong NPZ ordering | `test_evaluator.py`, `test_dashboard.py::test_npz_validation` |
| Nonfinite logits/probabilities | `test_evaluation_metrics.py`, `test_cnn.py` |
| Constant predictions with zero flips | `test_evaluation_metrics.py::test_counterfactual_hand_calculation_and_stable_wrong` |
| Genuine zero Grad-CAM | `test_gradcam.py`, `test_dashboard_real.py`, actual report example |
| Missing dashboard evidence | `test_dashboard.py::test_missing_artifact`, Chromium missing-state review |
| Missing report figure | `test_report.py::test_missing_report_source_figure` |
| Wrong report run ID / output collisions | `test_report.py` |
| Archive corruption / wrong remote provenance | `test_report.py`, existing ZIP handoff tests |

Existing tests were not weakened. Full-original tests require `COWBIAS_TEST_EVIDENCE` (seven Stage 5
tests) and `COWBIAS_TEST_FULL_RUN` (seven Stage 4 tests); routine CI intentionally skips those when
archives are absent. The separate acceptance job executes them with original pinned data.

## Gates and review evidence

QG-01/02/03 are supported by three-seed original audit and deterministic source/assignment replay.
QG-04/05 use original histories, paired initialization/permutation metadata and six checkpoint
reloads. QG-06/07/08 use the independent full verifier (2,880 predictions, 1,440 model-pair rows,
2,880 maps), raw image pairs, genuine Grad-CAM source and state/target checks. QG-09 uses original
and fresh Chromium evidence. QG-10 uses generated-table/figure checks plus full-page inspection.
QG-11 uses a separate clean clone/venv and genuine smoke pipeline. QG-12 is supported by pinned
sources, explicit synthetic scope, observed failure cases and the authorship/limitations review.

The final PR records actual test counts, fresh-clone outcome, CI/acceptance links and gate verdicts.
Do not infer completion merely from this checklist. `stage5-final-review` contains report, manifest,
rendered pages, exact reviewed SHA, installation environment, automated results and browser exports;
it excludes bulk ZIPs, checkpoints, data, credentials and pytest-generated temporary fixtures.
Long-term archival remains PENDING separately from scientific/report readiness.
