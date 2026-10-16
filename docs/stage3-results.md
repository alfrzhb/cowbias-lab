# Stage 3 execution and scientific evidence

**Complete — independently verified full held-out run `stage3-full-38874819983-1`.** All values below are exported from this run, separately from Stage 2 validation-selection results. The source is `f4c517e64c20ed3498a656ff31dcb67bdbc5e69d` with a clean source tree. No full evaluation reruns, model changes or post-test tuning occurred. Earlier fixture development caught and fixed an image-layout array error before this full evaluation.

## A. Implementation

Created eight modules under `src/cowbias/evaluation/`: `data.py`, `evaluator.py`, `metrics.py`, `counterfactual.py`, `aggregation.py`, `gradcam.py`, `plots.py`, and `verification.py`. They provide an audit-gated test adapter, fixed-checkpoint inference, strict rows, four-group metrics, genuine foreground pairs, same-image comparisons, sample-SD aggregation, real gradients, complete attribution, saved-data figures, and independent recomputation.

Created `scripts/evaluate_experiment.py`, `verify_evaluation.py`, `restore_stage2.py`, `verify_dataset_identity.py`, `summarize_evaluation.py`; three targeted test modules (`test_evaluation_metrics.py`, `test_evaluator.py`, `test_gradcam.py`); the dedicated full-evaluation workflow; and methodology/results documentation. Updated README, optional evaluation dependencies/CLI entry points, and routine CI. Stage 1/2 scientific source, configuration, PRD/PLAN, six checkpoint selections and training test-access restrictions remain unchanged. No Stage 4/5 functionality or academic PDF was implemented.

## B. Dataset and checkpoint verification

The original Stage 2 ZIP SHA-256, all 57 extracted members, six weights/history/metadata/probe contracts, validation-only selections and frozen scientific source hashes passed. Stage 2 was verified on a temporary copy, preserving the input bundle. Three regenerated manifests, dataset config hashes, source catalogs and allocations exactly match original snapshots. Source and family leakage is zero. Every seed contains 480 test images, 120/group, and 240 genuine foreground pairs.

| Seed | Checkpoints | Manifest SHA-256 | Dataset config SHA-256 | Leakage | Test images | Pairs |
|---|---|---|---|---|---|---|
| 42 | 2/2 PASS | `5ba7307417a41f941ef0504ba1e19997d4201f8f94a2a97af96a3f5bb6cfa4e7` | `02b6f40a2874e670bb94043919061fede89a5dc982fa289716d0a0a85816dac9` | 0 | 480 | 240 |
| 123 | 2/2 PASS | `0be5dea5687928d07c5a120a3050aca336e2decd1022a670f432ddd6c2aeb666` | `afff4790d99494f2d0389125eae736788cb6c261de1031feb52c0049901be68e` | 0 | 480 | 240 |
| 2026 | 2/2 PASS | `8e4912880d1f5899117d753f6c9d8993466dd487b8e1aaad3e22911c8c810017` | `0597988568c4d4234684120a3cbe614e5a0229c2ccc61014c36b851a4e20091b` | 0 | 480 | 240 |

Checkpoint identities (exact original selected epochs):

| Seed | Model | Epoch | Checkpoint SHA-256 | Verification |
|---|---|---|---|---|
| 42 | baseline | 7 | `0d7de966ea89431721796da08d3152c35d1c97ed98bb4d65497d2e4752d5819f` | PASS |
| 42 | treatment | 4 | `d6c577358920d8369adee9ef00027b06d0675526cca3f5e0f11992f095eaedec` | PASS |
| 123 | baseline | 7 | `61ce02a394f43f725ff49093963023e07cbf7c81b8c4b98ec48b9e851d7acd8f` | PASS |
| 123 | treatment | 7 | `f6222ad50e86cd44654f62215e061de16da7c31f87c486d57b50bbbf5ea0e666` | PASS |
| 2026 | baseline | 1 | `8e565291eb1629f11cddb39e055a91855273963087bf2bee3c9a03b51faf46f6` | PASS |
| 2026 | treatment | 4 | `ab1fc31eb5e8462cd9c6a914e05cd46b5948a507b65b2825e6a1f75bd69481bb` | PASS |

## C. Held-out classification

Group cells give correct/120 and accuracy. Overall cells give correct/480 and accuracy. Confusion matrices are `[[TN,FP],[FN,TP]]`, actual rows and predicted columns, order `[non_cow,cow]`.

| Seed | Model | Overall | WGA | Cow/grass | Cow/beach | Horse/grass | Horse/beach | Confusion |
|---|---|---|---|---|---|---|---|---|
| 42 | baseline | 410/480 (85.42%) | 47.50% | 120/120 (100.00%) | 113/120 (94.17%) | 57/120 (47.50%) | 120/120 (100.00%) | [[177, 63], [7, 233]] |
| 42 | treatment | 475/480 (98.96%) | 96.67% | 120/120 (100.00%) | 120/120 (100.00%) | 116/120 (96.67%) | 119/120 (99.17%) | [[235, 5], [0, 240]] |
| 123 | baseline | 261/480 (54.38%) | 7.50% | 120/120 (100.00%) | 9/120 (7.50%) | 12/120 (10.00%) | 120/120 (100.00%) | [[132, 108], [111, 129]] |
| 123 | treatment | 480/480 (100.00%) | 100.00% | 120/120 (100.00%) | 120/120 (100.00%) | 120/120 (100.00%) | 120/120 (100.00%) | [[240, 0], [0, 240]] |
| 2026 | baseline | 241/480 (50.21%) | 0.00% | 120/120 (100.00%) | 1/120 (0.83%) | 0/120 (0.00%) | 120/120 (100.00%) | [[120, 120], [119, 121]] |
| 2026 | treatment | 478/480 (99.58%) | 99.17% | 120/120 (100.00%) | 119/120 (99.17%) | 119/120 (99.17%) | 120/120 (100.00%) | [[239, 1], [1, 239]] |

Baseline scores 100% on both dominant groups for every seed. Contradictory combinations deteriorate: cow/beach is 94.17%, 7.50%, 0.83%; horse/grass is 47.50%, 10%, 0%. Treatment improves WGA and overall accuracy for all seeds, but seed 42 horse/beach decreases from 120/120 to 119/120.

## D. Counterfactuals and model disagreements

Each model/seed has 240 pairs. Outcome order is both correct / grass only correct / beach only correct / both incorrect.

| Seed | Model | Flips | Both correct | Four outcome counts | Mean abs ΔP | Median | Sample SD | Maximum |
|---|---|---|---|---|---|---|---|---|
| 42 | baseline | 70/240 (29.17%) | 170/240 (70.83%) | 170/7/63/0 | 0.336214 | 0.227723 | 0.333327 | 0.999362 |
| 42 | treatment | 3/240 (1.25%) | 236/240 (98.33%) | 236/0/3/1 | 0.024618 | 0.003067 | 0.085583 | 0.857973 |
| 123 | baseline | 219/240 (91.25%) | 21/240 (8.75%) | 21/111/108/0 | 0.728713 | 0.741613 | 0.174836 | 0.979630 |
| 123 | treatment | 0/240 (0.00%) | 240/240 (100.00%) | 240/0/0/0 | 0.006598 | 0.000359 | 0.032193 | 0.410100 |
| 2026 | baseline | 239/240 (99.58%) | 1/240 (0.42%) | 1/119/120/0 | 0.925261 | 0.942835 | 0.063290 | 0.982126 |
| 2026 | treatment | 2/240 (0.83%) | 238/240 (99.17%) | 238/1/1/0 | 0.039673 | 0.015284 | 0.067215 | 0.485977 |

Disagreement comparisons use exactly the same 480 images per seed:

| Seed | Disagreement | Both correct | Corrected errors | New errors | Both wrong | Net correct change |
|---|---|---|---|---|---|---|
| 42 | 13.96% | 409 | 66 | 1 | 4 | 65 |
| 123 | 45.62% | 261 | 219 | 0 | 0 | 219 |
| 2026 | 49.38% | 241 | 237 | 0 | 2 | 237 |

Across equal-size datasets, treatment corrects 522 baseline errors and introduces one new error (net +521 correct images). These pooled descriptive counts are not independent-observation uncertainty estimates. Treatment seed 42 retains one pair that is wrong on both backgrounds; lower flipping is therefore interpreted together with paired correctness and WGA. All class/background breakdowns and mean probability changes are in each `metrics.json`.

## E. Cross-seed statistics

Values are arithmetic mean ± sample SD over three complete seed runs. Accuracy/rate rows are percentages; their treatment-minus-baseline differences are **percentage points**. Probability sensitivity remains in probability units.

| Metric | Baseline mean ± SD | Treatment mean ± SD | Paired treatment − baseline mean ± SD |
|---|---|---|---|
| overall_accuracy | 63.3333 ± 19.2379 | 99.5139 ± 0.5243 | 36.1806 ± 19.6953 |
| worst_group_accuracy | 18.3333 ± 25.5359 | 98.6111 ± 1.7347 | 80.2778 ± 27.1484 |
| group_accuracy/cow/grass | 100.0000 ± 0.0000 | 100.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| group_accuracy/cow/beach | 34.1667 ± 52.0683 | 99.7222 ± 0.4811 | 65.5556 ± 51.8031 |
| group_accuracy/non_cow/grass | 19.1667 ± 25.0416 | 98.6111 ± 1.7347 | 79.4444 ± 26.6189 |
| group_accuracy/non_cow/beach | 100.0000 ± 0.0000 | 99.7222 ± 0.4811 | -0.2778 ± 0.4811 |
| background_flip_rate | 73.3333 ± 38.4757 | 0.6944 ± 0.6365 | -72.6389 ± 38.9117 |
| paired_correct_rate | 26.6667 ± 38.4757 | 99.1667 ± 0.8333 | 72.5000 ± 39.1511 |
| mean_absolute_probability_difference | 0.6634 ± 0.2999 | 0.0236 ± 0.0166 | -0.6398 ± 0.2957 |

Within-seed differences (percentage points, except probability units):

| Seed | Δ overall | Δ WGA | Δ flip | Δ paired correct | Δ mean abs probability |
|---|---|---|---|---|---|
| 42 | +13.541667 | +49.166667 | -27.916667 | +27.500000 | -0.311596 |
| 123 | +45.625000 | +92.500000 | -91.250000 | +91.250000 | -0.722115 |
| 2026 | +49.375000 | +99.166667 | -98.750000 | +98.750000 | -0.885587 |

Only three seeds were evaluated. Seeds jointly vary synthetic allocation/rendering and model initialization. The two background images share one foreground identity; no inference treats 480 rows as 480 independent animals. SD is descriptive, not a confidence interval or evidence of reliable statistical significance. No bootstrap intervals or p-values were generated.

## F. Explainability and attribution

All 2,880 heatmaps come from actual checkpoint gradients at the final convolutional ReLU (`features.7`). The true class is the shared target: cow `logit`, horse `-logit`. Model state fingerprints match before/after every map. Maps use standard ReLU, bilinear upsampling, per-map positive-peak normalization and an inferno overlay at alpha 0.50. Standard ReLU can yield an entirely zero map; these are preserved, not replaced.

Fixed examples are the first lexicographic foreground pair of each class, with both backgrounds. They were chosen by identity independently of performance; the seed 42 fixed horse/grass example is wrong for both models. Exploratory panels separately identify corrected, new-error and both-wrong cases.

| Seed | Model | Zero maps | Nonzero maps | Mean FG share | Mean BG share | Mean FG area | Mean enrichment |
|---|---|---|---|---|---|---|---|
| 42 | baseline | 202 | 278 | 27.13% | 72.87% | 15.01% | 1.815683 |
| 42 | treatment | 0 | 480 | 19.17% | 80.83% | 14.47% | 1.301634 |
| 123 | baseline | 78 | 402 | 15.66% | 84.34% | 14.69% | 1.067083 |
| 123 | treatment | 0 | 480 | 17.08% | 82.92% | 14.58% | 1.160418 |
| 2026 | baseline | 88 | 392 | 16.84% | 83.16% | 14.70% | 1.183526 |
| 2026 | treatment | 0 | 480 | 19.85% | 80.15% | 14.65% | 1.337068 |

Localization means condition on **nonzero heatmaps**. The baseline and treatment thus have different contributing image sets; direct comparisons of these conditional means are not paired causal estimates. Baseline has 368 zero maps in total, treatment zero. Foreground/background shares sum to one for positive mass.

The evidence complicates a simple “treatment attends more to the animal” account. Seed 42 mean foreground share declines (27.13% → 19.17%); seeds 123 and 2026 increase modestly. Cow-target treatment examples often emphasize the upper background; horse-target examples tend toward head/torso and nearby areas, with diffuse coarse maps. Foreground area is only about 14–15%, so raw background mass alone cannot establish shortcut use. The maps neither prove background independence nor negate the strong measured reduction in counterfactual prediction sensitivity.

Examples traced directly to saved predictions:

| Seed | Image ID | True group | Selection | Baseline class; P(cow) | Treatment class; P(cow) |
|---|---|---|---|---|---|
| 42 | s42-image-002273 | cow/beach | exploratory_treatment_only | 0; 0.446833 | 1; 0.984271 |
| 42 | s42-image-002717 | non_cow/beach | exploratory_baseline_only | 0; 0.484357 | 1; 0.510905 |
| 42 | s42-image-002542 | non_cow/grass | exploratory_both_incorrect | 1; 0.999135 | 1; 0.570842 |
| 123 | s123-image-002241 | cow/beach | exploratory_treatment_only | 0; 0.356135 | 1; 0.999589 |
| 2026 | s2026-image-002241 | cow/beach | exploratory_treatment_only | 0; 0.010093 | 1; 0.975699 |
| 2026 | s2026-image-002407 | cow/beach | exploratory_both_incorrect | 0; 0.085223 | 0; 0.494287 |

For a fixed genuine cow pair in seed 2026, `s2026-image-002258`/`002259`, baseline changes from cow on grass (P=0.9811) to horse on beach (P=0.0799), while treatment predicts cow on both (P=0.9298/0.9739). The foreground identity and mask are identical. On the seed 42 new-error image `s42-image-002717` (horse/beach), baseline P(cow)=0.4844 is correct, treatment P(cow)=0.5109 crosses the fixed threshold and is wrong. No threshold was adjusted.

All 19 exported figures were checked for readable labeling, full-range axes, consistent matrix scaling, genuine sample/pair identity and plausible overlays. Full-resolution PNGs and raw maps are retained in the artifact; chart source hashes are in `figure_sources.json`.

## G. Answers to the ten research questions

1. **Baseline deterioration:** yes. Every seed is perfect on dominant combinations but poorer on both contradictory combinations; the severity varies substantially.
2. **Held-out WGA:** increases for all three seeds, from 47.50/7.50/0% to 96.67/100/99.17%.
3. **Overall accuracy:** increases for all seeds, from 85.42/54.38/50.21% to 98.96/100/99.58%.
4. **Background flips:** decrease for every seed, from 29.17/91.25/99.58% to 1.25/0/0.83%. Probability sensitivity also falls, but remains nonzero and can be large for individual pairs.
5. **Stable and correct:** treatment paired correctness is 98.33/100/99.17%, supporting mostly correct stability, rather than simply constant predictions. One seed 42 pair remains consistently wrong.
6. **Corrections versus regressions:** 66/219/237 corrected images; 1/0/0 new errors. The exact IDs, probabilities and outcomes are preserved above and in `model_disagreements.csv`.
7. **Grad-CAM localization:** maps differ, but do not show a uniform shift to foreground. Some treatment cow maps emphasize background, and numerous baseline true-target maps are zero. This is descriptive, not causal proof.
8. **Mask attribution:** mixed. Foreground share decreases for seed 42 and increases for the other two seeds; unequal zero-map cohorts limit direct mean comparisons. It complicates, rather than uniformly confirms, a foreground-attention explanation.
9. **Across-seed consistency:** WGA, overall accuracy, flips, paired correctness and mean probability sensitivity change in the desired direction for all three seeds. Effect magnitude and localization evidence vary.
10. **Legitimate conclusion:** for these frozen CNNs and this controlled generator, balanced training backgrounds improve held-out group robustness and reduce measured background sensitivity. This does not establish universal causal reasoning, complete background independence, robustness to all rendering styles, other non-cow species, or real-photo generalization.

## H. Automated and independent verification

- Local regression/unit suite: **136 passed**; Ruff PASS. Genuine small CPU CLI evaluation and independent verification PASS.
- [Python 3.11 and 3.12 CI](https://github.com/alfrzhb/cowbias-lab/actions/runs/38874822764): **136 passed per job**, Ruff, actual training/evaluation smoke and artifact upload PASS.
- [Full controlled Actions execution](https://github.com/alfrzhb/cowbias-lab/actions/runs/38874819983): SUCCESS; **136 tests**, Ruff, exact ZIP/checkpoints, source/dataset audit, six held-out evaluations, figures, complete Grad-CAM, independent verification and upload PASS.
- Fresh downloaded Stage 3 ZIP matches GitHub digest. A separate remote-session verification independently recomputed **2,880 predictions, 1,440 model-pair evaluations and 2,880 attribution rows**, checked **70 scientific files**, and returned PASS.
- Hosted runtime exactly matches original Python 3.12.15, PyTorch 2.14.1+cpu, NumPy 2.5.4, Pillow 12.3.0 and PyYAML 6.0.3, plus Matplotlib 3.10.8; CPU, two threads, deterministic algorithms. Fresh local verification used Python 3.12.14 with the same scientific dependencies and passed source-pixel replay and all arithmetic. No universal bitwise reproduction is claimed.

## I. Durable artifact handoff

- Evaluation run: `stage3-full-38874819983-1`; started 2026-10-16T08:42:02.687474+00:00; completed 2026-10-16T08:42:31.426361+00:00.
- [Full Actions run](https://github.com/alfrzhb/cowbias-lab/actions/runs/38874819983); [download scientific artifact](https://github.com/alfrzhb/cowbias-lab/actions/runs/38874819983/artifacts/12035480745), ID `12035480745`, 18,190,613 bytes.
- Retention: **90 days**, expires **2027-01-14T08:36:30Z** (14 January 2027, 15:36:30 WIB). Archive before expiry for longer preservation.
- ZIP SHA-256: `75b42453cde76bb6b5e7f5c1db903908b592439949a324e79f8ff17186dbb0d4`.
- Executed evaluation source: `f4c517e64c20ed3498a656ff31dcb67bdbc5e69d`. Final reporting-only commits do not change this scientific source.
- [Original six-model Stage 2 artifact](https://github.com/alfrzhb/cowbias-lab/actions/runs/38827143763/artifacts/12015781574), training run `stage2-full`; original ZIP SHA-256 `d99ae16c94d57810e7cad11d25f62be061df943202ba4b9a59d10721012597be`.

```bash
gh run download 38874819983 --repo alfrzhb/cowbias-lab \
  --name stage3-scientific-results --dir artifacts/runs/downloaded-stage3
python scripts/verify_evaluation.py \
  --run artifacts/runs/downloaded-stage3/stage3-full-38874819983-1 --require-full
```

The upload includes the named evaluation folder plus retrieval/execution logs. Stage 4 reads saved predictions, pairs, disagreements, per-seed metrics, summary, original/model-resolution test arrays, raw heatmaps, figures and checkpoint references. It does not need to retrain or recompute scientific results. Live Grad-CAM can restore the exact referenced checkpoint and input. See the schema/interface and zero-map conventions in [methodology.md](methodology.md).

## J. GitHub delivery

- Branch: `feat/stage-3-evaluation-explainability`.
- Evaluation implementation commit: `f4c517e64c20ed3498a656ff31dcb67bdbc5e69d`.
- [Pull request #3](https://github.com/alfrzhb/cowbias-lab/pull/3) targets `main`. The current PR head contains subsequent reporting-only documentation. It remains **unmerged**; automatic merging is not enabled.

## K. Completion gates

| Gate | Requirement | Verdict |
|---|---|---|
| 1 | Frozen models | PASS |
| 2 | Reproducible datasets | PASS |
| 3 | True held-out inference | PASS |
| 4 | Individual predictions | PASS |
| 5 | Classification metrics | PASS |
| 6 | Counterfactual integrity | PASS |
| 7 | Counterfactual metrics | PASS |
| 8 | Model comparison | PASS |
| 9 | Cross-seed aggregation | PASS |
| 10 | Genuine Grad-CAM | PASS |
| 11 | Attribution analysis | PASS |
| 12 | Scientific figures | PASS |
| 13 | Independent verification | PASS |
| 14 | Reproducibility | PASS |
| 15 | Durable artifact handoff | PASS |
| 16 | QA | PASS |
| 17 | Methodology | PASS |
| 18 | GitHub delivery | PASS |

**Ready as the scientific input for Stage 4 — Streamlit Interactive Demonstration Dashboard.** The Stage 3 PR requires review and has not been merged. The findings support robustness improvement within this synthetic experiment, while the error, residual sensitivity, mixed attribution and external-validity limitations remain part of the handoff.
