# CowBias Lab methodology — Stage 3

## Research question and scope

Does balancing training backgrounds reduce shortcut dependence and improve held-out worst-group accuracy when architecture, initial weights, foreground training identities, sample order, preprocessing, optimizer, and training budget are controlled? A spurious correlation is a predictive association between label and background that need not persist when the animal moves to another setting.

This experiment uses procedural illustrations. `cow=1` denotes the synthetic cow renderer; `non_cow=0` denotes the synthetic **horse** renderer, not all other species. Grass and beach are controlled background categories. Results concern this generator and these frozen models. They do not establish recognition of real photographs, unseen species, unseen drawing styles, or universal causal understanding.

Stage 3 consumes existing Stage 2 weights. It performs no retraining, fine-tuning, new checkpoint selection, threshold tuning, or distribution changes. Streamlit and the academic PDF remain later stages.

## Dataset, intervention, and fairness

Stage 1 generator version 1.1 allocates foreground/background sources and families to exclusive train/validation/test splits before rendering. A current audit checks source and family identities, file hashes, deterministic source-pixel replay, allocation, seeded record assignments, compositing, masks, pair geometry, and zero cross-split leakage.

For each seed (42, 123, 2026), baseline training has 485 cow/grass, 15 cow/beach, 15 horse/grass, and 485 horse/beach images. Treatment has 250 images in each group. Both arms share the same 1,000 foregrounds and ordering. Training background assignment is the intended intervention. Validation has 60 images per group. Test has 120 per group (480 images), representing 240 unique foregrounds, each composited once on grass and once on beach. The animal, pose, geometry, and original foreground pixels are fixed within a pair. Soft-alpha boundary pixels naturally blend with the changed background.

Regeneration uses the unmodified YAML inside the Stage 2 bundle. Manifest/config hashes, source catalog bytes, and pre-compositing allocation bytes must match archived snapshots. No difficult examples or seeds may be removed.

## Frozen architecture and checkpoints

The shared SmallCNN uses three 3×3 Conv/ReLU/MaxPool blocks (16/32/64 channels), adaptive 4×4 pooling, a 64-unit ReLU classifier, and one cow logit (89,249 parameters). Pillow bilinear resizing converts RGB 48×48 to 32×32; float32 CHW normalization is `(pixel/255 - 0.5)/0.5`, with no augmentation.

Stage 2 selected highest **validation** WGA, then highest validation overall accuracy, then earliest epoch. Selected baseline/treatment epochs are 7/4 (seed 42), 7/7 (123), and 1/4 (2026). Stage 3 retains these choices, including the earliest-epoch tie for baseline seed 2026.

Original training source: `881be87ef2d83100b51bfb069e1fa75097eb9688`. Squash-merged main: `b0a3666d1ae382161c2aba230b17c0c516f89a6e`. The source-file hashes, rather than equality of these Git SHAs, establish compatibility. The [original Stage 2 bundle](https://github.com/alfrzhb/cowbias-lab/actions/runs/38827143763/artifacts/12015781574) has SHA-256 `d99ae16c94d57810e7cad11d25f62be061df943202ba4b9a59d10721012597be` and expires 2027-01-14 00:02:25 UTC.

Full evaluation verifies original ZIP bytes, every extracted member, exactly six checkpoint paths, restricted weights-only loading, sidecar/embedded metadata, strict parameter shapes, state hashes, histories, validation-only selection and saved validation-probe logits. Stage 2's verifier runs on a temporary copy because it writes a verification report; the downloaded bundle stays unchanged.

## Predeclared held-out protocol

Before full inference, tests exercise hand-computed metrics, counterfactual edge cases, actual small-fixture checkpoint inference, Grad-CAM gradients, masks, corrupt records, and independent artifact verification. The protocol is saved before predictions are generated. Existing Stage 1 and Stage 2 tests remain required.

A separate `evaluation.data.TestDataset` only permits `test/shared_eval` after an audit. Training's `TorchManifestDataset` continues rejecting test access. Both models receive exactly the same test tensors in manifest order. Models are in evaluation mode with parameters frozen. Ordinary prediction uses inference mode without gradients. No optimizer exists in the Stage 3 engine. State fingerprints before/after inference and Grad-CAM must match selected checkpoint states.

The immutable rule is `predicted = int(logit >= 0)`. `prob_cow = sigmoid(logit)`; cow is class 1 and horse is class 0. The zero-logit boundary predicts cow. Per-image rows, logits, probabilities and correctness flags are exported **before** metric aggregation. There are 960 rows per seed, 2,880 total. Model checkpoint and manifest hashes accompany every row.

## Classification and counterfactual definitions

Overall accuracy is correct/480. Group accuracy is correct/group count; all four denominators are 120. WGA is the minimum of four group accuracies and is the primary research metric. Confusion matrices have actual rows, predicted columns, class order `[non_cow,cow]`; cells are `[[TN,FP],[FN,TP]]` and sum to 480.

Pairs come directly from the audited `pair_id`, never from arbitrary matching. Each model/seed has 240 pairs. Flip rate is pairs with different grass/beach class predictions divided by 240. Paired correctness is pairs with **both** predictions correct divided by 240. Outcomes are both correct, grass only correct, beach only correct, or both incorrect. These sum to 240. A stable but wrong classifier can have a low flip rate; stability alone is not evidence of improvement.

Probability sensitivity is `abs(P(cow|grass)-P(cow|beach))`, summarized by mean, median, sample SD, minimum and maximum. It describes sensitivity on controlled pairs, not proof of causal understanding. All 1,440 per-model pair evaluations are exported.

Model disagreements join baseline and treatment on the same seed/image identity and verify the remaining metadata. Four disjoint outcomes are both correct, baseline only, treatment only, and both incorrect. Treatment-only cases are corrected errors; baseline-only cases are new errors. Their difference is net change in correct predictions. Counts, rates, probability changes and class/background breakdowns preserve improvements and regressions separately.

## Cross-seed statistics and independence

For accuracy, WGA, each group accuracy, flip rate, paired correctness, and mean absolute probability difference, report arithmetic mean and **sample SD (`ddof=1`)** across the three seeds. Within-seed treatment-minus-baseline differences are formed first, then summarized by mean and sample SD. Every seed remains visible.

Three runs provide limited evidence about robustness. The seeds vary allocation/rendering **and initialization**, so their spread is not initialization variability alone. The 480 images share 240 foreground identities; pair members are not independent animal observations. No row-independent confidence intervals, bootstrap intervals, p-values, or significance claims are reported. If future exploratory intervals are needed, resample paired foreground clusters with both models together, preferably stratified by object class.

## Grad-CAM and foreground attribution

Grad-CAM captures the ReLU after the final convolution (confirmed as `features.7` in this architecture). For true cow, target score is `logit`; for true horse it is `-logit`. Both models always use the same image and target class. Input gradients are enabled while parameter gradients remain disabled. Spatially averaged activation gradients weight channels; their weighted sum is passed through ReLU, bilinearly upsampled with `align_corners=False`, and divided by its positive peak. An exactly zero map stays zero. Hooks are removed even on failure.

Every held-out image receives genuine Grad-CAM for both models (2,880 maps). Raw float heatmaps and gradient/state evidence are saved. Each overlay uses the same inferno scale [0,1] and alpha 0.50. Maps are **individually peak normalized**, so their colors compare relative localization, not absolute score contribution between models. Diffuse, uniform or zero maps are retained and reported.

Representative figures use the lexicographically first foreground pair per class, including grass and beach, independently of model performance. Separate exploratory panels show the first image ID in treatment-only, baseline-only and both-wrong outcome categories when present. An absent category is an observed empty category, not a reason to invent an example. All example IDs and selection labels are exported.

Masks use nearest-neighbor 48-to-32 resizing, retaining soft-alpha values in [0,1]. For nonnegative heatmap H and aligned mask M, foreground share is `sum(H*M)/sum(H)`, background share is `sum(H*(1-M))/sum(H)`, area fraction is `mean(M)`, and enrichment is foreground share/area fraction. Positive-mass foreground/background shares sum to one. Exactly zero heatmaps have **undefined shares/enrichment (JSON null/empty CSV)** and are excluded from localization means; zero-map counts and nonzero denominators are explicit. Zero foreground area also yields undefined enrichment. No epsilon is used to turn undefined localization into an apparent measurement.

Localization is descriptive and secondary. Foreground share depends on object area, map coarseness, normalization, and target-score gradients. A high share does not establish background independence; a low share does not prove shortcut use. Counterfactual predictions and correctness provide stronger direct evidence about background sensitivity in this controlled generator.

## Verification, figures, and portable handoff

`verify_evaluation.py` reads saved CSVs, independently reconstructs classification, confusion, pair, disagreement and seed-level arithmetic, and checks exported JSON/CSV against those calculations. It intentionally does not call the evaluator's metric/aggregation/attribution implementations. It also checks coverage, ordering, finite values, fixed thresholds, checksums, source provenance, saved source images, mask resizing, raw heatmaps, target conventions and model-state fingerprints. It re-renders held-out sources from archived allocation to verify saved pixels and masks without the original training dataset. Hashes detect corruption; this is not a cryptographically signed defense against a party rewriting the entire bundle.

Figures A–F derive from saved metrics/predictions and real input/heatmap arrays. `figure_sources.json` maps exported charts to hashed scientific source artifacts. All axes show the full [0,1] metric range, matrix counts are integers with common scales, and synthetic designation is visible. PNG readability and identity/pair consistency require visual review in addition to automated checks.

The v1 interface for Stage 4 is `seed_<seed>/predictions.csv`, `counterfactual_pairs.csv`, `model_disagreements.csv`, `metrics.json`, `attribution.csv`, `gradcam_examples.csv`, `test_inputs.npz`, model heatmap NPZs, figures, and root `summary.json`/`source_references.json`. The input NPZ preserves original RGB/mask arrays, model-resolution RGB arrays/masks, and ordered IDs. Stage 4 can display results and explanations without training or recomputing scientific metrics. For live Grad-CAM, restore the checkpoint identified by its SHA-256 and the saved image preprocessing contract. Verify the bundle before use; do not guess the newest run.

`evaluation_manifest.json` covers all immutable scientific files. Mutable completion/verification bookkeeping is excluded to avoid a circular hash; run status becomes complete only after independent verification passes. Failures remain marked failed with a traceback. Existing outputs are never overwritten. Protocol/config, source hashes and commit, runtime dependencies/device/determinism, timestamps, selected checkpoint epochs, dataset hashes and archive linkage are retained. CPU versions match training dependencies; validation probes check numerical compatibility, not a universal bitwise guarantee across hardware or Python patch versions.

## Threats to validity and results

Internal threats include generator artifacts, soft-alpha compositing, remaining correlations in finite randomized assignments, coarse Grad-CAM localization and dependence within foreground pairs. Fairness controls and source replay reduce these threats but cannot eliminate all shortcuts. External threats include a shared procedural renderer, two fixed landscape categories, one horse comparator and only three jointly varying seeds. High synthetic accuracy is insufficient evidence of real-world understanding.

Actual held-out results, run IDs, scientific interpretation, workflow/artifact links and completion evidence are recorded in [stage3-results.md](stage3-results.md) after execution. These results must remain distinct from Stage 2 validation selection scores. Hypothesis support is judged from WGA, group performance, paired correctness and probability sensitivity together; no metric is changed after observing the test results.
