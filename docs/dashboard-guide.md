# CowBias Lab: lecturer demonstration

Use the original prepared Stage 3 full run. The header labels every section **SYNTHETIC
EXPERIMENT**. Run and seed controls, verification, provenance and limitations are in the native
sidebar. On mobile open the sidebar to change seed, then close it to inspect the content.

| Time | Section | Demonstration |
|---|---|---|
| 0–1 min | Overview | Explain the question, 97:3 baseline versus 50:50 treatment, identical CNN architecture and training controls. The intervention changes background assignment during preparation/training. |
| 1–2 min | Dataset Explorer | Open manifest group counts. Show a real saved cow or horse foreground on grass and beach. Read the shared pair and foreground IDs. Displayed pixels are test images; training/validation counts are metadata. |
| 2–3 min | Model Comparison | Open **Filter observed cases**, choose **Baseline wrong → Treatment correct**, and read true label, prediction, probability and explicit correctness. Toggle Grass/Beach without changing foreground. With seed 42, also inspect **Baseline correct → Treatment wrong**. |
| 3–4 min | Evaluation | Compare WGA, overall accuracy and all four group accuracies. Inspect confusion matrices. Read flip rate together with paired correctness. Switch the scope between Active seed and Across-seed aggregate; sidebar seeds are 42/123/2026. |
| 4–5 min | Explainability & Export | Show the same input and true-class Grad-CAM for both CNNs. Select **First baseline zero map**. Explain mixed localization, undefined zero-mass shares, and limits. Download an original CSV/JSON/PNG. |

The three inspection sections share foreground class, outcome, flip filter, pair and background.
An outcome filter qualifies a pair when **either** background has that outcome; its matching
background opens initially. Subsequent toggles retain the pair even when correctness changes.
Empty categories remain empty. Changing run or seed resets incompatible selections.
Attribution shortcuts come from saved example/attribution IDs, not invented samples.

WGA is the minimum of four group accuracies. A model can be stable and wrong, so a low flip
rate alone is insufficient. Mean ± sample SD is descriptive across three jointly varying data
and initialization seeds, not a confidence interval. Pair members share foreground identity.

Grad-CAM uses true cow score `logit` and true horse score `-logit`. Individual positive-peak
normalization makes colors relative localization, not comparable absolute contribution. Zero
maps stay zero; the original image is shown and shares/enrichment are **undefined**. ReLU
suppression can produce a zero map even with nonzero gradients. Localization evidence is mixed
and does not consistently shift toward foreground under treatment.

Close with: **Balancing training backgrounds improves robustness to controlled background
changes in this synthetic experiment. This does not establish universal background independence
or generalization to real photographs.** The new treatment error remains part of the evidence.

See [README setup, exports and remote review](../README.md#stage-4-interactive-research-dashboard).
The private GitHub Actions review artifact contains actual browser screenshots, reopened
downloads, consistency results and logs. No permanent public deployment or Stage 5 PDF is included.
