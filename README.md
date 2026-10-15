# CowBias Lab

**CowBias Lab** is an academic deep-learning mini-project for studying and mitigating *spurious correlations* in image classification.

The central example is a classifier that recognizes cows on grass but may fail when a cow appears on a beach. The project will compare a baseline model trained on correlated object–background pairs against a model trained with background randomization, then visualize and evaluate the differences.

## Planned implementation

- Reproducible synthetic dataset with `cow` / `non-cow` foregrounds and `grass` / `beach` backgrounds.
- Baseline and treatment CNNs using PyTorch.
- Group-wise evaluation, worst-group accuracy, and counterfactual image pairs.
- Grad-CAM explanations and a Streamlit comparison dashboard.
- Documented methodology, tests, and a final academic report.

## Project status

**Repository initialized.** Stage 1 (repository foundation and dataset generation) has **not yet been implemented**. This README establishes the initial `main` branch so the project can be developed with Codex.

## Academic context

Created in response to an Artificial Intelligence assignment asking for a designed system that implements a treatment for fake correlation, accompanied by a model visualization and an explanation of the treatment.

## Scope note

Results from synthetic images will be labeled as synthetic experiments; performance on real-world cow photographs must be validated separately.
