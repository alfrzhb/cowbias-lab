# CowBias Lab

**CowBias Lab** is a reproducible research-oriented mini-project about detecting and mitigating *spurious correlations* in image classification.

The motivating example comes from an Artificial Intelligence assignment: a model that identifies cows against grass may fail when the same animal appears on a beach. We will compare a **baseline CNN** trained with correlated object/background pairs against a **treatment CNN** trained with background randomization.

> **Current status:** Documentation and project scaffolding only. Dataset generation, training, evaluation, Grad-CAM, and Streamlit are **not implemented yet**. Follow Stage 1 in the master plan next. No model accuracy is claimed.

## Project documents

| Document | Role |
|---|---|
| [Product Requirements Document](docs/PRD.md) | Product scope, requirements, research methodology, acceptance criteria |
| [Master Implementation Plan](docs/PLAN.md) | Single detailed implementation plan with five sequential stages |
| [Documentation index](docs/README.md) | Document map and implementation status |

For code or AI agent execution, treat `docs/PRD.md` as the requirements source of truth and `docs/PLAN.md` as the implementation sequence.

## Repository layout

```text
cowbias-lab/
├── README.md
├── pyproject.toml
├── .gitignore
├── .editorconfig
├── config/
│   └── experiment.yaml
├── src/cowbias/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── visualization/
├── scripts/
├── tests/
├── app/
├── notebooks/
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   └── README.md
└── artifacts/
```

Directories for later stages contain only minimal package or placeholder files. They do not indicate finished functionality. Generated datasets, checkpoints, and run outputs belong under `artifacts/` and are ignored by Git.

## Experiment contract (planned)

- **Classes:** `cow` and `non_cow` (horse surrogate).
- **Backgrounds:** `grass` and `beach`.
- **Baseline:** object/background combinations correlated at approximately 97% in training.
- **Treatment:** reassign backgrounds to reach 50/50 per foreground class using the same training foreground identities.
- **Evaluation:** four held-out object/background groups, worst-group accuracy, counterfactual pairs, confusion matrices, and Grad-CAM.
- **Reproducibility:** seeds 42, 123, and 2026; source-level split integrity; report actual results only.

All data in the first MVP is **synthetic**, so successful tests must not be presented as proof of real-world cow recognition.

## Next development step: Stage 1

Build and verify a deterministic synthetic dataset pipeline, split by source identities before compositing, plus manifests, leakage checks, counterfactual test pairs, and tests.

Full commands for generating data and running the app will be documented **after the relevant scripts exist**. The current `pyproject.toml` and YAML file establish a consistent starting point, not an experimental result.

## Academic context

Artificial Intelligence, Graduate Informatics, UIN Sunan Kalijaga — Task 3 (14 October 2026).

This repository is an engineering response to that assignment; the university does not require GitHub, Streamlit, or a specific CNN architecture.
