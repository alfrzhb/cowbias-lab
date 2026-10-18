"""Five lazy sections. Scientific evidence stays read-only and Torch-free at runtime."""

import logging
import os
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from cowbias.dashboard.data import (
    GROUPS,
    MODELS,
    discover_runs,
    fingerprint,
    image_array,
    load_arrays,
    load_run,
    load_seed,
)
from cowbias.dashboard.state import choose_valid, reset_scope
from cowbias.dashboard.visuals import attribution_text, overlay, selected_png

SECTIONS = [
    "Overview",
    "Dataset Explorer",
    "Model Comparison",
    "Evaluation",
    "Explainability & Export",
]
OUTCOMES = {
    "All": "All observed cases",
    "treatment_only": "Baseline wrong → Treatment correct",
    "baseline_only": "Baseline correct → Treatment wrong",
    "both_correct": "Both models correct",
    "both_incorrect": "Both models wrong",
}
LIMITATION = "Procedural synthetic cow/horse illustrations only. These results do not establish generalization to real-world photographs."


@st.cache_data(show_spinner=False, max_entries=8)
def cached_run(root, token, fixture):
    return load_run(root, allow_fixture=fixture)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_seed(root, token, seed, fixture):
    return load_seed(cached_run(root, token, fixture), seed)


@st.cache_data(show_spinner=False, max_entries=18)
def cached_arrays(root, token, seed, model, fixture):
    return load_arrays(cached_seed(root, token, seed, fixture), model)


def arrays(data, token, model=None):
    return cached_arrays(str(data.run.root), token, data.seed, model, data.run.fixture)


def picture(rgb, caption):
    st.image(
        Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST),
        caption="SYNTHETIC · " + caption,
        width=256,
    )


def statistic(run, model, metric):
    return next(
        r for r in run.summary["statistics"] if r["model"] == model and r["metric"] == metric
    )


def metric_value(run, model, metric, seed=None):
    if seed is None:
        r = statistic(run, model, metric)
        return f"{r['mean']:.2%}" + (
            f" ± {r['sample_sd'] * 100:.2f} pp" if r["sample_sd"] is not None else ""
        )
    m = run.summary["per_seed"][str(seed)]["models"][model]
    return f"{(m[metric] if metric in m else m['counterfactual'][metric]):.2%}"


def metric_cards(run, seed=None):
    keys = [
        ("Worst-group accuracy", "worst_group_accuracy"),
        ("Overall accuracy", "overall_accuracy"),
        ("Background flip rate", "background_flip_rate"),
        ("Paired correctness", "paired_correct_rate"),
    ]
    for offset in (0, 2):
        for col, (label, metric) in zip(st.columns(2), keys[offset : offset + 2], strict=True):
            with col, st.container(border=True):
                st.markdown(f"**{label}**")
                for model in MODELS:
                    st.write(f"{model.title()}: **{metric_value(run, model, metric, seed)}**")


def distribution(data):
    frame = pd.DataFrame(data.distributions())
    wide = frame.pivot(index=["split", "variant"], columns="group", values="count")
    for (split, variant), row in wide.iterrows():
        st.markdown(f"**{split.title()} · {variant} · {int(row.sum())} images**")
        st.table(
            pd.DataFrame(
                {
                    "Group": [g.replace("non_cow", "Horse") for g in GROUPS],
                    "Images": [int(row[g]) for g in GROUPS],
                }
            ).set_index("Group")
        )


def controls(data):
    with st.expander("Filter observed cases"):
        st.selectbox("Foreground class", ["All", "cow", "non_cow"], key="foreground")
        st.selectbox(
            "Observed outcome",
            list(OUTCOMES),
            key="outcome",
            format_func=OUTCOMES.get,
            help="A pair matches when either background has this outcome.",
        )
        st.selectbox("Counterfactual flip filter", ["All", *MODELS], key="flips")
    options = data.pair_options(
        st.session_state.foreground, st.session_state.outcome, st.session_state.flips
    )
    if not options:
        st.info("No observed pairs match this category for the active seed. Choose another filter.")
        return None
    choose_valid(st.session_state, "pair", options)
    pair = st.selectbox("Foreground pair", options, key="pair")
    selection = (data.seed, pair, st.session_state.outcome)
    if st.session_state.get("case_selection") != selection:
        if st.session_state.outcome != "All":
            st.session_state.background = next(
                bg
                for bg, image in data.members[pair].items()
                if data.disagreements[image]["outcome"] == st.session_state.outcome
            )
        st.session_state.case_selection = selection
    bg = st.radio(
        "Background", ["grass", "beach"], key="background", horizontal=True, format_func=str.title
    )
    image = data.image_id(pair, bg)
    st.caption(f"Selection · seed={data.seed} · pair={pair} · background={bg} · image={image}")
    st.caption(f"Foreground identity: {data.manifest[image]['foreground_id']}")
    return image


def overview(data, token):
    st.header("Can a CNN recognize the animal when the background changes?")
    st.write(
        "**Research question:** Does balancing training background distributions reduce spurious background dependence in a CNN image classifier?"
    )
    st.write(
        "Cow versus synthetic horse; grass versus beach. Four object/background groups expose potential background shortcuts on a balanced held-out test set."
    )
    for col, title, text in zip(
        st.columns(2),
        ["Baseline · correlated training", "Treatment · balanced training"],
        [
            "97:3 per class. Cow → grass; horse → beach are dominant associations.",
            "50:50 background assignment for each class. Treatment happens during dataset preparation and training, not as a postprocessing image filter.",
        ],
        strict=True,
    ):
        with col, st.container(border=True):
            st.subheader(title)
            st.write(text)
    with st.expander("Training and evaluation counts from manifests"):
        distribution(data)
    st.subheader("One architecture, a controlled comparison")
    st.write(
        "Three convolutional blocks; RGB 32×32 input; identical initialization within each seed, optimizer and training budget. Six frozen checkpoints in the full experiment. The UI reads saved Stage 3 outputs."
    )
    st.markdown(
        "**Dataset generation → Baseline/Treatment training → Held-out evaluation → Counterfactual analysis → Grad-CAM**"
    )
    st.subheader("Held-out results across seeds")
    st.caption("Mean ± sample SD; rate SD is in percentage points (pp), not a confidence interval.")
    metric_cards(data.run)
    st.info(
        "A 3–5 minute review: inspect a genuine pair, compare predictions across backgrounds, then read Evaluation and Explainability."
    )
    st.warning(LIMITATION)


def dataset_explorer(data, token):
    st.header("Inspect the controlled test set")
    st.write(
        "Each pair fixes foreground identity, pose and mask, and changes the background. Stage 1 isolates source identities across train/validation/test before compositing; the archived audit verifies this."
    )
    with st.expander("Group counts from the verified manifest"):
        distribution(data)
    st.info(
        "Displayed pixels are held-out test images. Training/validation counts come from manifests; their image pixels are not included in this bundle."
    )
    image = controls(data)
    if image is None:
        return
    inputs = arrays(data, token)
    for col, bg in zip(st.columns(2), ("grass", "beach"), strict=True):
        i = data.image_id(st.session_state.pair, bg)
        with col:
            picture(
                image_array(inputs, i),
                f"{data.manifest[i]['object_label']} / {bg} · original RGB · {i}",
            )
    with st.expander("Active image metadata", expanded=True):
        st.json(data.manifest[image])


def model_comparison(data, token):
    st.header("Change the background. Keep the animal.")
    st.write(
        "Both frozen CNNs below saw this exact held-out image. Grass ↔ Beach preserves the selected pair and foreground identity."
    )
    image = controls(data)
    if image is None:
        return
    source = data.manifest[image]
    st.subheader("True label: " + ("Cow" if source["object_label"] == "cow" else "Horse (non-cow)"))
    inputs = arrays(data, token)
    picture(image_array(inputs, image), f"Active input · {source['background_label']} · {image}")
    for col, model in zip(st.columns(2), MODELS, strict=True):
        p = data.prediction(image, model)
        with col, st.container(border=True):
            st.subheader(f"{model.title()} CNN")
            st.write(f"**Prediction: {'Cow' if p['predicted'] else 'Horse (non-cow)'}**")
            st.metric("Probability of cow", f"{p['prob_cow']:.4%}")
            (st.success if p["correct"] else st.error)("CORRECT" if p["correct"] else "INCORRECT")
            st.caption(f"Checkpoint: {p['checkpoint_sha256'][:12]}…")
            with st.expander("Prediction provenance"):
                st.json(p)
    st.subheader("Counterfactual summary")
    for model in MODELS:
        p = data.pairs[model, source["pair_id"]]
        with st.container(border=True):
            st.markdown(f"**{model.title()} · {'FLIP' if p['flip'] else 'STABLE'}**")
            st.write(
                f"Grass: {'Cow' if int(p['grass_predicted']) else 'Horse'} · Beach: {'Cow' if int(p['beach_predicted']) else 'Horse'}"
            )
            st.write(
                f"Both images correct: {'Yes' if p['both_correct'] else 'No'} · Absolute probability difference: {p['delta_prob_cow']:.6f}"
            )
    bg = "beach" if source["background_label"] == "grass" else "grass"
    other = data.image_id(source["pair_id"], bg)
    picture(image_array(inputs, other), f"Alternative background · same foreground · {other}")
    st.caption(
        "Outcome filters are exploratory, including genuine new treatment errors. An absent category stays empty. A stable prediction can still be wrong."
    )


def evaluation(data, token):
    st.header("Held-out evidence, including errors")
    scope = st.radio(
        "Evaluation scope",
        ["Across-seed aggregate", "Active seed"],
        horizontal=True,
        key="evaluation_scope",
    )
    aggregate = scope == "Across-seed aggregate"
    metric_cards(data.run, None if aggregate else data.seed)
    st.caption(
        "WGA is the minimum group accuracy. Read flip rate with paired correctness: consistent predictions may be consistently wrong."
    )
    st.subheader("Accuracy in all four groups")
    rows = []
    for g in GROUPS:
        for m in MODELS:
            stats = statistic(data.run, m, f"group_accuracy/{g}")
            cells = [
                data.run.summary["per_seed"][str(s)]["models"][m]["groups"][g]
                for s in data.run.seeds
            ]
            value = (
                stats["mean"] if aggregate else data.metrics["models"][m]["groups"][g]["accuracy"]
            )
            n = (
                sum(c["total"] for c in cells)
                if aggregate
                else data.metrics["models"][m]["groups"][g]["total"]
            )
            rows.append(
                {
                    "Group": g.replace("non_cow", "Horse"),
                    "Model": m.title(),
                    "Accuracy": value,
                    "Images": n,
                }
            )
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Group:N", sort=None, title=None, axis=alt.Axis(labelLimit=130)),
            yOffset="Model:N",
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=["Baseline", "Treatment"], range=["#607d99", "#23867b"]),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=["Group", "Model", alt.Tooltip("Accuracy:Q", format=".2%"), "Images"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    with st.expander("Group accuracy and exact denominators"):
        for g in GROUPS:
            st.write(g)
            st.table(
                pd.DataFrame([r for r in rows if r["Group"] == g.replace("non_cow", "Horse")])
                .drop(columns="Group")
                .set_index("Model")
            )
    st.subheader("Confusion matrices")
    st.caption(
        "Actual rows, predicted columns; class order non-cow, cow. Aggregate matrices sum the saved per-seed counts. Both matrices share one color scale."
    )
    matrices = {
        m: (
            sum(
                np.array(data.run.summary["per_seed"][str(s)]["models"][m]["confusion_matrix"])
                for s in data.run.seeds
            )
            if aggregate
            else np.array(data.metrics["models"][m]["confusion_matrix"])
        )
        for m in MODELS
    }
    vmax = max(int(v.max()) for v in matrices.values())
    for col, m in zip(st.columns(2), MODELS, strict=True):
        with col:
            st.markdown(f"**{m.title()}**")
            frame = pd.DataFrame(
                [
                    {
                        "Actual": ["non-cow", "cow"][y],
                        "Predicted": ["non-cow", "cow"][x],
                        "Count": int(matrices[m][y, x]),
                    }
                    for y in range(2)
                    for x in range(2)
                ]
            )
            base = alt.Chart(frame).encode(
                x=alt.X("Predicted:N", sort=["non-cow", "cow"]),
                y=alt.Y("Actual:N", sort=["non-cow", "cow"]),
            )
            rect = base.mark_rect().encode(
                color=alt.Color(
                    "Count:Q", scale=alt.Scale(domain=[0, vmax], scheme="blues"), legend=None
                )
            )
            text = base.mark_text(size=18).encode(
                text="Count:Q",
                color=alt.condition(
                    alt.datum.Count > vmax * 0.6, alt.value("white"), alt.value("black")
                ),
            )
            st.altair_chart((rect + text).properties(height=220), width="stretch")
    st.subheader("Probability sensitivity")
    for m in MODELS:
        if aggregate:
            r = statistic(data.run, m, "mean_absolute_probability_difference")
            text = f"mean {r['mean']:.6f}; sample SD {r['sample_sd']}"
        else:
            r = data.metrics["models"][m]["counterfactual"]["probability_sensitivity"]
            text = f"mean {r['mean']:.6f}; median {r['median']:.6f}; maximum {r['maximum']:.6f}; sample SD {r['sample_sd']:.6f}"
        st.write(f"**{m.title()}:** {text}")
    st.subheader("Corrections and regressions")
    comparisons = (
        [data.run.summary["per_seed"][str(s)]["comparison"] for s in data.run.seeds]
        if aggregate
        else [data.metrics["comparison"]]
    )
    total = sum(c["total"] for c in comparisons)
    rate = sum(c["model_disagreement_rate"] * c["total"] for c in comparisons) / total
    st.write(f"Model disagreement: **{rate:.2%}** · {total} shared images")
    st.table(
        pd.DataFrame(
            {
                "Outcome": ["Errors corrected", "New treatment errors", "Net correct change"],
                "Images": [
                    sum(c[k] for c in comparisons)
                    for k in ("treatment_corrected", "new_errors", "net_correct_change")
                ],
            }
        ).set_index("Outcome")
    )
    st.subheader("Across-seed mean ± sample SD")
    st.caption(
        "Sample SD (ddof=1) is descriptive, not a confidence interval. Numeric displays retain SD even when mean ± SD crosses 0% or 100%."
    )
    metric_cards(data.run)
    with st.expander("All cross-seed statistics and paired differences"):
        st.dataframe(
            pd.DataFrame(data.run.summary["statistics"])[
                ["model", "metric", "n_seeds", "mean", "sample_sd"]
            ],
            hide_index=True,
        )
    st.warning(
        "Three seeds jointly vary synthetic data and initialization. Background pairs share foreground identity. No confirmatory statistical significance claim. "
        + LIMITATION
    )


def jump_example(data):
    value = st.session_state.example
    if value == "Current selection":
        return
    ids = [i for (m, i), a in data.attribution.items() if m == "baseline" and a["zero_mass"]]
    if value == "First baseline zero map":
        if not ids:
            return
        image = ids[0]
    else:
        image = value.split(" · ", 1)[0]
    r = data.manifest[image]
    st.session_state.update(
        foreground="All",
        outcome="All",
        flips="All",
        pair=r["pair_id"],
        background=r["background_label"],
    )


def explainability(data, token):
    st.header("Genuine Grad-CAM, with its limits")
    st.write(
        "Shared true-class targets: cow = logit; horse = −logit. Saved maps use individual positive-peak normalization [0,1], inferno colors and alpha 0.50. Intensity describes relative localization, not comparable absolute score contributions."
    )
    options = ["Current selection", "First baseline zero map"] + [
        e["image_id"] + " · " + e["selection"] for e in data.examples
    ]
    choose_valid(st.session_state, "example", options)
    st.selectbox("Attribution case", options, key="example", on_change=jump_example, args=(data,))
    if st.session_state.example == "First baseline zero map" and not any(
        a["zero_mass"] for (m, _), a in data.attribution.items() if m == "baseline"
    ):
        st.info("No baseline zero maps observed for this seed.")
    image = controls(data)
    if image is not None:
        rgb = image_array(arrays(data, token), image, "images")
        cols = st.columns(3)
        with cols[0]:
            picture(
                rgb, f"Original model input · true {data.manifest[image]['object_label']} · {image}"
            )
        for col, model in zip(cols[1:], MODELS, strict=True):
            with col:
                try:
                    heat = image_array(arrays(data, token, model), image, "heatmaps")
                    a = data.attribution[model, image]
                    picture(overlay(rgb, heat), f"{model.title()} Grad-CAM · true target")
                    st.write(f"**Zero map: {'Yes' if a['zero_mass'] else 'No'}**")
                    for label, key, percent in (
                        ("Foreground share", "foreground_attribution_share", True),
                        ("Background share", "background_attribution_share", True),
                        ("Foreground area", "foreground_area_fraction", True),
                        ("Foreground enrichment", "foreground_attribution_enrichment", False),
                    ):
                        st.write(f"{label}: " + attribution_text(a[key], percent))
                    meta = {
                        "run_id": data.run.config["run_id"],
                        "seed": data.seed,
                        "image_id": image,
                        "model": model,
                        "target_class": data.manifest[image]["object_label"],
                        "score": a["score"],
                        "attribution": a,
                        "overlay_alpha": 0.5,
                        "normalization": "positive peak; zero preserved",
                    }
                    st.download_button(
                        f"Download {model} overlay PNG",
                        selected_png(rgb, heat, meta),
                        file_name=f"{image}-{model}-true-target.png",
                        mime="image/png",
                        key=f"overlay_{model}",
                    )
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                    st.error(f"Grad-CAM unavailable for {image}/{model}: {exc}")
    st.info(
        "An all-zero Grad-CAM can result from ReLU suppression of signed attribution contributions even with nonzero gradients. It does not mean the model failed to process the image. Zero maps stay zero and their attribution shares are undefined."
    )
    totals = {
        m: sum(
            data.run.summary["per_seed"][str(s)]["attribution"][m]["zero_mass_count"]
            for s in data.run.seeds
        )
        for m in MODELS
    }
    st.caption(f"Saved zero maps: baseline {totals['baseline']}; treatment {totals['treatment']}.")
    st.warning(
        "Attribution is mixed. Treatment does not consistently move Grad-CAM localization toward foreground. Nonzero-map means may use different image sets. High foreground share does not establish background independence; Grad-CAM is not standalone causal proof."
    )
    st.subheader("Download original scientific evidence")
    prefix = f"seed_{data.seed}/"
    names = [
        prefix + n
        for n in (
            "predictions.csv",
            "counterfactual_pairs.csv",
            "model_disagreements.csv",
            "metrics.json",
            "attribution.csv",
            "gradcam_examples.csv",
        )
    ]
    names += [
        "summary.csv",
        "summary.json",
        "source_references.json",
        "evaluation_config.json",
        "evaluation_manifest.json",
        "figure_sources.json",
    ]
    choose_valid(st.session_state, "export_file", names)
    name = st.selectbox("Evidence file", names, key="export_file")
    st.download_button(
        "Download scientific file",
        data.run.export(name),
        file_name=Path(name).name,
        mime="text/csv" if name.endswith(".csv") else "application/json",
        key="scientific_download",
    )
    figures = sorted(
        n for n in data.run.files if n.endswith(".png") and n.startswith((prefix, "figures/"))
    )
    if figures:
        choose_valid(st.session_state, "figure", figures)
        name = st.selectbox("Scientific figure", figures, key="figure")
        raw = data.run.export(name)
        st.image(raw, caption="Original verified Stage 3 figure · " + name)
        st.download_button(
            "Download scientific figure PNG",
            raw,
            file_name=Path(name).name,
            mime="image/png",
            key="figure_download",
        )
    st.caption(
        "Scientific CSV/JSON/PNG exports preserve original bytes. Selected overlay PNGs embed image, seed, model, target and attribution provenance."
    )
    st.warning(LIMITATION)


def main():
    st.set_page_config(page_title="CowBias Lab", page_icon=":material/science:", layout="wide")
    st.title("CowBias Lab")
    st.markdown("**SYNTHETIC EXPERIMENT**")
    st.caption("Investigating Spurious Background Correlations in CNN Image Classification")
    roots = discover_runs()
    if not roots:
        st.info("Experiment not yet run")
        st.write(
            "Retrieve and verify the original Stage 3 artifact, then select its local evaluation directory."
        )
        st.code(
            "python scripts/prepare_dashboard.py\nCOWBIAS_EVAL_ROOT=artifacts/runs/downloaded-stage3/stage3-full-38874819983-1 streamlit run app/streamlit_app.py",
            language="bash",
        )
        return
    try:
        fixture = os.environ.get("COWBIAS_FIXTURE") == "1"
        roots.sort(key=lambda p: (not p.name.startswith("stage3-full"), str(p)))
        root = st.sidebar.selectbox(
            "Evaluation run",
            [str(p) for p in roots],
            format_func=lambda p: Path(p).name,
            key="active_run",
        )
        token = fingerprint(root)
        run = cached_run(root, token, fixture)
        if run.fixture:
            st.warning("DEVELOPMENT FIXTURE — NOT FINAL SCIENTIFIC RESULTS")
        st.sidebar.success(
            "Verification PASS · "
            + ("development fixture only" if run.fixture else "prepared scientific evidence")
        )
        choose_valid(st.session_state, "active_seed", run.seeds)
        seed = st.sidebar.selectbox("Dataset seed", run.seeds, key="active_seed")
        reset_scope(st.session_state, (root, run.config["run_id"], seed))
        for k in ("pair", "foreground", "outcome", "background", "flips", "example"):
            if k in st.session_state:
                st.session_state[k] = st.session_state[k]
        data = cached_seed(root, token, seed, fixture)
        with st.sidebar.expander("Artifact provenance"):
            st.json(
                {
                    "evaluation_run_id": run.config["run_id"],
                    "evaluation_source_commit": run.config["code_revision"],
                    "training_reference": run.references["stage2"],
                    "seed": seed,
                    "manifest_sha256": run.references["datasets"][str(seed)]["manifest_sha256"],
                    "archive_sha256": run.receipt["original_archive_sha256"],
                    "prepared_utc": run.receipt["prepared_utc"],
                    "origin": "procedural synthetic illustrations",
                }
            )
        with st.sidebar.expander("Experimental limitations"):
            st.write(LIMITATION)
            st.write(
                "Three joint data/initialization seeds; dependent foreground pairs; no significance claim; mixed descriptive Grad-CAM localization."
            )
        st.caption(f"Run {run.config['run_id']} · Active seed {seed} · Verification PASS")
        for tab, view in zip(
            st.tabs(SECTIONS, on_change="rerun", key="section"),
            (overview, dataset_explorer, model_comparison, evaluation, explainability),
            strict=True,
        ):
            if tab.open:
                with tab:
                    view(data, token)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        zipfile.BadZipFile,
        EOFError,
    ) as exc:
        logging.getLogger(__name__).exception("Dashboard artifact validation failed")
        st.error(f"Scientific artifact unavailable or invalid: {exc}")
        st.write(
            "Prepare a fresh copy with python scripts/prepare_dashboard.py. No substitute results are displayed."
        )
