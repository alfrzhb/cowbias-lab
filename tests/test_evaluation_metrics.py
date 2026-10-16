"""Independent hand-calculated scientific fixtures, before held-out execution."""

import math
from copy import deepcopy

import pytest

from cowbias.evaluation.aggregation import aggregate
from cowbias.evaluation.counterfactual import compare_models, pairs_from_predictions
from cowbias.evaluation.metrics import classification, validate_predictions


def fixture_rows(model="baseline", constant=False):
    rows = []
    for y, label in [(1, "cow"), (0, "non_cow")]:
        for bg in ("grass", "beach"):
            logit = 1.0 if constant or bg == "grass" else -1.0
            p = int(logit >= 0)
            rows.append(
                {
                    "run_id": "fixture",
                    "seed": 42,
                    "model": model,
                    "image_id": f"{label}-{bg}",
                    "pair_id": label,
                    "foreground_id": label,
                    "y_true": y,
                    "object_label": label,
                    "background_label": bg,
                    "predicted": p,
                    "prob_cow": 1 / (1 + math.exp(-logit)),
                    "logit": logit,
                    "correct": int(p == y),
                    "checkpoint_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                }
            )
    return rows


def test_known_accuracy_wga_matrix():
    m = classification(fixture_rows())
    assert m["overall_accuracy"] == 0.5
    assert m["worst_group_accuracy"] == 0
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert [g["correct"] for g in m["groups"].values()] == [1, 0, 0, 1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("y_true", 2),
        ("predicted", -1),
        ("correct", 3),
        ("prob_cow", float("nan")),
        ("prob_cow", 1.1),
        ("logit", float("inf")),
        ("background_label", "snow"),
        ("model", "unknown"),
        ("foreground_id", ""),
        ("checkpoint_sha256", "bad"),
        ("object_label", "cat"),
    ],
)
def test_invalid_prediction_schema(field, value):
    rows = fixture_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_predictions(rows)


def test_threshold_zero_is_cow():
    r = fixture_rows()[0]
    r.update(logit=0.0, prob_cow=0.5, predicted=1, correct=1)
    assert validate_predictions([r])[0]["predicted"] == 1
    r["predicted"] = 0
    with pytest.raises(ValueError, match="threshold"):
        validate_predictions([r])


def test_missing_duplicate_and_wrong_correctness():
    with pytest.raises(ValueError, match="Missing group"):
        classification(fixture_rows()[:-1])
    rows = fixture_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(rows + rows[:1])
    rows[0]["correct"] = 0
    with pytest.raises(ValueError, match="correctness"):
        validate_predictions(rows)


def test_counterfactual_hand_calculation_and_stable_wrong():
    pairs, m = pairs_from_predictions(fixture_rows())
    assert len(pairs) == 2 and m["background_flip_rate"] == 1
    assert m["paired_correct_rate"] == 0
    assert m["outcomes"] == {
        "both_correct": 0,
        "grass_correct_only": 1,
        "beach_correct_only": 1,
        "both_incorrect": 0,
    }
    _, stable = pairs_from_predictions(fixture_rows(constant=True))
    assert stable["background_flip_rate"] == 0
    assert stable["paired_correct_rate"] == 0.5
    assert stable["outcomes"]["both_incorrect"] == 1


@pytest.mark.parametrize("mutation", ["missing", "foreground", "class", "background", "duplicate"])
def test_invalid_pairs(mutation):
    rows = fixture_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "foreground":
        rows[1]["foreground_id"] = "other"
    elif mutation == "class":
        rows[1].update(y_true=0, object_label="non_cow", correct=1)
    elif mutation == "background":
        rows[1]["background_label"] = "grass"
    else:
        rows.append(rows[0].copy())
    with pytest.raises(ValueError):
        pairs_from_predictions(rows)


def test_disagreement_outcomes_and_missing_ids():
    rows = fixture_rows() + fixture_rows("treatment", constant=True)
    records, m = compare_models(rows)
    assert len(records) == 4 and m["model_disagreement_rate"] == 0.5
    assert m["new_errors"] == m["treatment_corrected"] == 1
    assert m["net_correct_change"] == 0
    with pytest.raises(ValueError, match="IDs"):
        compare_models(rows[:-1])


def test_sample_sd_and_paired_aggregation():
    base = classification(fixture_rows())
    _, cf = pairs_from_predictions(fixture_rows())
    base["counterfactual"] = cf
    data = {
        str(s): {"models": {"baseline": deepcopy(base), "treatment": deepcopy(base)}}
        for s in (42, 123, 2026)
    }
    for s, v in zip((42, 123, 2026), (0.0, 0.5, 1.0), strict=True):
        data[str(s)]["models"]["treatment"]["overall_accuracy"] = v
    summary, _ = aggregate(data)
    r = next(
        r
        for r in summary["statistics"]
        if r["metric"] == "overall_accuracy" and r["model"] == "treatment"
    )
    assert r["mean"] == 0.5 and r["sample_sd"] == 0.5
    d = next(
        r
        for r in summary["statistics"]
        if r["metric"] == "overall_accuracy" and r["model"] == "treatment_minus_baseline"
    )
    assert d["mean"] == 0 and d["sample_sd"] == 0.5
    assert aggregate(data) == aggregate(data)
