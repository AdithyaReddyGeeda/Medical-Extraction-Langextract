from utils.eval import (
    compute_metrics,
    compute_per_class_metrics,
    exact_match,
    load_gold,
    normalize_for_match,
    partial_match,
)


def test_normalize_strips_and_lowercases():
    assert normalize_for_match("  Hello World  ") == "hello world"


def test_normalize_collapses_spaces():
    assert normalize_for_match("hello   world") == "hello world"


def test_partial_match_substring():
    assert partial_match("WBC", "WBC 12.2") is True


def test_partial_match_reverse():
    assert partial_match("WBC 12.2", "WBC") is True


def test_partial_match_no_match():
    assert partial_match("glucose", "WBC") is False


def test_partial_match_empty():
    assert partial_match("", "WBC") is False


def test_exact_match_true():
    assert exact_match("Hypertension", "hypertension") is True


def test_exact_match_false():
    assert exact_match("Hypertension", "hypertension NOS") is False


def test_compute_metrics_perfect(sample_rows, gold_rows):
    predicted = [{"class": r["class"], "text": r["text"]} for r in gold_rows]
    m = compute_metrics(predicted, gold_rows, match_fn="exact")
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_zero():
    predicted = [{"class": "diagnosis", "text": "diabetes"}]
    gold = [{"class": "diagnosis", "text": "hypertension"}]
    m = compute_metrics(predicted, gold)
    assert m["tp"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_compute_metrics_empty_pred():
    m = compute_metrics([], [{"class": "diagnosis", "text": "HTN"}])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_compute_metrics_empty_gold():
    m = compute_metrics([{"class": "diagnosis", "text": "HTN"}], [])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_per_class_keys(sample_rows, gold_rows):
    pred = [{"class": r["class"], "text": r["text"]} for r in sample_rows]
    result = compute_per_class_metrics(pred, gold_rows)
    assert "medication" in result
    assert "diagnosis" in result
    for v in result.values():
        assert "precision" in v and "recall" in v and "f1" in v


def test_per_class_perfect_for_one_class():
    pred = [{"class": "medication", "text": "Lisinopril"}]
    gold = [{"class": "medication", "text": "Lisinopril"}]
    result = compute_per_class_metrics(pred, gold, match_fn="exact")
    assert result["medication"]["f1"] == 1.0
