import json
from pathlib import Path


def load_example_cases() -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "evaluation_dataset_hard.example.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_hard_dataset_has_required_fields():
    cases = load_example_cases()
    assert cases
    for case in cases:
        assert case.get("id")
        assert case.get("question")
        assert case.get("rewritten_query")
        assert "relevant_chunk_ids" in case


def test_hard_dataset_positive_cases_have_ground_truth():
    cases = load_example_cases()
    positives = [case for case in cases if case["relevant_chunk_ids"]]
    negatives = [case for case in cases if not case["relevant_chunk_ids"]]

    assert positives
    assert negatives
    for case in positives:
        assert case.get("ground_truth_answer")
        assert case.get("why_hard")
    for case in negatives:
        assert case.get("ground_truth_answer") == ""
