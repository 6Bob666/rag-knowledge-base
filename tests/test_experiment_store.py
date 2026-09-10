from pathlib import Path

from services.experiment_store import (
    format_comparison,
    get_strategy_summary,
    load_experiments,
    make_experiment_id,
    save_experiment,
)


def test_make_experiment_id_sanitizes_name():
    assert make_experiment_id("exp reranker 001") == "exp-reranker-001"
    assert make_experiment_id() != make_experiment_id()


def test_save_and_load_experiment(tmp_path: Path):
    report = {
        "metadata": {"top_k": 3},
        "summary": {
            "reranked": {
                "recall_at_k": 1.0,
                "mrr": 1.0,
            }
        },
    }

    path = save_experiment(
        report,
        output_dir=tmp_path,
        experiment_id="exp-reranker-001",
    )
    loaded = load_experiments(tmp_path)

    assert path.name == "exp-reranker-001.json"
    assert len(loaded) == 1
    assert loaded[0]["experiment_id"] == "exp-reranker-001"
    assert loaded[0]["summary"]["reranked"]["mrr"] == 1.0


def test_get_strategy_summary_falls_back_to_empty():
    assert get_strategy_summary({}, "baseline") == {}


def test_format_comparison_outputs_header_and_rows():
    reports = [
        {
            "experiment_id": "exp-a",
            "metadata": {"top_k": 3, "reranker_score_threshold": 0.5},
            "summary": {
                "reranked": {
                    "recall_at_k": 1.0,
                    "precision_at_k": 0.67,
                    "mrr": 1.0,
                    "rejection_rate": 0.0,
                    "avg_total_ms": 300.0,
                    "max_total_ms": 400.0,
                }
            },
        },
        {
            "experiment_id": "exp-b",
            "metadata": {"top_k": 3, "reranker_score_threshold": 0.0},
            "summary": {
                "reranked": {
                    "recall_at_k": 1.0,
                    "precision_at_k": 0.52,
                    "mrr": 0.94,
                    "rejection_rate": 0.0,
                    "avg_total_ms": 100.0,
                    "max_total_ms": 150.0,
                }
            },
        },
    ]

    table = format_comparison(reports)

    assert table.startswith(
        "experiment_id | top_k | threshold | recall_at_k"
    )
    assert "exp-a | 3 | 0.5 | 1.0 | 0.67 | 1.0 | 0.0 | 300.0 | 400.0" in table
    assert "exp-b | 3 | 0.0 | 1.0 | 0.52 | 0.94 | 0.0 | 100.0 | 150.0" in table
