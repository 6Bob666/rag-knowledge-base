from services.eval_timings import summarize_timing_values, summarize_timings


def test_summarize_timing_values_on_empty_input():
    summary = summarize_timing_values([])
    assert summary == {"count": 0, "avg_ms": 0.0, "max_ms": 0.0}


def test_summarize_timing_values_computes_avg_and_max():
    summary = summarize_timing_values([10.0, 20.0, 30.0])
    assert summary["count"] == 3
    assert summary["avg_ms"] == 20.0
    assert summary["max_ms"] == 30.0


def test_summarize_timings_groups_by_stage():
    records = [
        {"retrieval_ms": 100.0, "rerank_ms": 200.0, "total_ms": 300.0},
        {"retrieval_ms": 300.0, "rerank_ms": 400.0, "total_ms": 700.0},
    ]

    summary = summarize_timings(records)

    assert summary["retrieval_ms"]["avg_ms"] == 200.0
    assert summary["rerank_ms"]["avg_ms"] == 300.0
    assert summary["total_ms"]["max_ms"] == 700.0


def test_summarize_timings_accepts_generation_stages():
    records = [
        {
            "retrieval_ms": 10.0,
            "rerank_ms": 20.0,
            "answer_llm_ms": 100.0,
            "judge_ms": 300.0,
            "total_ms": 430.0,
        },
        {
            "retrieval_ms": 30.0,
            "rerank_ms": 40.0,
            "answer_llm_ms": 200.0,
            "judge_ms": 500.0,
            "total_ms": 770.0,
        },
    ]

    summary = summarize_timings(
        records,
        keys=(
            "retrieval_ms",
            "rerank_ms",
            "answer_llm_ms",
            "judge_ms",
            "total_ms",
        ),
    )

    assert summary["answer_llm_ms"]["avg_ms"] == 150.0
    assert summary["judge_ms"]["max_ms"] == 500.0
    assert summary["total_ms"]["avg_ms"] == 600.0
