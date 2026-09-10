from pathlib import Path

from services.failure_labeler import (
    LABELS,
    append_label,
    load_journal,
    load_labels,
)


def test_load_journal_ignores_missing_file(tmp_path: Path):
    assert load_journal(tmp_path / "missing.jsonl") == []


def test_load_journal_parses_lines(tmp_path: Path):
    target = tmp_path / "journal.jsonl"
    target.write_text(
        '{"request_id": "a", "status": "no_documents", "question": "q1"}\n'
        '{"request_id": "b", "status": "llm_error", "question": "q2"}\n',
        encoding="utf-8",
    )
    entries = load_journal(target)
    assert len(entries) == 2
    assert entries[0]["request_id"] == "a"
    assert entries[1]["status"] == "llm_error"


def test_append_and_load_labels(tmp_path: Path):
    target = tmp_path / "labels.jsonl"
    entry = {
        "request_id": "abc123",
        "status": "no_context",
        "question": "报销流程是什么？",
    }

    append_label(entry, "retrieval", target)
    labels = load_labels(target)

    assert labels == {"abc123": "retrieval"}
    assert LABELS["retrieval"] == "检索/表达问题"


def test_append_label_rejects_unknown_label(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError):
        append_label(
            {"request_id": "x", "status": "no_documents", "question": "q"},
            "unknown",
            tmp_path / "labels.jsonl",
        )
