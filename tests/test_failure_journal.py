import json
from pathlib import Path

from services.failure_journal import record_failure


def test_record_failure_writes_jsonl_line(tmp_path: Path):
    target = tmp_path / "failure_journal.jsonl"

    record_failure(
        request_id="req-001",
        status="no_context",
        question="一个真实失败问题",
        conversation_id="conv-001",
        candidate_count=6,
        context_count=0,
        retry_triggered=True,
        path=target,
        enabled=True,
    )

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["request_id"] == "req-001"
    assert entry["status"] == "no_context"
    assert entry["question"] == "一个真实失败问题"
    assert entry["retry_triggered"] is True


def test_record_failure_is_noop_when_disabled(tmp_path: Path):
    target = tmp_path / "failure_journal.jsonl"
    record_failure(
        request_id="req-002",
        status="no_context",
        question="不会写入的问题",
        path=target,
        enabled=False,
    )
    assert not target.exists()


def test_record_failure_skips_empty_question(tmp_path: Path):
    target = tmp_path / "failure_journal.jsonl"
    record_failure(
        request_id="req-003",
        status="no_context",
        question="   ",
        path=target,
        enabled=True,
    )
    assert not target.exists()
