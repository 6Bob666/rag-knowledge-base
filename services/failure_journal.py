"""把失败请求写入本地 JSONL，用于构建真实困难评测集。

默认关闭；需要在 .env 中设置 FAILURE_JOURNAL_ENABLED=true 才会落盘。
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from logging_config import logger


_lock = threading.Lock()


def record_failure(
    *,
    request_id: str,
    status: str,
    question: str,
    conversation_id: str | None = None,
    candidate_count: int = 0,
    context_count: int = 0,
    retry_triggered: bool = False,
    detail: str = "",
    path: str | Path | None = None,
    enabled: bool | None = None,
) -> None:
    """追加一条失败样本；不记录完整答案、API Key 或异常堆栈。"""
    if enabled is None:
        enabled = settings.failure_journal_enabled
    if not enabled or not question.strip():
        return

    target = Path(path or settings.failure_journal_path)
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "conversation_id": conversation_id,
        "status": status,
        "question": question.strip(),
        "candidate_count": candidate_count,
        "context_count": context_count,
        "retry_triggered": retry_triggered,
        "detail": detail,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _lock:
            with target.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except OSError as exc:
        logger.warning("失败样本落盘失败: %s", exc)
