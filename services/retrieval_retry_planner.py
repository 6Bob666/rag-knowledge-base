"""为多轮/困难评测规划有界检索查询。"""


def plan_attempt_queries(
    question: str,
    rewritten_query: str | None = None,
) -> list[str]:
    """返回最多两条不同的检索查询。

    第一条优先使用人工标注的 rewritten_query；
    如果改写结果与原问题不同，则把原问题作为第二条补救查询。
    """
    original = question.strip()
    primary = (rewritten_query or original).strip()
    if not primary:
        primary = original

    attempts = [primary]
    if primary != original:
        attempts.append(original)
    return attempts
