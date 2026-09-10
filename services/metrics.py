"""检索评测指标。

这里的函数只接收已经产生的 ID，不加载模型，也不访问数据库，便于单元测试。
"""


def calculate_recall_at_k(
    returned_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> int:
    """判断前 K 个结果中是否至少包含一个相关文档。

    返回 1 表示命中，返回 0 表示未命中。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if not relevant_ids:
        raise ValueError("relevant_ids 不能为空")

    top_k_ids = returned_ids[:k]
    return int(bool(set(top_k_ids) & set(relevant_ids)))


def calculate_reciprocal_rank(
    returned_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算前 K 个结果中第一个相关文档的倒数排名。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if not relevant_ids:
        raise ValueError("relevant_ids 不能为空")

    relevant_id_set = set(relevant_ids)
    for index, chunk_id in enumerate(returned_ids[:k], start=1):
        if chunk_id in relevant_id_set:
            return 1 / index
    return 0.0


def calculate_mrr(reciprocal_ranks: list[float]) -> float:
    """计算多道题的平均倒数排名。"""
    if not reciprocal_ranks:
        raise ValueError("reciprocal_ranks 不能为空")
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_precision_at_k(
    returned_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算前 K 个结果中相关文档所占的比例。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if not relevant_ids:
        raise ValueError("relevant_ids 不能为空")

    top_k_ids = returned_ids[:k]
    if not top_k_ids:
        return 0.0
    relevant_id_set = set(relevant_ids)
    relevant_count = sum(chunk_id in relevant_id_set for chunk_id in top_k_ids)
    return relevant_count / len(top_k_ids)
