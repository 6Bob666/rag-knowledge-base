"""验证器校准脚本的统计逻辑测试（不加载模型）。"""

from evaluate_verifier import summarize


def case(*, hit: bool, sufficient: bool, coverage=None, negative=False) -> dict:
    return {
        "is_negative": negative,
        "hit": hit,
        "verdicts": {
            "rules_coverage": {
                "sufficient": sufficient,
                "coverage": coverage,
            }
        },
    }


def test_summarize_counts_confusion_matrix():
    details = [
        case(hit=True, sufficient=True, coverage=1.0),
        case(hit=True, sufficient=False, coverage=0.0),
        case(hit=False, sufficient=True, coverage=0.0),
        case(hit=False, sufficient=False, coverage=0.0),
    ]

    summary = summarize(details, "rules_coverage")

    assert summary["true_pass"] == 1
    assert summary["false_block"] == 1
    assert summary["false_pass"] == 1
    assert summary["true_block"] == 1
    assert summary["agreement_with_labels"] == 0.5
    assert summary["positive_count"] == 4


def test_summarize_separates_negative_samples():
    """负样本没有标准答案，只统计"是否被误放行"。"""
    details = [
        case(hit=False, sufficient=True, coverage=0.0, negative=True),
        case(hit=False, sufficient=False, coverage=0.0, negative=True),
    ]

    summary = summarize(details, "rules_coverage")

    assert summary["negative_count"] == 2
    assert summary["negative_pass"] == 1
    # 负样本不参与正样本的一致率计算，否则分母会被稀释。
    assert summary["positive_count"] == 0
    assert summary["agreement_with_labels"] == 0.0


def test_summarize_ignores_missing_coverage():
    details = [
        case(hit=True, sufficient=True, coverage=None),
        case(hit=True, sufficient=True, coverage=1.0),
    ]

    summary = summarize(details, "rules_coverage")

    assert summary["avg_coverage"] == 1.0


def test_summarize_reports_replan_trigger_rate():
    details = [
        case(hit=True, sufficient=True, coverage=1.0),
        case(hit=False, sufficient=False, coverage=0.0),
    ]

    summary = summarize(details, "rules_coverage")

    assert summary["replan_trigger_rate"] == 0.5
