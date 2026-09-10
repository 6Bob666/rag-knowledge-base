"""Planner / Verifier 的纯逻辑测试（不加载模型、不发网络请求）。"""

from dataclasses import dataclass, field

from services.plan_verify import (
    ACTION_GENERATE,
    ACTION_REFUSE,
    ACTION_RETRIEVE,
    VERDICT_EMPTY,
    VERDICT_LOW_COVERAGE,
    VERDICT_LOW_SCORE,
    VERDICT_SUFFICIENT,
    AlwaysSufficientVerifier,
    PlanStep,
    PlanTrace,
    QueryStrategies,
    RetrievalPlanner,
    RuleBasedVerifier,
    build_keyword_query,
    extract_key_terms,
)


ON_TOPIC = {"text": "CNN 的全称是 Convolutional Neural Network"}
OFF_TOPIC = {"text": "今天天气不错，适合出门散步"}


@dataclass
class FakeState:
    """只带 Planner 需要的字段，避免依赖完整 AgentState。"""

    max_attempts: int = 2
    attempt: int = 1
    search_queries: list[str] = field(default_factory=list)

    def can_retry(self) -> bool:
        return self.attempt < self.max_attempts


def strategies(
    original="CNN 的全称是什么？", keyword="cnn 全称"
) -> QueryStrategies:
    return QueryStrategies(original=original, keyword=keyword)


# ---------------- 术语抽取与关键词查询 ----------------


def test_extract_key_terms_keeps_entity_and_drops_filler():
    terms = extract_key_terms("CNN 的全称是什么？")

    assert "cnn" in terms
    assert "全称" in terms
    # 虚词组成的字组不应该被当成关键术语
    assert "什么" not in terms
    assert "是什" not in terms


def test_build_keyword_query_compresses_question():
    assert build_keyword_query("CNN 的全称是什么？") == "cnn 全称"


def test_build_keyword_query_returns_empty_for_pure_filler():
    assert build_keyword_query("它有什么应用") == "应用"
    assert build_keyword_query("的是什么") == ""


# ---------------- Verifier ----------------


def test_verifier_reports_empty_when_nothing_retrieved():
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[],
        contexts=[],
    )

    assert result.verdict == VERDICT_EMPTY
    assert result.sufficient is False


def test_verifier_reports_low_score_when_all_candidates_filtered():
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[OFF_TOPIC],
        contexts=[],
    )

    assert result.verdict == VERDICT_LOW_SCORE
    assert result.candidate_count == 1
    assert result.context_count == 0


def test_verifier_reports_low_coverage_when_context_is_off_topic():
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[OFF_TOPIC],
        contexts=[OFF_TOPIC],
    )

    assert result.verdict == VERDICT_LOW_COVERAGE
    assert result.sufficient is False
    assert result.coverage == 0.0
    assert result.missing_terms


def test_verifier_accepts_partially_covered_context():
    """覆盖度只是启发式：部分命中就不该为了追求覆盖率把请求拖长。"""
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[ON_TOPIC],
        contexts=[ON_TOPIC],
    )

    assert result.verdict == VERDICT_SUFFICIENT
    assert result.sufficient is True
    assert result.coverage == 1.0


def test_verifier_skips_coverage_for_questions_with_few_terms():
    """问题里几乎没有实词时，宁可相信 Reranker。"""
    result = RuleBasedVerifier().verify(
        question="它有什么应用",
        candidates=[OFF_TOPIC],
        contexts=[OFF_TOPIC],
    )

    assert result.verdict == VERDICT_SUFFICIENT
    assert result.coverage is None


def test_verifier_threshold_is_configurable():
    verifier = RuleBasedVerifier(min_coverage=0.0)

    result = verifier.verify(
        question="CNN 的全称是什么？",
        candidates=[OFF_TOPIC],
        contexts=[OFF_TOPIC],
    )

    assert result.verdict == VERDICT_SUFFICIENT


def test_always_sufficient_verifier_only_blocks_empty_results():
    verifier = AlwaysSufficientVerifier()

    assert (
        verifier.verify(
            question="CNN 是谁发明的？",
            candidates=[OFF_TOPIC],
            contexts=[OFF_TOPIC],
        ).sufficient
        is True
    )
    assert (
        verifier.verify(
            question="CNN 是谁发明的？", candidates=[], contexts=[]
        ).sufficient
        is False
    )


# ---------------- Planner ----------------


def plan(verification, state=None, strategies_=None) -> PlanStep:
    return RetrievalPlanner().plan(
        state=state or FakeState(),
        verification=verification,
        strategies=strategies_ or strategies(),
    )


def test_planner_generates_when_verification_passes():
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[ON_TOPIC],
        contexts=[ON_TOPIC],
    )

    assert plan(result) == PlanStep(ACTION_GENERATE, reason="verified_sufficient")


def test_planner_retries_with_original_question_when_nothing_found():
    """空结果说明问得太窄或改写跑偏，回到原问题最稳。"""
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？", candidates=[], contexts=[]
    )

    step = plan(result, FakeState(attempt=1, max_attempts=2))

    assert step.action == ACTION_RETRIEVE
    assert step.query == "CNN 的全称是什么？"
    assert step.reason == "replan_after_empty"


def test_planner_retries_with_keyword_query_when_result_is_off_topic():
    """搜到了文档但完全跑题，换关键词查询而不是原样再搜一遍。"""
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[OFF_TOPIC],
        contexts=[OFF_TOPIC],
    )

    step = plan(result, FakeState(attempt=1, max_attempts=2))

    assert step.action == ACTION_RETRIEVE
    assert step.query == "cnn 全称"
    assert step.reason == "replan_after_low_coverage"


def test_planner_does_not_repeat_a_query_already_tried():
    state = FakeState(attempt=1, max_attempts=2)
    state.search_queries = ["CNN 的全称是什么？"]
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？", candidates=[], contexts=[]
    )

    step = RetrievalPlanner().plan(
        state=state,
        verification=result,
        strategies=strategies(original="CNN 的全称是什么？", keyword="cnn 全称"),
    )

    # 原问题搜过了，就换成关键词查询，而不是原封不动再搜一次。
    assert step.action == ACTION_RETRIEVE
    assert step.query == "cnn 全称"


def test_planner_refuses_when_no_attempt_left_and_nothing_found():
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？", candidates=[], contexts=[]
    )

    step = plan(result, FakeState(attempt=2, max_attempts=2))

    assert step.action == ACTION_REFUSE
    assert step.reason == VERDICT_EMPTY


def test_planner_refuses_when_candidates_were_all_filtered():
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[OFF_TOPIC],
        contexts=[],
    )

    step = plan(result, FakeState(attempt=2, max_attempts=2))

    assert step.action == ACTION_REFUSE
    assert step.reason == VERDICT_LOW_SCORE


def test_planner_degrades_to_generate_with_partial_coverage():
    """有相关文档就别硬拒答：部分覆盖时降级生成并标记 degraded。"""
    partially_covered = {"text": "CNN 是一种神经网络模型"}
    result = RuleBasedVerifier().verify(
        question="CNN 的卷积核是什么",
        candidates=[partially_covered],
        contexts=[partially_covered],
    )
    assert result.verdict == VERDICT_LOW_COVERAGE
    assert 0.0 < result.coverage < 0.5

    step = plan(result, FakeState(attempt=2, max_attempts=2))

    assert step.action == ACTION_GENERATE
    assert step.degraded is True
    assert step.reason == "best_effort_partial_coverage"


def test_planner_generates_instead_of_refusing_when_contexts_exist():
    """没有剩余重试机会但手上有上下文时，不得因为覆盖率判拒答。"""
    state = FakeState(attempt=1, max_attempts=1)
    state.search_queries = ["CNN 的全称是什么？"]
    result = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？",
        candidates=[OFF_TOPIC],
        contexts=[OFF_TOPIC],
    )

    step = RetrievalPlanner().plan(
        state=state,
        verification=result,
        strategies=strategies(),
    )

    assert step.action == ACTION_GENERATE
    assert step.degraded is True


# ---------------- 轨迹 ----------------


def test_plan_trace_records_verdicts_and_steps():
    trace = PlanTrace()
    verification = RuleBasedVerifier().verify(
        question="CNN 的全称是什么？", candidates=[], contexts=[]
    )
    step = PlanStep(ACTION_RETRIEVE, query="CNN 的全称是什么？")
    trace.record(verification, step)
    trace.record(verification, PlanStep(ACTION_GENERATE))

    assert trace.last_verdict == VERDICT_EMPTY
    assert trace.last_action == ACTION_GENERATE
    assert trace.degraded is False
    payload = trace.to_dict()
    assert payload["verifications"][0]["verdict"] == VERDICT_EMPTY
    assert payload["steps"][0]["query"] == "CNN 的全称是什么？"


def test_plan_trace_flags_degraded_run():
    trace = PlanTrace()
    trace.record(
        RuleBasedVerifier().verify(
            question="CNN 的全称是什么？", candidates=[], contexts=[]
        ),
        PlanStep(ACTION_GENERATE, reason="best_effort", degraded=True),
    )

    assert trace.degraded is True
