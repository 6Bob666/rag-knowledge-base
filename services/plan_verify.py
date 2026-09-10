"""Planner / Verifier：把检索循环升级成显式的"规划 → 执行 → 验证"状态机。

升级前的检索是固定脚本：先搜改写后的问题，搜不到再搜原问题，还搜不到就拒答。
脚本的问题在于"失败原因"没有被表达出来——空结果和"搜到的文档跑题"被当成同一件事。

拆成两个可替换的角色后：

    Verifier  判断当前检索结果够不够支撑回答，并说明缺什么
    Planner   按验证结论决定下一步：retrieve(query) / generate / refuse

完整状态机：

    ROUTE → RETRIEVE → VERIFY → REPLAN → RETRIEVE → ... → GENERATE → VALIDATE

三条硬约束：
1. 循环次数只由 AgentState.max_attempts 决定，Verifier 和 Planner 都不能
   自己发起检索；
2. Planner 只能返回白名单里的动作，非法动作一律降级为拒答；
3. 结论全部结构化，方便写进日志和评测，而不是只留一句自然语言。
"""

import re
from dataclasses import dataclass, field
from typing import Protocol


VERDICT_SUFFICIENT = "sufficient"
VERDICT_EMPTY = "empty"
VERDICT_LOW_SCORE = "low_score"
VERDICT_LOW_COVERAGE = "low_coverage"

ACTION_RETRIEVE = "retrieve"
ACTION_GENERATE = "generate"
ACTION_REFUSE = "refuse"

ALLOWED_PLAN_ACTIONS = (ACTION_RETRIEVE, ACTION_GENERATE, ACTION_REFUSE)

# 关键术语抽取时忽略的虚词字符；只用于判断"检索结果有没有覆盖问题"，
# 不参与任何检索打分，所以宁可宽松也不要误伤。
_STOPWORD_CHARS = set(
    "的得地是在和与及有吗呢吧啊呀什么哪些怎如何为请帮我你他她它们"
    "一下这个就都也还要会可以能把被让给对于之而且但或里外时候"
)

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class VerificationResult:
    """Verifier 的结构化结论。"""

    verdict: str
    sufficient: bool
    reason: str
    candidate_count: int = 0
    context_count: int = 0
    coverage: float | None = None
    missing_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "sufficient": self.sufficient,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "context_count": self.context_count,
            "coverage": self.coverage,
            "missing_terms": list(self.missing_terms),
        }


@dataclass(frozen=True)
class PlanStep:
    """Planner 的下一步动作。"""

    action: str
    query: str | None = None
    reason: str = ""
    degraded: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "query": self.query,
            "reason": self.reason,
            "degraded": self.degraded,
        }


@dataclass
class QueryStrategies:
    """可选的补救检索问法；Planner 按失败类型挑一个，而不是固定顺序。"""

    original: str = ""
    keyword: str = ""

    def by_name(self, name: str) -> str:
        if name == "original":
            return self.original
        if name == "keyword":
            return self.keyword
        return ""


class Verifier(Protocol):
    """Verifier 接口：任何实现都可以通过依赖注入替换。"""

    def verify(
        self,
        *,
        question: str,
        candidates: list[dict],
        contexts: list[dict],
    ) -> VerificationResult: ...


def _cjk_terms(run: str) -> list[str]:
    """中文按二字组合切分，并丢掉含虚词的字组。"""
    terms = []
    for index in range(len(run) - 1):
        bigram = run[index : index + 2]
        if any(char in _STOPWORD_CHARS for char in bigram):
            continue
        terms.append(bigram)
    return terms


def extract_key_terms(text: str, *, max_terms: int = 12) -> list[str]:
    """抽取用于覆盖度检查的关键术语（英文词 + 中文字组）。"""
    source = text or ""
    terms: list[str] = []

    for token in _ASCII_TOKEN_RE.findall(source):
        if len(token) >= 2:
            terms.append(token.lower())

    for run in _CJK_RUN_RE.findall(source):
        terms.extend(_cjk_terms(run))

    unique: list[str] = []
    for term in terms:
        if term not in unique:
            unique.append(term)
    return unique[:max_terms]


def build_keyword_query(question: str, *, max_terms: int = 6) -> str:
    """把问题压缩成"关键词查询"，用于第一次检索跑题时的补救。

    改写后的问题可能更口语、更长；关键词查询反过来更短、更接近 BM25 的
    口味，所以它和"用原问题再搜一次"是两个不同的补救策略。
    """
    terms = extract_key_terms(question, max_terms=max_terms)
    return " ".join(terms)


class RuleBasedVerifier:
    """零成本、确定性的规则验证器。

    判断顺序（先看有没有东西，再看东西对不对题）：
    1. 候选为空            → empty       知识库里就没有相关文档
    2. 上下文为空          → low_score   有候选但都被相关性阈值拦掉
    3. 关键术语覆盖不足    → low_coverage 有文档，但内容跑题
    4. 其余                → sufficient
    """

    name = "rules"

    def __init__(self, *, min_coverage: float = 0.5, min_terms: int = 2):
        self.min_coverage = min_coverage
        self.min_terms = min_terms

    def verify(
        self,
        *,
        question: str,
        candidates: list[dict],
        contexts: list[dict],
    ) -> VerificationResult:
        candidate_count = len(candidates or [])
        context_count = len(contexts or [])

        if candidate_count == 0:
            return VerificationResult(
                verdict=VERDICT_EMPTY,
                sufficient=False,
                reason="检索没有返回任何候选文档",
                candidate_count=0,
                context_count=0,
            )

        if context_count == 0:
            return VerificationResult(
                verdict=VERDICT_LOW_SCORE,
                sufficient=False,
                reason="候选文档都没有达到相关性阈值",
                candidate_count=candidate_count,
                context_count=0,
            )

        terms = extract_key_terms(question)
        if len(terms) < self.min_terms:
            # 问题里的字太少（例如"它有什么应用"），覆盖度没有参考价值，
            # 这时宁可相信 Reranker，也不要凭一个字就判跑题。
            return VerificationResult(
                verdict=VERDICT_SUFFICIENT,
                sufficient=True,
                reason="问题术语过少，跳过覆盖度检查",
                candidate_count=candidate_count,
                context_count=context_count,
            )

        haystack = " ".join(str(item.get("text") or "") for item in contexts)
        haystack_lower = haystack.lower()
        missing = [term for term in terms if term not in haystack_lower]
        coverage = (len(terms) - len(missing)) / len(terms)

        if coverage >= self.min_coverage:
            return VerificationResult(
                verdict=VERDICT_SUFFICIENT,
                sufficient=True,
                reason=f"关键术语覆盖率 {coverage:.2f}",
                candidate_count=candidate_count,
                context_count=context_count,
                coverage=coverage,
            )

        return VerificationResult(
            verdict=VERDICT_LOW_COVERAGE,
            sufficient=False,
            reason=f"关键术语覆盖率仅 {coverage:.2f}，检索结果可能跑题",
            candidate_count=candidate_count,
            context_count=context_count,
            coverage=coverage,
            missing_terms=tuple(missing[:5]),
        )


class AlwaysSufficientVerifier:
    """关闭 Planner 时使用的退化验证器：有上下文就当够用。

    它等价于升级前的行为（只在空结果时重试），保留它是为了出问题时
    能一键退回，而不是被迫回滚代码。
    """

    name = "always_sufficient"

    def verify(
        self,
        *,
        question: str,
        candidates: list[dict],
        contexts: list[dict],
    ) -> VerificationResult:
        candidate_count = len(candidates or [])
        context_count = len(contexts or [])
        if candidate_count == 0:
            return VerificationResult(
                verdict=VERDICT_EMPTY,
                sufficient=False,
                reason="检索没有返回任何候选文档",
            )
        if context_count == 0:
            return VerificationResult(
                verdict=VERDICT_LOW_SCORE,
                sufficient=False,
                reason="候选文档都没有达到相关性阈值",
                candidate_count=candidate_count,
            )
        return VerificationResult(
            verdict=VERDICT_SUFFICIENT,
            sufficient=True,
            reason="未启用覆盖度检查",
            candidate_count=candidate_count,
            context_count=context_count,
        )


class RetrievalPlanner:
    """按验证结论决定下一步动作。

    策略差异体现在"换哪种问法"：
    - 空结果 / 全被阈值拦掉：说明问得太窄或改写跑偏 → 回到原问题
    - 有文档但完全没覆盖问题：说明问得太泛、搜来的是别的话题 → 换关键词查询
    - 有文档且部分覆盖：勉强够用 → 降级生成，不为了追求覆盖率而把请求拖长
    """

    name = "rules"

    def plan(
        self,
        *,
        state,
        verification: VerificationResult,
        strategies: QueryStrategies,
    ) -> PlanStep:
        if verification.sufficient:
            return PlanStep(ACTION_GENERATE, reason="verified_sufficient")

        is_empty = verification.verdict in (VERDICT_EMPTY, VERDICT_LOW_SCORE)
        # 覆盖率完全为 0 才值得再搜一次；部分覆盖说明文档确实相关，
        # 只是措辞不同，继续换问法收益低、延迟高。
        worth_retry = is_empty or verification.coverage == 0.0

        if worth_retry and state.can_retry():
            order = ("original", "keyword") if is_empty else ("keyword", "original")
            query = self._pick_query(state, strategies, order)
            if query:
                return PlanStep(
                    ACTION_RETRIEVE,
                    query=query,
                    reason=f"replan_after_{verification.verdict}",
                )

        if verification.context_count > 0:
            return PlanStep(
                ACTION_GENERATE,
                reason="best_effort_partial_coverage",
                degraded=True,
            )

        return PlanStep(ACTION_REFUSE, reason=verification.verdict)

    @staticmethod
    def _pick_query(
        state,
        strategies: QueryStrategies,
        order: tuple[str, ...],
    ) -> str | None:
        """挑一个没搜过的补救问法，避免第二次检索原封不动地空转。"""
        tried = set(state.search_queries)
        for name in order:
            candidate = strategies.by_name(name).strip()
            if candidate and candidate not in tried:
                return candidate
        return None


@dataclass
class PlanTrace:
    """一次请求的规划轨迹，供日志、评测和前端排查使用。"""

    verifications: list[VerificationResult] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)

    def record(
        self,
        verification: VerificationResult,
        step: PlanStep,
    ) -> None:
        self.verifications.append(verification)
        self.steps.append(step)

    @property
    def last_verdict(self) -> str:
        return self.verifications[-1].verdict if self.verifications else ""

    @property
    def last_action(self) -> str:
        return self.steps[-1].action if self.steps else ""

    @property
    def degraded(self) -> bool:
        return any(step.degraded for step in self.steps)

    def to_dict(self) -> dict:
        return {
            "verifications": [item.to_dict() for item in self.verifications],
            "steps": [item.to_dict() for item in self.steps],
        }
