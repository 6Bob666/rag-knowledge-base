# routers/chat.py
import inspect
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from services.reranker import ReRanker
from services.llm_service import ask_llm, ask_llm_stream, rewrite_query
from services.agent_router import route_question, route_question_with_llm
from services.agent_state import AgentState, RouteDecision
from services.tool_agent import (
    KNOWLEDGE_SEARCH_TOOL_NAME,
    call_llm_tool,
    run_tool_agent,
)
from schemas import AnswerResponse, QuestionRequest
from config import settings
from logging_config import logger
from services.dependencies import (
    get_conversation_store,
    get_mcp_tool_provider,
    get_memory_service,
    get_store,
)
from services.failure_journal import record_failure
from services.output_guard import UnsafeOutputError, validate_answer
from services.cache import TTLCache
from services.query_rewrite_cache import CachedQueryRewriter
from services.resilience import FixedWindowRateLimiter, RateLimitExceeded, retry_call

if TYPE_CHECKING:
    from services.vector_store import VectorStore

router = APIRouter()
reranker = None
reranker_lock = Lock()
query_rewrite_cache = TTLCache[str](
    ttl_seconds=settings.query_rewrite_cache_ttl_seconds,
    max_entries=settings.query_rewrite_cache_max_entries,
)
cached_query_rewriter = CachedQueryRewriter(
    rewrite_query,
    cache=query_rewrite_cache,
    model_name=settings.llm_model_name,
    knowledge_base_version=settings.knowledge_base_version,
)
chat_rate_limiter = FixedWindowRateLimiter()


def _check_rate_limit(http_request: Request) -> None:
    if not settings.rate_limit_enabled:
        return
    client_host = http_request.client.host if http_request.client else "unknown"
    try:
        chat_rate_limiter.check(
            client_host,
            settings.rate_limit_requests,
            settings.rate_limit_window_seconds,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc


def new_request_id() -> str:
    """生成一次请求的短标识。"""
    return uuid.uuid4().hex[:12]


def build_rag_trace(
    request_id: str,
    conversation_id: str | None,
    status: str,
    candidate_count: int,
    context_count: int,
    timings: dict,
    extra_fields: dict | None = None,
) -> str:
    """生成可观测日志文本，避免记录完整问题和答案。

    extra_fields 用于补充 Agent 决策类字段，例如 route_action、
    attempt、search_count，方便日志分析器统计。
    """
    fields = [
        f"request_id={request_id}",
        f"conversation_id={conversation_id or '-'}",
        f"status={status}",
        f"candidate_count={candidate_count}",
        f"context_count={context_count}",
    ]
    for key, value in (extra_fields or {}).items():
        fields.append(f"{key}={value}")
    for key, value in timings.items():
        fields.append(f"{key}={value}")
    return " | ".join(fields)


def _agent_trace_fields(state: AgentState) -> dict:
    """把 Agent 状态摘要放入日志，供后续按 route_action 等维度统计。"""
    return {
        "route_action": state.route_action,
        "decision": state.decision,
        "attempt": state.attempt,
        "max_attempts": state.max_attempts,
        "search_count": len(state.search_queries),
    }


def _journal_failure(
    *,
    request_id: str,
    conversation_id: str | None,
    status: str,
    question: str,
    candidate_count: int,
    context_count: int,
    timings: dict,
    detail: str,
) -> None:
    record_failure(
        request_id=request_id,
        status=status,
        question=question,
        conversation_id=conversation_id,
        candidate_count=candidate_count,
        context_count=context_count,
        retry_triggered=bool(timings.get("retry_ms")),
        detail=detail,
    )


def get_reranker() -> ReRanker:
    """线程安全地按需加载重排序模型。"""
    global reranker
    if reranker is None:
        with reranker_lock:
            if reranker is None:
                reranker = ReRanker()
    return reranker


def get_query_rewriter() -> Callable[[str], str]:
    """提供查询改写函数，测试时可以替换为假的实现。"""
    if settings.query_rewrite_cache_enabled:
        return cached_query_rewriter
    return rewrite_query


def get_agent_router() -> Callable[[str, list[dict[str, str]]], RouteDecision]:
    """提供 Agent Router 决策函数，测试时可以替换为假的实现。"""
    if settings.agent_router_mode.lower() == "llm":
        return route_question_with_llm
    return route_question


def _call_query_rewriter(
    query_rewriter: Callable,
    question: str,
    history: list[dict[str, str]],
) -> str:
    """调用查询改写器，兼容旧版只接收 question 的测试替身。"""
    try:
        parameters = list(inspect.signature(query_rewriter).parameters.values())
        accepts_history = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            or parameter.name == "history"
            for parameter in parameters
        ) or len(
            [
                parameter
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        ) >= 2
    except (TypeError, ValueError):
        accepts_history = True

    return (
        query_rewriter(question, history)
        if accepts_history
        else query_rewriter(question)
    )


def _call_agent_router(
    agent_router: Callable,
    question: str,
    history: list[dict[str, str]],
) -> RouteDecision:
    """调用 Agent Router，兼容只接收 question 的测试替身。"""
    try:
        parameters = list(inspect.signature(agent_router).parameters.values())
        accepts_history = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            or parameter.name == "history"
            for parameter in parameters
        ) or len(
            [
                parameter
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        ) >= 2
    except (TypeError, ValueError):
        accepts_history = True

    result = (
        agent_router(question, history)
        if accepts_history
        else agent_router(question)
    )
    if isinstance(result, RouteDecision):
        return result
    if isinstance(result, dict):
        return RouteDecision(**result)
    raise TypeError(f"Agent Router 返回了不支持的类型: {type(result).__name__}")


def _call_llm(
    llm: Callable,
    question: str,
    context: str,
    history: list[dict[str, str]],
    memory: str = "",
):
    """调用生成函数，兼容旧版只接收 question、context 的测试替身。"""
    try:
        parameters = list(inspect.signature(llm).parameters.values())
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        accepts_memory = accepts_kwargs or any(
            parameter.name == "memory" for parameter in parameters
        )
        accepts_history = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            or parameter.name == "history"
            for parameter in parameters
        ) or len(
            [
                parameter
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        ) >= 3
    except (TypeError, ValueError):
        accepts_memory = False
        accepts_history = True

    # 测试替身没有 memory 参数时，把记忆并入上下文，保持兼容。
    effective_context = context
    if memory and not accepts_memory:
        effective_context = f"{memory}\n\n{context}" if context else memory

    if accepts_memory and accepts_history:
        return llm(question, effective_context, history, memory=memory)
    if accepts_memory:
        return llm(question, effective_context, memory=memory)
    if accepts_history:
        return llm(question, effective_context, history)
    return llm(question, effective_context)


def get_llm() -> Callable[[str, str], str]:
    """提供 LLM 调用函数，测试时可以替换为假的实现。"""
    return ask_llm


def get_llm_stream():
    """提供流式 LLM 调用函数，测试时可以替换为假的生成器。"""
    return ask_llm_stream


def get_tool_agent_responder() -> Callable[[list[dict]], dict]:
    """提供 Function Calling 响应函数，测试时可以替换为假的实现。"""
    return call_llm_tool


def to_context_response(record: dict) -> dict:
    metadata = record.get("metadata") or {}
    return {
        "text": record["text"],
        "source": str(metadata.get("source", "unknown")),
        "chunk_id": str(metadata.get("chunk_id", record.get("id", "unknown"))),
    }


def _mcp_tool_definitions(provider) -> list[dict] | None:
    """取 MCP 工具声明；未启用、拿不到或结果为空时返回 None。"""
    if provider is None:
        return None
    try:
        definitions = provider.function_calling_tools()
    except Exception as exc:
        logger.warning("MCP 工具列表获取失败，回退内置检索工具: %s", exc)
        return None
    return definitions or None


def _collect_mcp_contexts(raw: str, bucket: list[dict]) -> None:
    """MCP 工具返回 JSON 文本，解析后回填成接口响应里的 contexts。

    只有形如检索结果的列表才会被采纳，例如 list_knowledge_documents
    这类其它工具的输出会被忽略。
    """
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return
    if not isinstance(payload, list):
        return
    for item in payload:
        if not isinstance(item, dict) or "text" not in item:
            continue
        bucket.append(
            {
                "text": item["text"],
                "metadata": {
                    "source": item.get("source", "unknown"),
                    "chunk_id": item.get("chunk_id", "unknown"),
                },
            }
        )


def sse_event(event: str, data: dict) -> str:
    """将事件编码为 SSE 格式。"""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


async def _search_once(
    store: "VectorStore",
    reranker: ReRanker,
    query: str,
    top_k: int,
    timings: dict | None = None,
):
    """执行一轮“混合检索 + Reranker”，返回候选与最终上下文。"""
    retrieval_start = time.perf_counter()
    try:
        candidates = await run_in_threadpool(
            store.hybrid_search_records,
            query,
            top_k * 3,
        )
    except Exception as e:
        logger.exception("检索失败")
        raise HTTPException(status_code=503, detail="知识库检索暂时不可用") from e
    if timings is not None:
        timings["retrieval_ms"] = round(
            (time.perf_counter() - retrieval_start) * 1000, 1
        )

    if not candidates:
        return [], []

    rerank_start = time.perf_counter()
    try:
        contexts = await run_in_threadpool(
            reranker.rerank_records,
            query,
            candidates,
            top_k,
        )
    except Exception as e:
        logger.exception("Reranker 不可用，退回混合检索")
        contexts = candidates[:top_k]
    if timings is not None:
        timings["rerank_ms"] = round(
            (time.perf_counter() - rerank_start) * 1000, 1
        )

    return candidates, contexts


@dataclass
class _AgentPrepResult:
    """一次请求在进入生成层之前的状态快照。"""

    request_id: str
    original_question: str
    conversation_id: str | None
    history: list[dict[str, str]]
    state: AgentState
    timings: dict
    total_start: float
    route_action: str
    candidates: list[dict]
    contexts: list[dict]
    terminal_message: str | None = None
    terminal_status: str | None = None
    memory_text: str = ""


async def _prepare_agent_request(
    *,
    request: QuestionRequest,
    store: "VectorStore",
    reranker: ReRanker,
    query_rewriter: Callable[[str], str],
    agent_router: Callable,
    conversation_store,
    memory_service=None,
    log_prefix: str = "rag_answer",
) -> _AgentPrepResult:
    """执行 ROUTE -> RETRIEVE -> CHECK，普通接口和流式接口共用。

    terminal_message 非空时表示请求已在生成前终止（refuse /
    no_documents / no_context），调用方直接返回该提示即可。
    """
    request_id = new_request_id()
    timings: dict = {}
    total_start = time.perf_counter()
    original_question = request.question.strip()
    history = conversation_store.get_history(request.conversation_id)

    memory_text = ""
    if memory_service is not None and request.user_id:
        memory_start = time.perf_counter()
        try:
            memories = await run_in_threadpool(
                memory_service.recall,
                request.user_id,
                original_question,
            )
            memory_text = memory_service.format_for_prompt(memories)
        except Exception as exc:
            logger.warning("长期记忆召回失败，跳过: %s", exc)
        timings["memory_recall_ms"] = round(
            (time.perf_counter() - memory_start) * 1000, 1
        )

    state = AgentState(
        request_id=request_id,
        conversation_id=request.conversation_id,
        question=original_question,
        history=history,
        max_attempts=2 if request.enable_retry else 1,
        timings=timings,
    )

    def build_result(
        *,
        route_action: str,
        terminal_message: str | None = None,
        terminal_status: str | None = None,
    ) -> _AgentPrepResult:
        return _AgentPrepResult(
            request_id=request_id,
            original_question=original_question,
            conversation_id=request.conversation_id,
            history=history,
            state=state,
            timings=timings,
            total_start=total_start,
            route_action=route_action,
            candidates=state.candidates,
            contexts=state.contexts,
            terminal_message=terminal_message,
            terminal_status=terminal_status,
            memory_text=memory_text,
        )

    # ---------- ROUTE ----------
    state.decision = "route"
    route_start = time.perf_counter()
    try:
        decision = await run_in_threadpool(
            _call_agent_router,
            agent_router,
            original_question,
            history,
        )
    except Exception as exc:
        logger.warning("Agent Router 决策失败，安全回退为检索: %s", exc)
        decision = RouteDecision(
            action="retrieve",
            query=None,
            reason="router_error_fallback",
            confidence=0.5,
        )
    state.route_action = decision.action
    state.route_reason = decision.reason
    state.route_confidence = decision.confidence
    timings["route_ms"] = round(
        (time.perf_counter() - route_start) * 1000, 1
    )

    # ---------- REFUSE ----------
    if decision.action == "refuse":
        state.decision = "refuse"
        state.failure_status = "refused"
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        logger.info(
            "%s %s",
            log_prefix,
            build_rag_trace(
                request_id,
                request.conversation_id,
                "refused",
                0,
                0,
                timings,
                _agent_trace_fields(state),
            ),
        )
        _journal_failure(
            request_id=request_id,
            conversation_id=request.conversation_id,
            status="refused",
            question=original_question,
            candidate_count=0,
            context_count=0,
            timings=timings,
            detail="Agent Router 拒绝回答",
        )
        return build_result(
            route_action="refuse",
            terminal_message="抱歉，我暂时无法回答这个问题。",
            terminal_status="refused",
        )

    # ---------- DIRECT ANSWER（不检索，直接进入生成层） ----------
    if decision.action == "direct_answer":
        state.decision = "direct_answer"
        return build_result(route_action="direct_answer")

    # ---------- RETRIEVE：查询改写 + 有界检索循环 ----------
    state.decision = "retrieve"
    router_query = (decision.query or "").strip()
    rewrite_start = time.perf_counter()
    if router_query:
        # Router 显式给出了检索问法，跳过重复的查询改写。
        optimized_question = router_query
    else:
        try:
            optimized_question = await run_in_threadpool(
                _call_query_rewriter,
                query_rewriter,
                original_question,
                history,
            )
        except Exception as exc:
            logger.warning("查询改写失败，使用原问题: %s", exc)
            optimized_question = original_question
    optimized_question = (optimized_question or "").strip() or original_question
    timings["rewrite_ms"] = round(
        (time.perf_counter() - rewrite_start) * 1000, 1
    )

    queries_to_try = [optimized_question]
    # 第二次检索换用原始问法，而不是重复搜同一个问题。
    if request.enable_retry and optimized_question != original_question:
        queries_to_try.append(original_question)
    state.max_attempts = len(queries_to_try)

    for index, query in enumerate(queries_to_try):
        state.attempt = index + 1
        state.current_query = query
        state.search_queries.append(query)

        # 第一次检索耗时写入通用字段，方便现有日志分析器解析；
        # 第二次检索单独记录 attempt2 字段。
        attempt_timings = timings if index == 0 else {}
        attempt_start = time.perf_counter()
        candidates, contexts = await _search_once(
            store,
            reranker,
            query,
            request.top_k,
            attempt_timings,
        )
        if index == 0:
            state.candidates = list(candidates)
            state.contexts = list(contexts)
        else:
            # 第二轮结果优先覆盖第一轮，避免无关文档叠加污染生成层。
            if candidates:
                state.candidates = list(candidates)
            if contexts:
                state.contexts = list(contexts)
            timings["attempt2_retrieval_ms"] = attempt_timings.get(
                "retrieval_ms", 0.0
            )
            timings["attempt2_rerank_ms"] = attempt_timings.get("rerank_ms", 0.0)
            timings["retry_ms"] = round(
                (time.perf_counter() - attempt_start) * 1000, 1
            )
        if contexts:
            break

    candidates, contexts = state.candidates, state.contexts

    # ---------- CHECK ----------
    if not contexts:
        if not candidates:
            state.decision = "refuse"
            state.failure_status = "no_documents"
            status = "no_documents"
            detail = "未找到相关文档"
            message = "未找到相关文档。"
        else:
            state.decision = "refuse"
            state.failure_status = "no_context"
            status = "no_context"
            detail = "未找到足够相关的文档"
            message = "未找到足够相关的文档。"
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        logger.info(
            "%s %s",
            log_prefix,
            build_rag_trace(
                request_id,
                request.conversation_id,
                status,
                len(candidates),
                0,
                timings,
                _agent_trace_fields(state),
            ),
        )
        _journal_failure(
            request_id=request_id,
            conversation_id=request.conversation_id,
            status=status,
            question=original_question,
            candidate_count=len(candidates),
            context_count=0,
            timings=timings,
            detail=detail,
        )
        return build_result(
            route_action="retrieve",
            terminal_message=message,
            terminal_status=status,
        )

    state.decision = "generate"
    return build_result(route_action="retrieve")


async def _generate_answer(
    *,
    llm: Callable,
    question: str,
    context_text: str,
    history: list[dict[str, str]],
    memory_text: str = "",
    request_id: str,
    conversation_id: str | None,
    timings: dict,
    total_start: float,
    candidate_count: int,
    context_count: int,
) -> str:
    """调用 LLM 生成答案，并统一做输出安全检查。"""
    llm_start = time.perf_counter()
    def call_once():
        return _call_llm(llm, question, context_text, history, memory_text)

    try:
        answer = await asyncio.wait_for(
            run_in_threadpool(
                retry_call,
                call_once,
                max_attempts=settings.llm_max_attempts,
                backoff_seconds=settings.llm_retry_backoff_seconds,
            ),
            timeout=settings.llm_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error("LLM 调用超时")
        raise HTTPException(status_code=503, detail="问答模型响应超时") from exc
    except Exception as exc:
        timings["llm_ms"] = round(
            (time.perf_counter() - llm_start) * 1000, 1
        )
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        logger.exception("LLM 调用失败")
        _journal_failure(
            request_id=request_id,
            conversation_id=conversation_id,
            status="llm_error",
            question=question,
            candidate_count=candidate_count,
            context_count=context_count,
            timings=timings,
            detail="问答模型暂时不可用",
        )
        raise HTTPException(status_code=503, detail="问答模型暂时不可用") from exc
    timings["llm_ms"] = round(
        (time.perf_counter() - llm_start) * 1000, 1
    )

    try:
        answer = validate_answer(answer)
    except UnsafeOutputError as exc:
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        logger.warning("LLM 输出未通过安全检查: %s", exc)
        _journal_failure(
            request_id=request_id,
            conversation_id=conversation_id,
            status="invalid_output",
            question=question,
            candidate_count=candidate_count,
            context_count=context_count,
            timings=timings,
            detail="问答模型返回了无效内容",
        )
        raise HTTPException(status_code=503, detail="问答模型返回了无效内容") from exc
    return answer


async def _run_agent_answer(
    *,
    request: QuestionRequest,
    store: "VectorStore",
    reranker: ReRanker,
    query_rewriter: Callable[[str], str],
    llm: Callable[[str, str], str],
    agent_router: Callable,
    conversation_store,
    memory_service=None,
) -> dict:
    """轻量 Agentic RAG：ROUTE/RETRIEVE/CHECK + 非流式生成。"""
    prep = await _prepare_agent_request(
        request=request,
        store=store,
        reranker=reranker,
        query_rewriter=query_rewriter,
        agent_router=agent_router,
        conversation_store=conversation_store,
        memory_service=memory_service,
    )
    if prep.terminal_message is not None:
        return {"answer": prep.terminal_message, "contexts": []}

    state = prep.state
    state.decision = "generate"
    state.answer = await _generate_answer(
        llm=llm,
        question=prep.original_question,
        context_text="\n".join(item["text"] for item in prep.contexts),
        history=prep.history,
        memory_text=prep.memory_text,
        request_id=prep.request_id,
        conversation_id=prep.conversation_id,
        timings=prep.timings,
        total_start=prep.total_start,
        candidate_count=len(prep.candidates),
        context_count=len(prep.contexts),
    )
    state.decision = "success"
    prep.timings["total_ms"] = round(
        (time.perf_counter() - prep.total_start) * 1000, 1
    )
    conversation_store.append_turn(
        prep.conversation_id,
        prep.original_question,
        state.answer,
    )
    if memory_service is not None and request.user_id:
        try:
            await run_in_threadpool(
                memory_service.remember_turn,
                request.user_id,
                prep.original_question,
            )
        except Exception as exc:
            logger.warning("长期记忆写入失败: %s", exc)
    logger.info(
        "rag_answer %s",
        build_rag_trace(
            prep.request_id,
            prep.conversation_id,
            "direct_answer"
            if prep.route_action == "direct_answer"
            else "success",
            len(prep.candidates),
            len(prep.contexts),
            prep.timings,
            _agent_trace_fields(state),
        ),
    )
    return {
        "answer": state.answer,
        "contexts": [to_context_response(item) for item in prep.contexts],
    }


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    http_request: Request,
    store: "VectorStore" = Depends(get_store),
    reranker: ReRanker = Depends(get_reranker),
    query_rewriter: Callable[[str], str] = Depends(get_query_rewriter),
    llm: Callable[[str, str], str] = Depends(get_llm),
    agent_router: Callable[[str, list[dict[str, str]]], RouteDecision] = Depends(
        get_agent_router
    ),
    conversation_store=Depends(get_conversation_store),
    memory_service=Depends(get_memory_service),
):
    _check_rate_limit(http_request)
    return await _run_agent_answer(
        request=request,
        store=store,
        reranker=reranker,
        query_rewriter=query_rewriter,
        llm=llm,
        agent_router=agent_router,
        conversation_store=conversation_store,
        memory_service=memory_service,
    )


@router.post("/ask/stream")
async def ask_question_stream(
    request: QuestionRequest,
    http_request: Request,
    store: "VectorStore" = Depends(get_store),
    reranker: ReRanker = Depends(get_reranker),
    query_rewriter: Callable[[str], str] = Depends(get_query_rewriter),
    llm_stream=Depends(get_llm_stream),
    agent_router: Callable[[str, list[dict[str, str]]], RouteDecision] = Depends(
        get_agent_router
    ),
    conversation_store=Depends(get_conversation_store),
    memory_service=Depends(get_memory_service),
):
    _check_rate_limit(http_request)
    prep = await _prepare_agent_request(
        request=request,
        store=store,
        reranker=reranker,
        query_rewriter=query_rewriter,
        agent_router=agent_router,
        conversation_store=conversation_store,
        memory_service=memory_service,
        log_prefix="rag_stream",
    )
    request_id = prep.request_id
    state = prep.state

    if prep.terminal_message is not None:
        def generate_error():
            yield sse_event(
                "start",
                {
                    "request_id": request_id,
                    "conversation_id": request.conversation_id,
                },
            )
            code = "REFUSED" if prep.terminal_status == "refused" else "NO_CONTEXT"
            yield sse_event(
                "error",
                {
                    "request_id": request_id,
                    "code": code,
                    "message": prep.terminal_message,
                },
            )

        return StreamingResponse(
            generate_error(),
            media_type="text/event-stream; charset=utf-8",
        )

    context_text = "\n".join(item["text"] for item in prep.contexts)

    def generate():
        llm_start = time.perf_counter()
        yield sse_event(
            "start",
            {
                "request_id": request_id,
                "conversation_id": request.conversation_id,
            },
        )
        try:
            stream = _call_llm(
                llm_stream,
                prep.original_question,
                context_text,
                prep.history,
                prep.memory_text,
            )
            generated_parts = list(stream)
            answer = validate_answer("".join(generated_parts))
            prep.timings["llm_ms"] = round(
                (time.perf_counter() - llm_start) * 1000, 1
            )
            prep.timings["total_ms"] = round(
                (time.perf_counter() - prep.total_start) * 1000, 1
            )
            state.answer = answer
            state.decision = "success"

            # 先完成安全检查，再模拟分段发送，避免未检查内容提前泄露。
            for part in generated_parts:
                yield sse_event("token", {"content": part})
            conversation_store.append_turn(
                prep.conversation_id,
                prep.original_question,
                answer,
            )
            if memory_service is not None and request.user_id:
                try:
                    memory_service.remember_turn(
                        request.user_id,
                        prep.original_question,
                    )
                except Exception as exc:
                    logger.warning("长期记忆写入失败: %s", exc)
            logger.info(
                "rag_stream %s",
                build_rag_trace(
                    request_id,
                    request.conversation_id,
                    "direct_answer"
                    if prep.route_action == "direct_answer"
                    else "success",
                    len(prep.candidates),
                    len(prep.contexts),
                    prep.timings,
                    _agent_trace_fields(state),
                ),
            )
            yield sse_event(
                "done",
                {"request_id": request_id, "answer": answer},
            )
        except UnsafeOutputError as e:
            logger.warning("流式 LLM 输出未通过安全检查: %s", e)
            prep.timings["llm_ms"] = round(
                (time.perf_counter() - llm_start) * 1000, 1
            )
            prep.timings["total_ms"] = round(
                (time.perf_counter() - prep.total_start) * 1000, 1
            )
            state.failure_status = "invalid_output"
            logger.info(
                "rag_stream %s",
                build_rag_trace(
                    request_id,
                    request.conversation_id,
                    "invalid_output",
                    len(prep.candidates),
                    len(prep.contexts),
                    prep.timings,
                    _agent_trace_fields(state),
                ),
            )
            _journal_failure(
                request_id=request_id,
                conversation_id=request.conversation_id,
                status="invalid_output",
                question=prep.original_question,
                candidate_count=len(prep.candidates),
                context_count=len(prep.contexts),
                timings=prep.timings,
                detail="问答模型返回了无效内容",
            )
            yield sse_event(
                "error",
                {
                    "request_id": request_id,
                    "code": "INVALID_OUTPUT",
                    "message": "问答模型返回了无效内容",
                },
            )
        except Exception as e:
            logger.exception("流式 LLM 调用失败")
            prep.timings["llm_ms"] = round(
                (time.perf_counter() - llm_start) * 1000, 1
            )
            prep.timings["total_ms"] = round(
                (time.perf_counter() - prep.total_start) * 1000, 1
            )
            state.failure_status = "llm_error"
            logger.info(
                "rag_stream %s",
                build_rag_trace(
                    request_id,
                    request.conversation_id,
                    "llm_error",
                    len(prep.candidates),
                    len(prep.contexts),
                    prep.timings,
                    _agent_trace_fields(state),
                ),
            )
            _journal_failure(
                request_id=request_id,
                conversation_id=request.conversation_id,
                status="llm_error",
                question=prep.original_question,
                candidate_count=len(prep.candidates),
                context_count=len(prep.contexts),
                timings=prep.timings,
                detail="问答模型暂时不可用",
            )
            yield sse_event(
                "error",
                {
                    "request_id": request_id,
                    "code": "LLM_UNAVAILABLE",
                    "message": "问答模型暂时不可用",
                },
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
    )


@router.post("/agent/tool", response_model=AnswerResponse)
async def ask_tool_agent(
    request: QuestionRequest,
    http_request: Request,
    response: Response,
    store: "VectorStore" = Depends(get_store),
    reranker: ReRanker = Depends(get_reranker),
    tool_responder: Callable[[list[dict]], dict] = Depends(
        get_tool_agent_responder
    ),
    conversation_store=Depends(get_conversation_store),
    mcp_tool_provider=Depends(get_mcp_tool_provider),
):
    """标准 Function Calling Agent：LLM 自主决定是否调用检索工具。"""
    _check_rate_limit(http_request)
    request_id = new_request_id()
    timings: dict = {}
    total_start = time.perf_counter()
    question = request.question.strip()
    history = conversation_store.get_history(request.conversation_id)
    context_bucket: list[dict] = []

    # 工具声明优先来自 MCP Server；拿不到工具就退回内置本地工具，
    # 保证 MCP 子进程启动失败不会直接让接口不可用。
    tool_definitions = _mcp_tool_definitions(mcp_tool_provider)

    def local_execute_tool(name: str, arguments: dict) -> str:
        if name != KNOWLEDGE_SEARCH_TOOL_NAME:
            return json.dumps(
                {"error": f"工具 {name} 不在白名单内"},
                ensure_ascii=False,
            )
        query = (arguments.get("query") or "").strip()
        if not query:
            return json.dumps(
                {"error": "query 不能为空"},
                ensure_ascii=False,
            )
        try:
            candidates = store.hybrid_search_records(query, top_k=request.top_k * 3)
        except Exception as exc:
            logger.warning("Tool Agent 检索失败: %s", exc)
            return json.dumps(
                {"error": "知识库检索暂时不可用"},
                ensure_ascii=False,
            )
        if not candidates:
            return "知识库中没有找到相关文档。"

        try:
            contexts = reranker.rerank_records(query, candidates, request.top_k)
        except Exception as exc:
            logger.warning("Tool Agent Reranker 失败，退回混合检索: %s", exc)
            contexts = candidates[: request.top_k]
        if not contexts:
            return "检索结果没有达到相关性要求。"
        context_bucket.extend(contexts)
        return json.dumps(
            [to_context_response(item) for item in contexts],
            ensure_ascii=False,
        )

    def execute_tool(name: str, arguments: dict) -> str:
        """工具执行入口：MCP 模式下转发到 MCP Server，否则本地执行。"""
        if not tool_definitions:
            return local_execute_tool(name, arguments)
        try:
            raw = mcp_tool_provider.call_tool(name, arguments)
        except Exception as exc:
            logger.warning("MCP 工具调用失败: %s", exc)
            return json.dumps(
                {"error": f"工具 {name} 执行失败: {exc}"},
                ensure_ascii=False,
            )
        _collect_mcp_contexts(raw, context_bucket)
        return raw

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                run_tool_agent,
                question=question,
                history=history,
                execute_tool=execute_tool,
                respond=tool_responder,
                tool_definitions=tool_definitions,
                max_steps=3,
            ),
            timeout=settings.tool_agent_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error("Tool Agent 调用超时")
        raise HTTPException(status_code=503, detail="Agent 响应超时") from exc
    except Exception as exc:
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        logger.exception("Tool Agent 调用失败")
        _journal_failure(
            request_id=request_id,
            conversation_id=request.conversation_id,
            status="llm_error",
            question=question,
            candidate_count=len(context_bucket),
            context_count=len(context_bucket),
            timings=timings,
            detail="问答模型暂时不可用",
        )
        raise HTTPException(status_code=503, detail="问答模型暂时不可用") from exc

    timings["tool_agent_ms"] = round(
        (time.perf_counter() - total_start) * 1000, 1
    )
    response.headers["X-Agent-Tool-Calls"] = str(result.tool_call_count)
    response.headers["X-Agent-Steps"] = str(result.steps)
    response.headers["X-Agent-Termination"] = result.termination

    if result.termination == "max_steps" or not result.answer.strip():
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        _journal_failure(
            request_id=request_id,
            conversation_id=request.conversation_id,
            status="no_answer",
            question=question,
            candidate_count=len(context_bucket),
            context_count=len(context_bucket),
            timings=timings,
            detail=f"Tool Agent 达到最大步数仍未给出答案, tool_calls={result.tool_call_count}",
        )
        logger.info(
            "rag_answer %s",
            build_rag_trace(
                request_id,
                request.conversation_id,
                "no_answer",
                len(context_bucket),
                len(context_bucket),
                timings,
                {
                    "tool_calls": result.tool_call_count,
                    "termination": result.termination,
                },
            ),
        )
        return {
            "answer": "未能在限定步数内给出可靠答案。",
            "contexts": [],
        }

    try:
        answer = validate_answer(result.answer)
    except UnsafeOutputError as exc:
        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000, 1
        )
        logger.warning("Tool Agent 输出未通过安全检查: %s", exc)
        _journal_failure(
            request_id=request_id,
            conversation_id=request.conversation_id,
            status="invalid_output",
            question=question,
            candidate_count=len(context_bucket),
            context_count=len(context_bucket),
            timings=timings,
            detail="问答模型返回了无效内容",
        )
        raise HTTPException(status_code=503, detail="问答模型返回了无效内容") from exc

    timings["total_ms"] = round(
        (time.perf_counter() - total_start) * 1000, 1
    )
    conversation_store.append_turn(
        request.conversation_id,
        question,
        answer,
    )
    seen: set[str] = set()
    context_items = []
    for item in context_bucket:
        context = to_context_response(item)
        if context["chunk_id"] not in seen:
            seen.add(context["chunk_id"])
            context_items.append(context)
    logger.info(
        "rag_answer %s",
        build_rag_trace(
            request_id,
            request.conversation_id,
            "tool_agent_success",
            len(context_bucket),
            len(context_items),
            timings,
            {
                "tool_calls": result.tool_call_count,
                "termination": result.termination,
            },
        ),
    )
    return {"answer": answer, "contexts": context_items}
