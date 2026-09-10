from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from logging_config import logger
from routers import documents, chat, health, metrics
from database import Base, engine
from models import Document
from services.dependencies import get_store

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 模型是应用级资源，在启动阶段加载一次并复用。
    logger.info("开始启动 RAG 应用")
    from services.health_state import health_state

    health_state.mark_starting()
    try:
        await run_in_threadpool(get_store)
        health_state.mark_dependency_ready("vector_store")
        await run_in_threadpool(chat.get_reranker)
        health_state.mark_dependency_ready("reranker")
    except Exception as exc:
        health_state.mark_failed("startup", str(exc))
        logger.exception("RAG 应用启动失败")
        raise
    health_state.mark_ready()
    logger.info("模型加载完成，应用已就绪")
    yield
    logger.info("RAG 应用正在关闭")
    from services.dependencies import shutdown_mcp_client

    shutdown_mcp_client()


app = FastAPI(title="RAG Knowledge Base", lifespan=lifespan)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    from time import monotonic

    started_at = monotonic()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        from services.metrics_registry import record_request_metric

        record_request_metric(
            request.method,
            request.url.path,
            status_code,
            started_at,
        )

# 注册路由
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(health.router)
app.include_router(metrics.router)

@app.get("/")
def root():
    return {"message": "RAG Knowledge Base API"}
