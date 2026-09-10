"""Kubernetes/Docker 风格的健康检查接口。"""

from fastapi import APIRouter, HTTPException

from services.health_state import health_state

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live")
def liveness_probe() -> dict:
    """进程能响应 HTTP 即视为存活。"""
    return {"status": "alive"}


@router.get("/ready")
def readiness_probe() -> dict:
    """只有 lifespan 完成依赖加载后才允许接收业务流量。"""
    snapshot = health_state.snapshot()
    if not health_state.is_ready():
        raise HTTPException(status_code=503, detail=snapshot)
    return snapshot
