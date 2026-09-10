from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from services.metrics_registry import metrics_registry

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics() -> str:
    return metrics_registry.render_prometheus()
