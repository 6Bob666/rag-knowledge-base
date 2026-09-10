from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.health import router
from services.health_state import HealthState


def build_app(state: HealthState) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    # 健康路由使用模块状态；测试结束后由 fixture 风格的 finally 恢复。
    import routers.health as health_router

    app.state.original_health_state = health_router.health_state
    health_router.health_state = state
    return app


def test_liveness_does_not_require_models(monkeypatch):
    state = HealthState()
    app = build_app(state)
    try:
        with TestClient(app) as client:
            response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}
    finally:
        import routers.health as health_router

        health_router.health_state = app.state.original_health_state


def test_readiness_returns_503_while_starting(monkeypatch):
    state = HealthState()
    state.mark_starting()
    app = build_app(state)
    try:
        with TestClient(app) as client:
            response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "starting"
    finally:
        import routers.health as health_router

        health_router.health_state = app.state.original_health_state


def test_readiness_returns_503_when_dependency_failed():
    state = HealthState()
    state.mark_failed("reranker", "model file missing")
    app = build_app(state)
    try:
        with TestClient(app) as client:
            response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["checks"]["reranker"]["ok"] is False
    finally:
        import routers.health as health_router

        health_router.health_state = app.state.original_health_state


def test_readiness_returns_200_after_all_dependencies_ready():
    state = HealthState()
    state.mark_starting()
    state.mark_dependency_ready("vector_store")
    state.mark_dependency_ready("reranker")
    state.mark_ready()
    app = build_app(state)
    try:
        with TestClient(app) as client:
            response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    finally:
        import routers.health as health_router

        health_router.health_state = app.state.original_health_state
