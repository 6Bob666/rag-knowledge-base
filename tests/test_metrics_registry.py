from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.metrics import router
from services.metrics_registry import MetricsRegistry


def test_metrics_registry_records_status_and_p95():
    registry = MetricsRegistry(max_samples_per_key=10)
    registry.record_request(
        method="post", path="/chat/ask", status_code=200, duration_ms=10
    )
    registry.record_request(
        method="POST", path="/chat/ask", status_code=503, duration_ms=100
    )
    registry.record_request(
        method="POST", path="/chat/ask", status_code=200, duration_ms=20
    )
    text = registry.render_prometheus()
    assert 'method="POST",path="/chat/ask",status="200"} 2' in text
    assert 'method="POST",path="/chat/ask",status="503"} 1' in text
    assert 'rag_http_request_duration_ms_count{method="POST",path="/chat/ask"} 3' in text
    assert 'rag_http_request_duration_ms_p95{method="POST",path="/chat/ask"}' in text


def test_metrics_endpoint_returns_plain_text():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "rag_http_requests_total" in response.text
