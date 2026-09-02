"""Integration checks for the FastAPI application wiring.

These tests verify the real application entry point rather than inspecting
``app.routes`` as a flat list. FastAPI >= 0.137 preserves included routers as
nested router objects, so OpenAPI is the stable public view of registered API
paths.
"""

from fastapi.testclient import TestClient

from app import app
from api import query_routes
from insight_agent_industry import IndustryInsightAgent


def test_main_application_exposes_pipeline_routes():
    paths = set(app.openapi()["paths"])

    expected = {
        "/api/auth/login",
        "/api/connect",
        "/api/schema",
        "/api/ask",
        "/api/insight",
        "/api/health",
        "/api/providers",
    }

    assert expected.issubset(paths), f"Missing application routes: {sorted(expected - paths)}"


def test_api_uses_industry_insight_agent():
    assert query_routes.InsightAgent is IndustryInsightAgent
    assert query_routes.InsightAgent.VERSION == "3.0"


def test_app_starts_and_public_health_endpoints_respond():
    with TestClient(app) as client:
        health = client.get("/api/health")
        providers = client.get("/api/providers")
        index = client.get("/")

    assert health.status_code == 200
    assert providers.status_code == 200
    assert index.status_code == 200
    assert "providers" in providers.json()
