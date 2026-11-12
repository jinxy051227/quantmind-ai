import pytest

def test_health_endpoint():
    fastapi=pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from service_api import app
    c=TestClient(app)
    r=c.get("/health")
    assert r.status_code==200
    assert r.json()["version"]=="1.0.0"

def test_bounded_demo_input():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from service_api import app
    c=TestClient(app)
    assert c.post("/v1/demo/study",json={"rows":100}).status_code==422
