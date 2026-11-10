"""FastAPI 层测试：状态码、响应结构、Swagger 示例与参数校验。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EX_HALL_CONFLICT, EX_RANGE, EX_TOTAL_CONFLICT

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "suppression-check"}


def test_feasible_endpoint():
    resp = client.post("/api/v1/suppression-check", json=EX_RANGE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "feasible"
    assert body["hidden_count"] == 6
    assert body["recoverable_count"] == 0
    assert len(body["cells"]) == 6
    assert len(body["minimizer_matrix"]) == 3
    assert set(body["cells"][0]) == {"row", "col", "min", "max", "recoverable"}


def test_total_conflict_returns_409_without_ranges():
    resp = client.post("/api/v1/suppression-check", json=EX_TOTAL_CONFLICT)
    assert resp.status_code == 409
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["reasons"]
    assert "witness" in body
    # 无解响应不得携带任何区间信息
    assert "cells" not in body
    assert "min" not in str(body)


def test_hall_conflict_returns_409_with_witness():
    resp = client.post("/api/v1/suppression-check", json=EX_HALL_CONFLICT)
    assert resp.status_code == 409
    body = resp.json()
    assert body["status"] == "conflict"
    assert body["witness"]["max_flow"] < body["witness"]["required_flow"]


def test_bad_shape_422():
    # 13 行超出 12x12 限制
    resp = client.post(
        "/api/v1/suppression-check",
        json={
            "matrix": [[0]] * 13,
            "row_totals": [0] * 13,
            "col_totals": [0],
        },
    )
    assert resp.status_code == 422

    # 行长度不一致
    resp = client.post(
        "/api/v1/suppression-check",
        json={"matrix": [[1, 2], [3]], "row_totals": [3, 3], "col_totals": [4, 2]},
    )
    assert resp.status_code == 422

    # 合计长度不匹配
    resp = client.post(
        "/api/v1/suppression-check",
        json={"matrix": [[1, None]], "row_totals": [1], "col_totals": [1]},
    )
    assert resp.status_code == 422

    # 负数 / 浮点 / 布尔
    for bad_value in (-1, 1.5, True):
        resp = client.post(
            "/api/v1/suppression-check",
            json={
                "matrix": [[bad_value, 0]],
                "row_totals": [1],
                "col_totals": [1, 0],
            },
        )
        assert resp.status_code == 422


def test_openapi_contains_examples():
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/api/v1/suppression-check"]["post"]["requestBody"]
    examples = body["content"]["application/json"]["examples"]
    summaries = {e["summary"] for e in examples.values()}
    assert any("区间" in s for s in summaries)
    assert any("可还原" in s for s in summaries)
    assert any("冲突" in s for s in summaries)
    # Swagger UI 可访问
    assert client.get("/docs").status_code == 200


def test_method_endpoint():
    resp = client.get("/method")
    assert resp.status_code == 200
    assert "全幺模" in resp.json()["model"]


def test_bigint_grand_total_exact_beyond_int64():
    """总计超 2^63 仍以 JSON 整数精确返回，遮盖格区间不被浮点误判。"""
    b63 = (1 << 63) - 1
    payload = {
        "matrix": [[None] * 12 for _ in range(12)],
        "row_totals": [b63] * 12,
        "col_totals": [b63] * 12,
    }
    resp = client.post("/api/v1/suppression-check", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    expected_total = 12 * b63
    assert body["grand_total"] == expected_total
    assert body["grand_total"] > (1 << 63) - 1
    # JSON 文本里必须是整数字面量，不带小数点/指数（精确承载）。
    raw = resp.content.decode()
    assert str(expected_total) in raw
    assert "." not in str(body["grand_total"])
    assert all(c["recoverable"] is False for c in body["cells"])
    first = body["cells"][0]
    assert body["minimizer_matrix"][first["row"]][first["col"]] == first["min"]
    assert body["maximizer_matrix"][first["row"]][first["col"]] == first["max"]


def test_int64_overflow_returns_422():
    """单格或单项合计超 2^63-1 在参数校验层即 422。"""
    b63 = (1 << 63) - 1
    overflow_cell = {
        "matrix": [[b63 + 1, None]],
        "row_totals": [b63 + 1],
        "col_totals": [b63 + 1, 0],
    }
    resp = client.post("/api/v1/suppression-check", json=overflow_cell)
    assert resp.status_code == 422

    overflow_margin = {
        "matrix": [[1, None]],
        "row_totals": [b63 + 1],
        "col_totals": [1, b63],
    }
    resp = client.post("/api/v1/suppression-check", json=overflow_margin)
    assert resp.status_code == 422


def test_bigint_recoverable_not_misjudged():
    """BIGINT 边界下本可还原的格子 min==max，且响应结构保持不变。"""
    b63 = (1 << 63) - 1
    payload = {
        "matrix": [[b63, None], [None, b63]],
        "row_totals": [b63, b63],
        "col_totals": [b63, b63],
    }
    resp = client.post("/api/v1/suppression-check", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["grand_total"] == 2 * b63
    assert body["hidden_count"] == 2
    assert body["recoverable_count"] == 2
    assert all(c["recoverable"] and c["min"] == c["max"] for c in body["cells"])
    assert set(body) == {
        "status",
        "dimensions",
        "grand_total",
        "hidden_count",
        "recoverable_count",
        "cells",
        "minimizer_matrix",
        "maximizer_matrix",
    }
