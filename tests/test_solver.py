"""solver 单元测试：已知区间、可还原、冲突，以及暴力枚举交叉验证。"""

from __future__ import annotations

from itertools import product

import pytest

from app import solver
from app.schemas import EX_HALL_CONFLICT, EX_RANGE, EX_RECOVERABLE, EX_TOTAL_CONFLICT


def _feasible_assignments(matrix, row_totals, col_totals):
    """枚举隐藏格全部非负整数赋值，返回满足所有合计的赋值字典列表。"""
    rows, cols = len(matrix), len(matrix[0])
    hidden = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if matrix[i][j] is None
    ]
    total = sum(row_totals)
    solutions = []
    for values in product(range(total + 1), repeat=len(hidden)):
        full = [row[:] for row in matrix]
        for (i, j), v in zip(hidden, values):
            full[i][j] = v
        if all(sum(full[i]) == row_totals[i] for i in range(rows)) and all(
            sum(full[i][j] for i in range(rows)) == col_totals[j]
            for j in range(cols)
        ):
            solutions.append(dict(zip(hidden, values)))
    return solutions


def test_range_example_known_bounds():
    result = solver.analyze(**EX_RANGE)
    assert result["status"] == "feasible"
    assert result["hidden_count"] == 6
    assert result["recoverable_count"] == 0
    by_pos = {(c["row"], c["col"]): c for c in result["cells"]}
    expected = {
        (0, 1): (0, 4),
        (0, 2): (3, 7),
        (1, 0): (0, 4),
        (1, 2): (3, 7),
        (2, 0): (0, 4),
        (2, 1): (2, 6),
    }
    for pos, (lo, hi) in expected.items():
        assert by_pos[pos]["min"] == lo
        assert by_pos[pos]["max"] == hi
        assert by_pos[pos]["recoverable"] is False

    # 两张完整矩阵都必须真实满足公开合计
    for key in ("minimizer_matrix", "maximizer_matrix"):
        m = result[key]
        assert all(
            sum(m[i]) == EX_RANGE["row_totals"][i] for i in range(3)
        )
        assert all(
            sum(m[i][j] for i in range(3)) == EX_RANGE["col_totals"][j]
            for j in range(3)
        )
        assert all(v >= 0 for row in m for v in row)

    # 极值矩阵对扫描序第一个隐藏格 (0,1) 达到各自界
    first = result["cells"][0]
    assert result["minimizer_matrix"][0][1] == first["min"]
    assert result["maximizer_matrix"][0][1] == first["max"]


def test_recoverable_example_all_locked():
    result = solver.analyze(**EX_RECOVERABLE)
    assert result["hidden_count"] == 4
    assert result["recoverable_count"] == 4
    assert all(c["recoverable"] and c["min"] == c["max"] for c in result["cells"])
    expected_full = [[2, 5, 3], [2, 4, 4], [5, 1, 3]]
    assert result["minimizer_matrix"] == expected_full
    assert result["maximizer_matrix"] == expected_full
    by_pos = {(c["row"], c["col"]): c["min"] for c in result["cells"]}
    assert by_pos == {(0, 1): 5, (1, 0): 2, (1, 2): 4, (2, 2): 3}


def test_total_mismatch_conflict():
    with pytest.raises(solver.SuppressionConflict) as exc:
        solver.analyze(**EX_TOTAL_CONFLICT)
    assert "行合计之和" in exc.value.reasons[0]
    assert exc.value.witness is None


def test_hall_conflict_with_witness():
    with pytest.raises(solver.SuppressionConflict) as exc:
        solver.analyze(**EX_HALL_CONFLICT)
    assert exc.value.witness is not None
    w = exc.value.witness
    assert w["max_flow"] < w["required_flow"]
    # Hall 条件：行子集需求 > 其邻接列可承接的供给
    assert w["subset_demand"] > w["neighbor_supply"]
    assert 0 in w["row_subset"] and 0 in w["neighbor_cols"]
    assert "最大流" in " ".join(exc.value.reasons)


def test_fixed_cells_exceed_totals():
    payload = {
        "matrix": [[9, None], [None, 1]],
        "row_totals": [5, 5],
        "col_totals": [10, 0],
    }
    with pytest.raises(solver.SuppressionConflict) as exc:
        solver.analyze(**payload)
    assert any("超过公开行合计" in r for r in exc.value.reasons)


def test_no_hidden_cells_matching():
    payload = {
        "matrix": [[1, 2], [3, 4]],
        "row_totals": [3, 7],
        "col_totals": [4, 6],
    }
    result = solver.analyze(**payload)
    assert result["hidden_count"] == 0
    assert result["cells"] == []


def test_no_hidden_cells_mismatch():
    payload = {
        "matrix": [[1, 2], [3, 4]],
        "row_totals": [3, 7],
        "col_totals": [5, 5],
    }
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze(**payload)


def test_brute_force_cross_check():
    """随机小规模可行问题：LP 区间必须与暴力枚举完全一致。"""
    import random

    rng = random.Random(20261110)
    checked = 0
    for seed in range(40):
        rng.seed(1000 + seed)
        rows, cols = rng.randint(2, 3), rng.randint(2, 3)
        full = [
            [rng.randint(0, 3) for _ in range(cols)] for _ in range(rows)
        ]
        row_totals = [sum(r) for r in full]
        col_totals = [sum(full[i][j] for i in range(rows)) for j in range(cols)]
        if sum(row_totals) > 14:
            continue

        positions = [(i, j) for i in range(rows) for j in range(cols)]
        k = rng.randint(2, min(3, len(positions)))
        masked = set(rng.sample(positions, k))
        matrix = [
            [None if (i, j) in masked else full[i][j] for j in range(cols)]
            for i in range(rows)
        ]

        result = solver.analyze(matrix, row_totals, col_totals)
        brute = _feasible_assignments(matrix, row_totals, col_totals)
        assert brute, "随机问题由真实矩阵生成，必然存在可行解"
        checked += 1

        for cell in result["cells"]:
            pos = (cell["row"], cell["col"])
            vals = [sol[pos] for sol in brute]
            assert cell["min"] == min(vals)
            assert cell["max"] == max(vals)
            assert cell["recoverable"] == (min(vals) == max(vals))

        # 极值矩阵中的对应格确实达到界，且矩阵整体可行
        first_pos = (result["cells"][0]["row"], result["cells"][0]["col"])
        assert (
            result["minimizer_matrix"][first_pos[0]][first_pos[1]]
            == result["cells"][0]["min"]
        )
        assert (
            result["maximizer_matrix"][first_pos[0]][first_pos[1]]
            == result["cells"][0]["max"]
        )
    assert checked >= 20


def test_brute_force_on_infeasible():
    """随机矩阵 + 随机合计：solver 判无解当且仅当枚举无可行解。"""
    import random

    rng = random.Random(424242)
    for seed in range(30):
        rng.seed(seed)
        rows = cols = 2
        matrix = [
            [rng.randint(0, 4) if rng.random() < 0.4 else None for _ in range(cols)]
            for _ in range(rows)
        ]
        if not any(v is None for row in matrix for v in row):
            matrix[0][0] = None
        row_totals = [rng.randint(0, 10) for _ in range(rows)]
        col_totals = [rng.randint(0, 10) for _ in range(cols)]
        brute = _feasible_assignments(matrix, row_totals, col_totals)
        try:
            result = solver.analyze(matrix, row_totals, col_totals)
            assert brute, "solver 认为可行，但暴力枚举无解"
            for cell in result["cells"]:
                pos = (cell["row"], cell["col"])
                vals = [sol[pos] for sol in brute]
                assert cell["min"] == min(vals)
                assert cell["max"] == max(vals)
        except solver.SuppressionConflict:
            assert not brute, "solver 判无解，但暴力枚举存在可行解"


def test_max_size_12x12_runs():
    import random

    rng = random.Random(7)
    n = 12
    full = [[rng.randint(0, 99) for _ in range(n)] for _ in range(n)]
    row_totals = [sum(r) for r in full]
    col_totals = [sum(full[i][j] for i in range(n)) for j in range(n)]
    matrix = [[v for v in row] for row in full]
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                matrix[i][j] = None
    result = solver.analyze(matrix, row_totals, col_totals)
    assert result["dimensions"] == {"rows": 12, "cols": 12}
    assert result["hidden_count"] == 72
    for m in (result["minimizer_matrix"], result["maximizer_matrix"]):
        assert all(sum(m[i]) == row_totals[i] for i in range(n))
        assert all(
            sum(m[i][j] for i in range(n)) == col_totals[j] for j in range(n)
        )


def test_validation_errors_solver_level():
    # solver 被独立调用时，结构性/类型性输入问题同样以冲突异常拒绝
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze([[0]] * 13, [0] * 13, [0])
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze([[-1, None]], [0], [0, 0])
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze([[1, None]], [0, 1], [1, 0])
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze([[True, None]], [1], [1, 0])


def test_schema_rejects_malformed_inputs():
    """Pydantic 层负责形状/类型/负数，统一返回 ValueError（API 中即 422）。"""
    import pytest as _pt

    from app.schemas import SuppressionRequest

    bad_cases = [
        {"matrix": [[0]] * 13, "row_totals": [0] * 13, "col_totals": [0]},
        {"matrix": [[1, 2], [3]], "row_totals": [3, 3], "col_totals": [4, 2]},
        {"matrix": [[1, None]], "row_totals": [1], "col_totals": [1]},
        {"matrix": [[-1, 0]], "row_totals": [1], "col_totals": [1, 0]},
        {"matrix": [[1.5, 0]], "row_totals": [1], "col_totals": [1, 0]},
        {"matrix": [[True, 0]], "row_totals": [1], "col_totals": [1, 0]},
        {"matrix": [[]], "row_totals": [0], "col_totals": []},
    ]
    for payload in bad_cases:
        with _pt.raises(Exception):
            SuppressionRequest(**payload)


# ---------------------------------------------------------------------------
# BIGINT 边界与任意精度整数（本次修复的回归测试）
# ---------------------------------------------------------------------------

B63 = (1 << 63) - 1


def test_bigint_boundary_recoverable_exact():
    """单格与行列合计取 2^63-1：纯整数精确求解，不被浮点舍入误伤。"""
    payload = {
        "matrix": [[B63, None], [None, B63]],
        "row_totals": [B63, B63],
        "col_totals": [B63, B63],
    }
    result = solver.analyze(**payload)
    # 整表总计 = 2*(2^63-1)，超过 2^63-1 仍须精确返回。
    assert result["grand_total"] == 2 * B63
    assert result["grand_total"] > B63
    by_pos = {(c["row"], c["col"]): c for c in result["cells"]}
    for pos in ((0, 1), (1, 0)):
        cell = by_pos[pos]
        assert cell["min"] == 0 and cell["max"] == 0
        assert cell["recoverable"] is True
    for key in ("minimizer_matrix", "maximizer_matrix"):
        m = result[key]
        assert all(sum(m[i]) == payload["row_totals"][i] for i in range(2))
        assert all(
            sum(m[i][j] for i in range(2)) == payload["col_totals"][j]
            for j in range(2)
        )


def test_grand_total_beyond_int64_is_exact():
    """每条行列合计 <= 2^63-1，但 12 行累加之和远超 2^63，须逐位精确。"""
    n = 12
    matrix = [[None] * n for _ in range(n)]
    row_totals = [B63] * n
    col_totals = [B63] * n
    result = solver.analyze(matrix, row_totals, col_totals)
    expected_total = 12 * B63
    assert result["grand_total"] == expected_total
    assert result["grand_total"] > (1 << 63) - 1
    # 全遮盖规则二分图：每个遮盖格可取 0..B63，均不可还原。
    assert result["hidden_count"] == n * n
    assert all(c["min"] == 0 and c["max"] == B63 for c in result["cells"])
    assert all(c["recoverable"] is False for c in result["cells"])
    first = result["cells"][0]
    assert result["minimizer_matrix"][first["row"]][first["col"]] == 0
    assert result["maximizer_matrix"][first["row"]][first["col"]] == B63
    for m in (result["minimizer_matrix"], result["maximizer_matrix"]):
        assert all(sum(m[i]) == B63 for i in range(n))
        assert all(sum(m[i][j] for i in range(n)) == B63 for j in range(n))


def test_scale_above_2p53_keeps_real_ranges():
    """2^53 之上旧浮点 LP 会误判；这里有自由区间的遮盖格必须仍不可还原。"""
    p53 = 1 << 53
    payload = {
        "matrix": [
            [p53, None, None],
            [None, p53, None],
            [None, None, p53],
        ],
        "row_totals": [3 * p53] * 3,
        "col_totals": [3 * p53] * 3,
    }
    result = solver.analyze(**payload)
    assert result["grand_total"] == 9 * p53
    # 与小规模同构问题对照：自由块遮盖格存在非平凡区间。
    assert any(c["min"] < c["max"] for c in result["cells"])
    assert any(c["recoverable"] is False for c in result["cells"])
    for m in (result["minimizer_matrix"], result["maximizer_matrix"]):
        assert all(sum(m[i]) == payload["row_totals"][i] for i in range(3))
        assert all(
            sum(m[i][j] for i in range(3)) == payload["col_totals"][j]
            for j in range(3)
        )


def test_solver_rejects_cell_or_margin_above_int64():
    """单格或单项行列合计超过 2^63-1 时 solver 层同样拒绝。"""
    over = B63 + 1
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze([[over, None]], [over], [over, 0])
    with pytest.raises(solver.SuppressionConflict):
        solver.analyze([[1, None]], [over], [1, over - 1])


def test_witness_matrices_hit_first_cell_bounds_bigint():
    """大整数见证矩阵必须让首个遮盖格分别达到其下界与上界。"""
    payload = {
        "matrix": [
            [B63 // 2, None, None, 100],
            [None, B63 // 2, 50, None],
            [None, 70, B63 // 2, None],
            [200, None, None, B63 // 2],
        ],
        "row_totals": [
            4611686020427387903,
            4611686021427387903,
            4611686019427387903,
            4611686022427387903,
        ],
        "col_totals": [
            4611686021127388103,
            4611686020727387973,
            4611686022127387653,
            4611686019727387883,
        ],
    }
    result = solver.analyze(**payload)
    assert result["grand_total"] > B63
    first = result["cells"][0]
    assert first["min"] < first["max"]
    assert (
        result["minimizer_matrix"][first["row"]][first["col"]] == first["min"]
    )
    assert (
        result["maximizer_matrix"][first["row"]][first["col"]] == first["max"]
    )
