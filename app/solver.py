"""区间反推核查的核心算法（全程精确整数运算）。

问题建模
--------
给定 r x c 非负整数计数矩阵，部分格子被遮盖（null），同时公开：

* 每一行的行合计 row_totals；
* 每一列的列合计 col_totals。

设第 k 个隐藏格的取值为变量 x_k（非负整数）。所有隐藏格与固定格必须
同时满足全部行/列合计等式，这是一个整数运输（二分网络流）可行性问题。

为什么不用浮点 LP
-----------------
旧实现借助 scipy/HiGHS 的浮点线性规划求界。float64 只有 53 位尾数，
当行/列合计超过 2**53 时，舍入误差会让本可取不同值的遮盖格被误判为
可还原，或把可行解报为冲突。本模块改为纯 Python 整数的最大流算法：
Python int 为任意精度整数，BIGINT（0 .. 2**63-1）乃至整表总计超过
2**63-1 时结果仍然逐位精确。

算法
----
运输网络（s -> 行节点 -> 列节点 -> t）：

* s -> 行 i，容量为该行扣除固定格后的剩余需求 row_demand[i]；
* 行 i -> 列 j（仅遮盖格），容量为隐藏格总流量（天然上界）；
* 列 j -> t，容量为该列剩余供给 col_supply[j]。

1. 可行性：s -> t 最大流达到隐藏格总流量即存在整数可行解，否则由最小割
   构造 Hall 型冲突见证。
2. 求单个遮盖格 e = (i, j) 的下界：先求一个任意可行流 f，再在其残量网络
   上求 j -> i 的最大流 g，则 e 可减少的最大流量就是 g，
   ``min(x_e) = f(e) - g``。
3. 求上界：同一残量网络上求 i -> j 的最大流 h，则 e 可增加的最大流量
   就是 h，``max(x_e) = f(e) + h``。
4. 下界等于上界即该格取值唯一（recoverable）。
5. 两张见证矩阵取扫描序第一个遮盖格：沿残量路径实际推送流量得到的新流，
   与公开固定格及全部行/列合计严格相符。

上述残量路径增广保持守恒关系与容量约束，因此得到的仍是同一运输问题的
整数可行解（全幺模性质在整数算法里体现为“整数容量 => 整数最大流”）。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 单个遮盖格计数与单条行/列合计允许的最大值（有符号 64 位整数上界）。
BIGINT_MAX = (1 << 63) - 1


class SuppressionConflict(ValueError):
    """约束无解时抛出，携带面向用户的冲突说明。"""

    def __init__(self, reasons: list[str], witness: dict | None = None):
        super().__init__("; ".join(reasons) if reasons else "infeasible constraints")
        self.reasons = reasons
        self.witness = witness


@dataclass(frozen=True)
class HiddenCell:
    index: int  # 变量序号（按行优先扫描）
    row: int
    col: int


# ---------------------------------------------------------------------------
# 输入检查
# ---------------------------------------------------------------------------

def validate_request(
    matrix: list[list[int | None]],
    row_totals: list[int],
    col_totals: list[int],
) -> tuple[int, int]:
    """防御性尺寸与类型校验。

    API 入口的 Pydantic 模型（schemas.SuppressionRequest）已完成严格的
    形状、类型与范围校验并返回 422；这里保留同构检查，保证 solver 可被
    独立调用。注意：整表总计不受 BIGINT_MAX 限制（Python int 任意精度），
    受限的只是单格计数与单条行/列合计。
    """
    if not matrix or not matrix[0]:
        raise SuppressionConflict(["matrix 不能为空矩阵"])
    rows = len(matrix)
    cols = len(matrix[0])
    if rows > 12 or cols > 12:
        raise SuppressionConflict(["matrix 尺寸不能超过 12x12"])
    if any(len(row) != cols for row in matrix):
        raise SuppressionConflict(["matrix 各行长度不一致"])
    if len(row_totals) != rows or len(col_totals) != cols:
        raise SuppressionConflict(["row_totals/col_totals 长度与矩阵尺寸不一致"])
    if any(
        (v is not None) and (not isinstance(v, int) or isinstance(v, bool) or v < 0)
        for row in matrix
        for v in row
    ) or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0
        for v in row_totals + col_totals
    ):
        raise SuppressionConflict(["所有计数与合计必须为非负整数"])
    if any(
        v is not None and v > BIGINT_MAX
        for row in matrix
        for v in row
    ) or any(v > BIGINT_MAX for v in row_totals + col_totals):
        raise SuppressionConflict(
            [f"单格计数与每项行/列合计不能超过有符号 64 位整数上界 {BIGINT_MAX}"]
        )
    return rows, cols


# ---------------------------------------------------------------------------
# Dinic 最大流（纯 Python 任意精度整数）
# ---------------------------------------------------------------------------

class _Edge:
    __slots__ = ("to", "rev", "cap")

    def __init__(self, to: int, rev: int, cap: int):
        self.to = to
        self.rev = rev
        self.cap = cap


class _Dinic:
    """整数 Dinic 最大流。容量全部为 Python int，无浮点、无精度损失。"""

    def __init__(self, n: int):
        self.g: list[list[_Edge]] = [[] for _ in range(n)]
        self.level = [-1] * n
        self.it = [0] * n

    def add_edge(self, fr: int, to: int, cap: int) -> _Edge:
        forward = _Edge(to, len(self.g[to]), cap)
        backward = _Edge(fr, len(self.g[fr]), 0)
        self.g[fr].append(forward)
        self.g[to].append(backward)
        return forward

    def _bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * len(self.g)
        self.level[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for e in self.g[v]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[v] + 1
                    queue.append(e.to)
        return self.level[t] >= 0

    def _dfs(self, v: int, t: int, f: int) -> int:
        if v == t:
            return f
        while self.it[v] < len(self.g[v]):
            e = self.g[v][self.it[v]]
            if e.cap > 0 and self.level[e.to] == self.level[v] + 1:
                d = self._dfs(e.to, t, min(f, e.cap))
                if d > 0:
                    e.cap -= d
                    self.g[e.to][e.rev].cap += d
                    return d
            self.it[v] += 1
        return 0

    def max_flow(self, s: int, t: int, limit: int) -> int:
        """从 s 到 t 推送最多 limit 的最大流（容量均为 Python 整数）。"""
        flow = 0
        while flow < limit:
            if not self._bfs(s, t):
                break
            self.it = [0] * len(self.g)
            while flow < limit:
                pushed = self._dfs(s, t, limit - flow)
                if pushed == 0:
                    break
                flow += pushed
        return flow

    def reachable_from(self, s: int) -> list[bool]:
        seen = [False] * len(self.g)
        seen[s] = True
        stack = [s]
        while stack:
            v = stack.pop()
            for e in self.g[v]:
                if e.cap > 0 and not seen[e.to]:
                    seen[e.to] = True
                    stack.append(e.to)
        return seen


# ---------------------------------------------------------------------------
# 运输网络构造
# ---------------------------------------------------------------------------

@dataclass
class _Network:
    dinic: _Dinic
    source: int
    sink: int
    row_base: int
    col_base: int
    # 每条遮盖格运输边在其行节点邻接表中的位置。
    edge_pos: list[int]


def _build_network(
    rows: int,
    cols: int,
    hidden: list[HiddenCell],
    row_demand: list[int],
    col_supply: list[int],
    hidden_total: int,
) -> _Network:
    """构造 s -> 行 -> 列 -> t 运输网络。

    遮盖格运输边容量取 hidden_total（任何单格流量不可能超过隐藏格总流量），
    避免使用脱离问题规模的“无限大”。
    """
    source = 0
    row_base = 1
    col_base = row_base + rows
    sink = col_base + cols
    dinic = _Dinic(sink + 1)

    for i, d in enumerate(row_demand):
        if d > 0:
            dinic.add_edge(source, row_base + i, d)
    for j, s in enumerate(col_supply):
        if s > 0:
            dinic.add_edge(col_base + j, sink, s)

    edge_pos: list[int] = []
    for h in hidden:
        edge_pos.append(len(dinic.g[row_base + h.row]))
        dinic.add_edge(row_base + h.row, col_base + h.col, hidden_total)

    return _Network(
        dinic=dinic,
        source=source,
        sink=sink,
        row_base=row_base,
        col_base=col_base,
        edge_pos=edge_pos,
    )


def _edge_flow(net: _Network, dinic: _Dinic, k: int, row: int) -> int:
    """遮盖格 k 在指定网络上的当前流量。

    正向边残量会因“固定边容量”调整而改变基准，不能用初始容量相减；
    反向边的残量始终等于该边已推送的流量（初始为 0，每推送 d 增加 d），
    与正向边初始容量无关，因此直接读取反向边残量最稳妥。
    """
    row_node = net.row_base + row
    forward = dinic.g[row_node][net.edge_pos[k]]
    return dinic.g[forward.to][forward.rev].cap


# ---------------------------------------------------------------------------
# 可行性判定与冲突见证
# ---------------------------------------------------------------------------

def _maxflow_feasibility(
    rows: int,
    cols: int,
    hidden: list[HiddenCell],
    row_demand: list[int],
    col_supply: list[int],
    hidden_total: int,
) -> tuple[dict | None, _Network | None]:
    """返回 (None, 可行网络) 表示可行；否则返回 (冲突见证字典, None)。"""
    net = _build_network(rows, cols, hidden, row_demand, col_supply, hidden_total)
    flow = net.dinic.max_flow(net.source, net.sink, hidden_total)
    if flow == hidden_total:
        return None, net

    reachable = net.dinic.reachable_from(net.source)
    r_subset = [i for i in range(rows) if reachable[net.row_base + i]]
    r_set = set(r_subset)
    # 与 R 之间存在遮盖格运输边的列即邻接列 N(R)；最小割保证 N(R) 全部
    # 可达（若不可达则该运输边横跨割集、容量为 hidden_total，与有限最小割矛盾）。
    neighbor_cols = sorted({h.col for h in hidden if h.row in r_set})
    demand = sum(row_demand[i] for i in r_subset)
    neighbor_supply = sum(col_supply[j] for j in neighbor_cols)
    witness = {
        "max_flow": flow,
        "required_flow": hidden_total,
        "row_subset": r_subset,
        "neighbor_cols": neighbor_cols,
        "subset_demand": demand,
        "neighbor_supply": neighbor_supply,
    }
    return witness, None


# ---------------------------------------------------------------------------
# 精确求界
# ---------------------------------------------------------------------------

def _flow_with_edge_upper_bound(
    rows: int,
    cols: int,
    hidden: list[HiddenCell],
    row_demand: list[int],
    col_supply: list[int],
    hidden_total: int,
    target_k: int,
    cap: int,
) -> tuple[_Dinic, _Network] | None:
    """判定目标遮盖边流量 **<= cap** 时是否存在可行整数流。

    把该边正向容量限制为 cap（其余边容量不变）后做 s-t 最大流，达到
    hidden_total 即可行。容量越宽松（cap 越大）越容易可行，因此对 cap
    是前缀单调，可二分求出最小可行 cap —— 即该边的最小可达流量 min。
    """
    net = _build_network(rows, cols, hidden, row_demand, col_supply, hidden_total)
    h = hidden[target_k]
    row_node = net.row_base + h.row
    net.dinic.g[row_node][net.edge_pos[target_k]].cap = cap
    flow = net.dinic.max_flow(net.source, net.sink, hidden_total)
    if flow != hidden_total:
        return None
    return net.dinic, net


def _flow_with_edge_exact(
    rows: int,
    cols: int,
    hidden: list[HiddenCell],
    row_demand: list[int],
    col_supply: list[int],
    hidden_total: int,
    target_k: int,
    value: int,
) -> tuple[_Dinic, _Network] | None:
    """构造“目标遮盖边恰好取 value”的可行整数流，不可行则返回 None。

    思路：先把 value 个单位预置到目标边（从对应行需求、列供给各扣 value），
    再把目标边在剩余问题中的容量清零，强制剩余流量全部经其他遮盖边消化。
    剩余问题可达剩余总量，当且仅当原问题存在“目标边 = value”的解。
    最后把预置 value 补记到目标边反向残量，读回流量恰为 value。
    """
    h = hidden[target_k]
    if value > row_demand[h.row] or value > col_supply[h.col]:
        return None

    rem_row_demand = row_demand[:]
    rem_col_supply = col_supply[:]
    rem_row_demand[h.row] -= value
    rem_col_supply[h.col] -= value
    rem_total = hidden_total - value

    net = _build_network(
        rows, cols, hidden, rem_row_demand, rem_col_supply, hidden_total
    )
    row_node = net.row_base + h.row
    target_edge = net.dinic.g[row_node][net.edge_pos[target_k]]
    target_edge.cap = 0  # 剩余流量不得再经过目标边

    if rem_total > 0:
        residual_flow = net.dinic.max_flow(net.source, net.sink, rem_total)
        if residual_flow != rem_total:
            return None

    # 补记预置流量：反向残量 = value，读回的目标边流量恰好为 value。
    net.dinic.g[target_edge.to][target_edge.rev].cap = value
    return net.dinic, net


def _flow_with_edge_lower_bound(
    rows: int,
    cols: int,
    hidden: list[HiddenCell],
    row_demand: list[int],
    col_supply: list[int],
    hidden_total: int,
    target_k: int,
    lower: int,
) -> tuple[_Dinic, _Network] | None:
    """判定目标遮盖边流量 **>= lower** 时是否存在可行整数流。

    把 lower 个单位预置到目标边（从对应行需求、列供给各扣 lower），目标边
    在剩余问题中仍可继续接收流量（容量收窄为 hidden_total - lower），再对
    剩余量做最大流。剩余问题可达剩余总量即可行。该判定对 lower 是后缀
    单调的（lower 越小约束越宽松），可二分求出最大可行 lower —— 即该边
    的最大可达流量 max。
    """
    h = hidden[target_k]
    if lower > row_demand[h.row] or lower > col_supply[h.col]:
        return None

    rem_row_demand = row_demand[:]
    rem_col_supply = col_supply[:]
    rem_row_demand[h.row] -= lower
    rem_col_supply[h.col] -= lower
    rem_total = hidden_total - lower

    net = _build_network(
        rows, cols, hidden, rem_row_demand, rem_col_supply, hidden_total
    )
    row_node = net.row_base + h.row
    target_edge = net.dinic.g[row_node][net.edge_pos[target_k]]
    target_edge.cap = hidden_total - lower

    if rem_total > 0:
        residual_flow = net.dinic.max_flow(net.source, net.sink, rem_total)
        if residual_flow != rem_total:
            return None
    # 补记预置 lower，使读回流量 = lower + 剩余推送（仅判定用途，不做见证）。
    net.dinic.g[target_edge.to][target_edge.rev].cap += lower
    return net.dinic, net


def _cell_bounds_and_witnesses(
    rows: int,
    cols: int,
    hidden: list[HiddenCell],
    row_demand: list[int],
    col_supply: list[int],
    hidden_total: int,
) -> tuple[list[tuple[int, int]], _Dinic, _Network, _Dinic, _Network]:
    """求每个遮盖格的精确 (min, max)，并产出首个遮盖格的两张见证流。

    纯整数、任意精度。可达流量集合是连续整数区间 [min, max]：

    * min —— 对上界限制做前缀二分：边容量 <= cap 下可行的最小 cap；
    * max —— 对下界限制做后缀二分：边流量 >= lower 下可行的最大 lower。

    首个遮盖格在两个界值处的可行流被保留为见证矩阵，分别达到其下界与
    上界，且与公开格、全部行/列合计严格相符。
    """
    bounds: list[tuple[int, int]] = []
    min_witness: tuple[_Dinic, _Network] | None = None
    max_witness: tuple[_Dinic, _Network] | None = None

    for k, h in enumerate(hidden):
        # ---- 下界：前缀二分最小可行上界 cap ----
        lo, hi = 0, hidden_total
        while lo < hi:
            mid = (lo + hi) // 2
            if (
                _flow_with_edge_upper_bound(
                    rows, cols, hidden, row_demand, col_supply,
                    hidden_total, k, mid,
                )
                is not None
            ):
                hi = mid
            else:
                lo = mid + 1
        lower = lo

        # ---- 上界：后缀二分最大可行下界 lower ----
        lo, hi = 0, hidden_total
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (
                _flow_with_edge_lower_bound(
                    rows, cols, hidden, row_demand, col_supply,
                    hidden_total, k, mid,
                )
                is not None
            ):
                lo = mid
            else:
                hi = mid - 1
        upper = lo

        # 首个遮盖格：在两个界值处各构造一次“目标边恰好取界值”的可行流
        # 作为见证矩阵，保证两张矩阵分别真实达到该格的下界与上界。
        if k == 0:
            min_result = _flow_with_edge_exact(
                rows, cols, hidden, row_demand, col_supply,
                hidden_total, k, lower,
            )
            max_result = _flow_with_edge_exact(
                rows, cols, hidden, row_demand, col_supply,
                hidden_total, k, upper,
            )
            assert min_result is not None and max_result is not None
            min_witness = min_result
            max_witness = max_result

        bounds.append((lower, upper))

    assert min_witness is not None and max_witness is not None
    min_witness_dinic, min_witness_net = min_witness
    max_witness_dinic, max_witness_net = max_witness
    return (
        bounds,
        min_witness_dinic,
        min_witness_net,
        max_witness_dinic,
        max_witness_net,
    )


def _flow_to_matrix(
    base: list[list[int | None]],
    hidden: list[HiddenCell],
    net: _Network,
    dinic: _Dinic,
) -> list[list[int]]:
    """把网络上的当前流还原为完整计数矩阵（固定格保持公开值）。"""
    matrix = [row[:] for row in base]
    for k, h in enumerate(hidden):
        matrix[h.row][h.col] = _edge_flow(net, dinic, k, h.row)
    return matrix


# ---------------------------------------------------------------------------
# 解的独立复核
# ---------------------------------------------------------------------------

def _verify_solution(
    matrix: list[list[int]],
    row_totals: list[int],
    col_totals: list[int],
) -> None:
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        if sum(matrix[i]) != row_totals[i]:
            raise SuppressionConflict(
                [f"内部校验失败：第 {i} 行合计与公开值不一致"]
            )
    for j in range(cols):
        if sum(matrix[i][j] for i in range(rows)) != col_totals[j]:
            raise SuppressionConflict(
                [f"内部校验失败：第 {j} 列合计与公开值不一致"]
            )
    if any(v < 0 for row in matrix for v in row):
        raise SuppressionConflict(["内部校验失败：解中出现负数"])
    if any(not isinstance(v, int) or isinstance(v, bool) for row in matrix for v in row):
        raise SuppressionConflict(["内部校验失败：解中出现非整数"])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def analyze(
    matrix: list[list[int | None]],
    row_totals: list[int],
    col_totals: list[int],
) -> dict:
    """执行完整核查，返回可直接序列化为 API 响应的字典。

    无解时抛出 :class:`SuppressionConflict`（调用方映射为 HTTP 409，
    且不携带任何取值区间）。
    """
    rows, cols = validate_request(matrix, row_totals, col_totals)

    row_sum = sum(row_totals)
    col_sum = sum(col_totals)
    if row_sum != col_sum:
        raise SuppressionConflict([
            f"行合计之和（{row_sum}）与列合计之和（{col_sum}）不相等，"
            "任何矩阵都无法同时满足两组合计"
        ])
    total = row_sum  # 可能超过 2**63-1，Python int 与 JSON 整数均可精确承载

    fixed_row = [0] * rows
    fixed_col = [0] * cols
    hidden: list[HiddenCell] = []
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v is None:
                hidden.append(HiddenCell(index=len(hidden), row=i, col=j))
            else:
                fixed_row[i] += v
                fixed_col[j] += v

    row_demand = [row_totals[i] - fixed_row[i] for i in range(rows)]
    col_supply = [col_totals[j] - fixed_col[j] for j in range(cols)]

    reasons: list[str] = []
    for i, d in enumerate(row_demand):
        if d < 0:
            reasons.append(
                f"第 {i} 行固定格之和（{fixed_row[i]}）已超过公开行合计"
                f"（{row_totals[i]}）"
            )
    for j, s in enumerate(col_supply):
        if s < 0:
            reasons.append(
                f"第 {j} 列固定格之和（{fixed_col[j]}）已超过公开列合计"
                f"（{col_totals[j]}）"
            )
    if reasons:
        raise SuppressionConflict(reasons)

    # 固定格之外、必须由隐藏格承担的总流量（行、列两侧应相等）。
    hidden_total = sum(row_demand)
    if hidden_total != sum(col_supply) or any(
        d < 0 for d in row_demand + col_supply
    ):
        # 负值已在上面收集；此处为防御性总量检查。
        raise SuppressionConflict(reasons or ["隐藏格需要满足的总流量不一致"])

    # 没有隐藏格：直接核对固定矩阵是否等于公开合计。
    if not hidden:
        fixed_matrix: list[list[int]] = [
            [v for v in row] for row in matrix  # type: ignore[arg-type]
        ]
        _verify_solution(fixed_matrix, row_totals, col_totals)
        return {
            "status": "feasible",
            "dimensions": {"rows": rows, "cols": cols},
            "grand_total": total,
            "hidden_count": 0,
            "recoverable_count": 0,
            "cells": [],
            "minimizer_matrix": fixed_matrix,
            "maximizer_matrix": fixed_matrix,
        }

    # 隐藏格总流量为 0：所有隐藏格都只能取 0，直接判定可还原。
    if hidden_total == 0:
        full_matrix = [
            [0 if v is None else v for v in row] for row in matrix
        ]
        _verify_solution(full_matrix, row_totals, col_totals)
        cells = [
            {
                "row": h.row,
                "col": h.col,
                "min": 0,
                "max": 0,
                "recoverable": True,
            }
            for h in hidden
        ]
        return {
            "status": "feasible",
            "dimensions": {"rows": rows, "cols": cols},
            "grand_total": total,
            "hidden_count": len(hidden),
            "recoverable_count": len(hidden),
            "cells": cells,
            "minimizer_matrix": full_matrix,
            "maximizer_matrix": full_matrix,
        }

    witness, feasible_net = _maxflow_feasibility(
        rows, cols, hidden, row_demand, col_supply, hidden_total
    )
    if witness is not None:
        r_set = ", ".join(str(i) for i in witness["row_subset"]) or "（空集）"
        c_set = (
            ", ".join(str(j) for j in witness["neighbor_cols"]) or "（无邻接列）"
        )
        conflict_reasons = [
            "不存在满足全部公开合计的非负整数矩阵：最大流可行性检验未通过",
            f"行子集 [{r_set}] 的隐藏格需求合计为 {witness['subset_demand']}，"
            f"但这些行的隐藏格只能进入列 [{c_set}]，其可承接供给仅为 "
            f"{witness['neighbor_supply']}（Hall 条件被破坏：需求 > 邻接供给）",
            f"最大流 {witness['max_flow']} < 隐藏格所需总流量 "
            f"{witness['required_flow']}",
        ]
        raise SuppressionConflict(conflict_reasons, witness=witness)

    (
        bounds,
        min_dinic,
        min_net,
        max_dinic,
        max_net,
    ) = _cell_bounds_and_witnesses(
        rows, cols, hidden, row_demand, col_supply, hidden_total
    )

    cell_results: list[dict] = []
    recoverable_count = 0
    for h, (lower, upper) in zip(hidden, bounds):
        if lower > upper:  # 防御性检查，正确算法下不会发生
            raise SuppressionConflict([
                f"隐藏格 ({h.row}, {h.col}) 求得最小值 {lower} 大于最大值 "
                f"{upper}，求解异常"
            ])
        recoverable = lower == upper
        if recoverable:
            recoverable_count += 1
        cell_results.append(
            {
                "row": h.row,
                "col": h.col,
                "min": lower,
                "max": upper,
                "recoverable": recoverable,
            }
        )

    # 两张见证矩阵：扫描序第一个隐藏格分别达到其下界、上界。每张矩阵由
    # “固定该边为界值后重新最大流”得到，必然与公开格及全部合计相符。
    first = hidden[0]
    min_matrix = _flow_to_matrix(matrix, hidden, min_net, min_dinic)
    max_matrix = _flow_to_matrix(matrix, hidden, max_net, max_dinic)

    _verify_solution(min_matrix, row_totals, col_totals)
    _verify_solution(max_matrix, row_totals, col_totals)
    if min_matrix[first.row][first.col] != cell_results[0]["min"]:
        raise SuppressionConflict(["内部校验失败：下界见证矩阵未达到首个隐藏格下界"])
    if max_matrix[first.row][first.col] != cell_results[0]["max"]:
        raise SuppressionConflict(["内部校验失败：上界见证矩阵未达到首个隐藏格上界"])

    return {
        "status": "feasible",
        "dimensions": {"rows": rows, "cols": cols},
        "grand_total": total,
        "hidden_count": len(hidden),
        "recoverable_count": recoverable_count,
        "cells": cell_results,
        "minimizer_matrix": min_matrix,
        "maximizer_matrix": max_matrix,
    }


def fraction_explanation() -> dict:
    """返回算法说明（供 /method 端点展示）。"""
    return {
        "model": "整数运输可行性（全幺模约束矩阵 + 整数容量网络流，全程精确整数）",
        "variables": "每个 null 格子对应一个非负整数变量",
        "constraints": "每个变量参与所在行、所在列的合计等式",
        "solver": (
            "纯 Python 整数 Dinic 最大流：可行流残量网络上做列->行/行->列"
            "增广，逐变量精确求 min/max，不使用浮点，支持 0..2^63-1 的"
            "单格与行列合计，整表总计超过该范围仍精确"
        ),
        "feasibility": "Dinic 最大流预判无解并由最小割给出 Hall 型冲突见证",
        "recoverable_rule": "min == max 时该隐藏格取值唯一，可被公开合计反推",
        "integrality_note": (
            "运输约束矩阵全幺模且容量/右侧均为整数，整数最大流给出的顶点"
            "天然为整数解；Python 任意精度整数保证 BIGINT 边界与超 2^63-1"
            " 的总计逐位精确，输出前另做合计复核"
        ),
    }
