# 医院计数矩阵遮盖核查 API

发布分年龄组（列）、乡镇（行）的病例计数前，用本服务检查 `null` 遮盖格
能否被**公开的行/列合计精确反推**。对每个隐藏格，在所有合计同时成立的
前提下返回：

- 可行最小值 `min`、最大值 `max`（隐藏值可取任意非负整数）；
- 分别达到两界的完整可行矩阵 `minimizer_matrix` / `maximizer_matrix`；
- `min == max` 时标记 `recoverable: true`，表示该格可被唯一反推，
  工程师应调整遮盖位置后重新核查；
- 约束无解时返回 HTTP 409 与结构化冲突说明，**不输出任何区间**。

## 大整数（BIGINT）精度保证

- 单个遮盖格计数与每一项行/列合计支持 **0 到 2^63−1**
  （有符号 64 位整数 `BIGINT` 上界，`9223372036854775807`）；
- **整表总计 `grand_total` 不受此限**：即使超过 2^63−1 也以 JSON 整数
  逐位精确返回（Python 任意精度整数，全程无浮点运算）；
- 求界与可行性判定全部基于纯 Python 整数最大流，不使用 `float`/`numpy`/
  `scipy`，因此在 2^53 之上不会出现舍入导致的“有区间被误判可还原”或
  “可行解被误报冲突”。

`examples/bigint-request.json` 是一份可直接 `POST` 到本 API 的大整数
请求示例：单格与各项行列合计均在 BIGINT 范围内，而整表总计约
1.84×10^19（超过 2^63），8 个遮盖格各自带非平凡取值区间。

```bash
curl -s -X POST http://localhost:8000/api/v1/suppression-check \
  -H "Content-Type: application/json" \
  --data @examples/bigint-request.json
```

## 目录结构

```
.
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI 应用与路由（含 Swagger 示例）
│   ├── schemas.py     # Pydantic 请求/响应模型、BIGINT 范围校验、内置示例
│   └── solver.py      # 纯 Python 整数 Dinic 最大流：可行性判定 + 精确求界
├── examples/
│   └── bigint-request.json   # 可直接提交的大整数（总计超 2^63）请求示例
├── tests/             # pytest 测试（含暴力枚举交叉验证与 BIGINT 回归）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## 启动方式（Docker Compose）

```bash
docker compose up --build
```

启动后：

- Swagger UI：<http://localhost:8000/docs> （可直接选用内置示例提交）
- ReDoc：<http://localhost:8000/redoc>
- 健康检查：<http://localhost:8000/health>

停止：`docker compose down`。

## 请求与响应

`POST /api/v1/suppression-check`

请求体（矩阵最多 12×12；`null` 为遮盖格，其余为固定非负整数）：

```json
{
  "matrix": [
    [5, null, null],
    [null, 3, null],
    [null, null, 2]
  ],
  "row_totals": [12, 10, 8],
  "col_totals": [9, 9, 12]
}
```

可行响应（HTTP 200，节选）：

```json
{
  "status": "feasible",
  "grand_total": 30,
  "hidden_count": 6,
  "recoverable_count": 0,
  "cells": [
    {"row": 0, "col": 1, "min": 0, "max": 4, "recoverable": false}
  ],
  "minimizer_matrix": [[5, 0, 7], [4, 3, 3], [0, 6, 2]],
  "maximizer_matrix": [[5, 4, 3], [0, 3, 7], [4, 2, 2]]
}
```

无解响应（HTTP 409，不含任何区间信息）：

```json
{
  "status": "conflict",
  "reasons": [
    "不存在满足全部公开合计的非负整数矩阵：最大流可行性检验未通过",
    "行子集 [0] 的隐藏格需求合计为 6，但这些行的隐藏格只能进入列 [0]，其可承接供给仅为 2（Hall 条件被破坏：需求 > 邻接供给）",
    "最大流 6 < 隐藏格所需总流量 10"
  ],
  "witness": {
    "max_flow": 6,
    "required_flow": 10,
    "row_subset": [0],
    "neighbor_cols": [0],
    "subset_demand": 6,
    "neighbor_supply": 2
  }
}
```

状态码约定：

| 场景 | 状态码 |
| --- | --- |
| 存在可行矩阵 | 200，返回区间与两张完整矩阵 |
| 合计约束无解（总量不符 / 固定格超限 / Hall 条件破坏） | 409，返回冲突原因与见证，不含任何区间 |
| 请求结构或类型错误（超 12×12、负数、布尔、长度不符） | 422，Pydantic 校验信息 |
| 单格计数或单项行/列合计超过 2^63−1 | 422，BIGINT 范围校验信息（整表总计不在此限） |

## 算法说明

1. **建模**：每个遮盖格对应一个非负整数变量，行/列合计构成等式约束，
   即整数运输（二分网络流）问题。运输约束矩阵是全幺模矩阵，整数容量下
   最大流给出的顶点天然是整数解。
2. **可行性**：用自实现的整数 Dinic 最大流构造网络
   source→行（剩余行需求）、遮盖格所在行列之间建容量边、列→sink
   （剩余列供给）。最大流达不到隐藏格总流量即无解，并由最小割给出
   Hall 型冲突见证（哪些行的需求超出其邻接列供给）。
3. **精确求界**（全程 Python 任意精度整数，无浮点）：遮盖格的可达流量
   构成连续整数区间 `[min, max]`。对每个格子配合二分搜索重复最大流：
   - 求 `min`：把该边容量上界限制为 `cap`，前缀二分最小可行 `cap`；
   - 求 `max`：预置 `lower` 单位到该边后对剩余量做最大流，后缀二分
     最大可行 `lower`。
4. **见证矩阵**：对扫描序第一个遮盖格，分别在其下界、上界处用“预置 +
   剩余最大流”构造两张完整可行矩阵，保证该格恰好取到界值，并与公开格、
   全部行/列合计严格相符；返回前再独立逐行逐列复核。
5. **可还原判定**：`min == max` 即该格取值唯一，可被公开合计反推。

复杂度：设隐藏格数为 h（h ≤ 144），每格做 O(log T) 次整数最大流
（T 为隐藏格总流量），12×12 实测在数秒内完成，且耗时与数值位数无关地
保持精确。

## 本地开发（无 Docker 时）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
.venv/bin/python -m pytest tests -q
```

测试集包含随机小规模问题的**暴力枚举交叉验证**：整数最大流求出的每个
区间与枚举全部可行赋值的 min/max 完全一致，并覆盖无解、可还原、零矩阵、
1×1、12×12、2^53 量级、BIGINT 边界、超 2^63 总计与各类 422/409 场景。
