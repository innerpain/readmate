# agent eval（L2/L3/L4）

对 **Agent 主路径**（`/agent/chat` → `AgentRuntime`）的分层评测。L1 是独立的检索探针（`src/evaluation/retrieval_probe.py`），不在这里。

## 分层与归属

| 层 | 测什么 | 入口 |
|---|---|---|
| L0 预检 | 上游漂移：问题集 schema、索引指纹、collection 文档集、API 路由表 | `eval.agent_eval --preflight-only` |
| L1 检索 | 目标页排名 / recall@k / top-5 噪声（无 LLM，秒级） | `python -m src.evaluation.retrieval_probe` |
| L2 轨迹 | 闸门、轮数 vs 工具调用数、步数预算、按页 read 契约洞、被丢弃引用 | `eval.agent_eval` 的 `trace` / `assertions` |
| L3 答案 | citation 命中（文件,页，真源映射）、数字覆盖、误拒、无引用 | `eval.agent_eval` |
| L4 拒答 | `expect=deny`（正确否证）/ `expect=refuse`（该拒未拒）三态 | `eval.agent_eval` + `audit` 清单 |

## 三条口径铁律

1. **优先信原生信号，不猜文本。** `refused` / `failure` / `gate` 是结构化字段；关键词匹配只能当兜底。2026-09-15 那次 `unanswerable_refused=0` 是假警报——三道不可答题其实都是正确否证，只是 harness 在匹配别的链路的文案。
2. **每个比率必须带分母。** `answerable_number_coverage_ge_half=14` 大于 `answerable_with_numbers=12`，因为无数字参考答案的题被默认记为 coverage=1.0。现在 `number_coverage()` 对无数字返回 `None`，永不给免费的 1.0。
3. **每行只有一个 primary outcome。** 同一根因（步数预算耗尽）曾被同时算成「误拒」和「空引用」两个指标。

## outcome 取值

`ok` · `adjacent_page` · `no_citation` · `no_answer` · `budget_exhausted` · `gate_failed` · `transport_error` · `refuse_correct` · `deny_unverified` · `unfounded_answer`

`deny_unverified` / `unfounded_answer` 进入 `audit`，**未验证前不计为正确**。判定器分层：确定性规则优先，LLM judge 只在 `outcomes.judge_hook` 处预留，本轮未启用。

## 运行

在容器内跑（需要 chromadb / sentence-transformers / reranker 权重）。compose 已挂载 `data`/`src`/`config`，只需补 `eval`/`test_data`/`tmp`：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/eval:/app/eval" \
  -v "$PWD/test_data:/app/test_data" \
  -v "$PWD/tmp:/app/tmp" \
  api python -m eval.agent_eval \
    --questions test_data/evaluation_questions.jsonl,test_data/evaluation_questions_refusal.jsonl \
    --collection col_0d83015737fd \
    --out tmp/baselines/agent_eval_$(date +%Y%m%d).jsonl
```

L0 预检单独跑（不调模型）：

```bash
docker compose run --rm --no-deps -v ".../eval:/app/eval" api python -m eval.agent_eval --preflight-only --out tmp/preflight.jsonl
```

分段与回归：

```bash
python -m eval.agent_eval --indices 7,9,17 --out tmp/rerun.jsonl          # 只重跑指定题
python -m eval.agent_eval --only deny --resume --out tmp/retry.jsonl      # 只补跑判不了的行
python -m eval.agent_eval --baseline tmp/baselines/agent_eval_20260915.summary.json ...
```

## 硬约束

- **只写 `tmp/` 或显式 `--out`，永不写 `data/`**：`outcomes.assert_writable_output()` 会直接拒绝 `data/` 下的路径。
- **不污染产品库**：每次运行把 `data/app.db` 复制到 `<out>.work/app.db`，用 `AGENT_DB_PATH` 指过去。会话与记忆写进副本，collection 与文档关联照旧。
- **基线必须带指纹**：索引 `version_id` / `chunker_version` / `chunk_count`、collection 与文档集、agent 与 router 模型名、`AGENT_MAX_STEPS` / `AGENT_ROUTE_MODE` / `RETRIEVAL_MIN_SCORE` / `RERANK_ENABLED` / `HF_HUB_OFFLINE`、harness 版本。缺指纹的差异无法归因。

## 已知事实（不要当成 bug 重复发现）

- **Agent 路径没有充分性门控**：`min_score` 在 `persistent_retriever` / `adapter` / `agent` 里零引用，`search` 永远返回 top-k 命中 → 闸门只看「有没有搜」，不看「证据够不够」。所以 `expect=refuse` 离题题**大概率不会触发 `refused`**，它测的是幻觉作答风险，不是拒答路径。结构性拒答只在空库 / adapter 报错 / 空 collection 时可达（由 `tests/test_agent_runtime_gate.py` 覆盖）。
- **按页 read 不产生 observed chunk**（`tools.py::_read`：`hits=[payload] if result.chunk_id else []`）→ 该页证据无法被引用。模型目前都按 chunk_id 读，未触发。
- **`gate` 的键是精确等值断言**（`tests/test_agent_runtime_gate.py`）。插桩只能加在 `tool_trace` 与 `rounds` 上，不能往 `gate` 里塞字段。