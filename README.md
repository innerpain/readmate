# Document QA Assistant（ReadMate）

**本地单用户、多 PDF、引用可追溯的 RAG + Agent 问答服务。** PDF、解析真源、切块产物与向量索引全部留在本机 `data/`；每次提问只把本轮证据 Context 发给模型，不上传原始 PDF。

- 入库：Docling 解析 → `ordered_document_v1` 真源 → 结构感知切块 → 版本化 Chroma 索引（Celery 异步，重启可恢复）。
- 问答：ReadMate Agent（ReAct + 证据闸门），引用只允许来自**本轮工具真实观察到的 chunk**。
- 界面：React SPA，由 FastAPI 同源托管（含 SSE 流式、引用卡片、阅读轨迹、记忆确认）。

权威方案（实现细节以此为准）：

> 设计文档（方案册、问题台账、接续记录等）属内部资料，**不在本仓库内**；本 README 与 `docs/` 之外的代码注释是唯一公开口径。

| 文档 | 内容 |
|------|------|
| [`eval/README.md`](eval/README.md) | Agent 分层评测口径与运行方式 |

## 快速开始

前置：Docker Desktop（WSL2）、NVIDIA 驱动与可用 GPU（`api` / `worker` 均 `gpus: all`，解析与 embedding 走 CUDA）、磁盘余量 30GB 级（torch + Docling + CUDA wheels；两个镜像的层实际共享）。

```bash
cp .env.example .env    # Windows: copy .env.example .env
# 填 API_KEY / BASE_URL / MODEL（OpenAI 兼容端点，只收本轮 Context）

docker compose build    # 首次构建为「数十分钟级」；此后增量重建
docker compose up -d
docker compose ps
curl http://127.0.0.1:8000/health
```

**`BASE_IMAGE` 构建顺序（D50）**：`Dockerfile` 是 `FROM ${BASE_IMAGE}` —— 增量叠加层，**不是从零构建**。因此第一次构建前必须先在本地存在一个基础镜像（torch + Docling + onnxruntime），并按 compose 传的 `BASE_IMAGE` 打标签：`document-qa-assistant-worker:latest` 与 `document-qa-assistant-api:latest`。顺序：① 构建/拉取基础镜像并打上上述两个 tag；② `docker compose build`（叠加 pydantic 版本钉住 + `onnxruntime-gpu`）。`BASE_IMAGE` 是 compose 的 build ARG（构建期），不是运行时环境变量，别写进 `.env`。

**端口绑定（D40）**：api 端口绑在 `127.0.0.1:8000`（回环），只允许本机访问 —— 30+ 路由（含 DELETE/PATCH 删数据/改记忆）目前**无任何鉴权**，绑 `0.0.0.0` 会让同局域网/热点下任何人可读全部文档、删数据。需要手机或其他设备访问时，请先加 token 鉴权（台账 D40 方案 b）再放开绑定。

打开 **http://localhost:8000** 使用界面（与 API 同源，无 CORS 代码）；OpenAPI 在 http://localhost:8000/docs。

首次使用流程：左栏上传 PDF → Celery worker 依次 `parsing → chunking → embedding → publishing` → 建资料集（collection = 已发布快照上的文档视图）→ 提问。

权重与离线：Embedding（Qwen3-Embedding-0.6B）、Docling 版面/OCR、reranker（bge-reranker-v2-m3）首次需联网拉取；宿主机 HF 缓存已挂载进容器，暖缓存后可设 `HF_HUB_OFFLINE=1`。挂载路径已参数化（D49）：默认 `${HOME}/.cache/huggingface`，缓存不在默认位置时在 `.env` 设 `HF_CACHE_DIR`（Windows 用正斜杠，如 `HF_CACHE_DIR=D:/hf-cache`），不再写死用户名。reranker 默认走本地快照 `data/models/modelscope/models/BAAI--bge-reranker-v2-m3/snapshots/master`。

前端开发（可选）：`cd frontend && npm install && npm run dev` —— Vite dev proxy 把 `/agent` `/documents` `/indexes` `/tasks` `/health` 转发到 `:8000`。生产构建 `npm run build`（= `tsc --noEmit && vite build`）产出 `frontend/dist/`，由 FastAPI 托管，compose 已挂载该目录。

## 架构

### 入库（证据生产）

```text
PDF 上传
  -> DocumentRegistry（内容 SHA-256 幂等；UUID 落盘）
  -> Celery：parse_document / rebuild_index
  -> Docling StandardPdfPipeline
       table_mode=ACCURATE · formula/code enrichment · figures REFERENCED · scale=3.0
  -> rag_export -> OrderedRefiner（Role Gate / facts / latex / 路径）
  -> data/parsed/<doc_id>/ordered.json          # ordered_document_v1 真源
  -> OrderedChunker（chunker_version=ordered-aware-1）
       prose 320/512/64 · table_summary + table_pack · formula 原子 · figure caption
  -> EmbeddingGenerator（Qwen3-Embedding-0.6B；仅 retrieval_text）
  -> Chroma 版本目录 + current.json 原子发布
```

单文档任务会基于**全部已有 `ordered.json`** 重建全库快照，保证多文档索引一致。

### 提问（产品主路径 `/agent/chat` 与 `/agent/chat/stream`）

```text
AgentRuntime ReAct + 证据闸门（src/agent/runtime.py）
  -> RagAdapter.search（冻结契约 readmate-rag-adapter-1）
  -> RetrievalRouter 双轨：direct | planned（planned 才走 QueryPlanner）
  -> PersistentRetriever.retrieve_multi
       dense 多通道 + 可选 FTS5 词法 -> RRF 融合
  -> 宽池 candidate_k ->（默认开）bge-reranker-v2-m3 重排
  -> diversify（同页/同表折叠）+ B/C 表噪声压制 + dense top1 保底
  -> 表/图命中展开 prev/next 邻居进证据窗
  -> LLM 结构化作答 / 拒答 -> validate_citations 只保留本轮观察过的 chunk
  -> 后端组装真实引用（文件名、页码、snippet）
```

运行拓扑：

```text
Redis          -> Celery broker / result（仅容器网络，不对外暴露端口）
FastAPI (api)  -> 上传 / 状态 / 前端 SPA / /agent/* / /health   :8000
Celery worker  -> 解析 + 切块 + embed + 发布（--pool=solo，建议并发 1）
LLM API        -> .env 的 API_KEY / BASE_URL / MODEL（只收 Context）
```

单轮问答链路（`/chat/ask`、`src/rag/qa_service.py`、`src/generation/` 等）已于 2026-09-15 整条删除，产品只保留 Agent 主路径；旧解析栈（pypdf / PyMuPDF / PaddleOCR / document_ir / element_chunker / FAISS）已全部替换，不保留双轨实现。

## 关键目录

| 路径 | 作用 |
|------|------|
| `src/api/` | FastAPI：`app.py`（上传/任务/health/SPA 托管）+ `agent_routes.py`（`/agent/*`） |
| `src/tasks/` | Celery 应用与入库任务 |
| `src/ingestion/` | Docling 转换、ordered 精炼、`ordered_chunker` |
| `src/retrieval/` | embed、Chroma 读写、融合、rerank、去重、邻居展开 |
| `src/rag/` | 查询规划与中英术语表（供 Agent 双轨路由使用） |
| `src/agent/` | ReadMate ReAct runtime、工具、记忆、回答协议、资料集 |
| `src/adapter/` | 冻结的 RAG Adapter（`readmate-rag-adapter-1`）与双轨路由 |
| `src/llm/` | 模型注册表 + OpenAI 兼容客户端（含 `stream_complete`） |
| `src/storage/` `src/models/` | 文档注册表 / Agent DB 与 Pydantic 契约 |
| `src/evaluation/` | L1 检索探针与评测目标 |
| `frontend/` | React18 + Vite + TS + Tailwind + react-query + KaTeX 前端（`src/`、`dist/`） |
| `eval/` | Agent 分层评测 L0–L4（独立顶层包，产品不反向依赖） |
| `config/models.toml` | 模型注册表（密钥只从环境变量引用） |
| `test_data/` | 固定问题集与题库（评测用论文 PDF **未随仓库分发**，获取方式见 [`test_data/README.md`](test_data/README.md)） |
| `data/` | 运行产物：`uploads/`、`parsed/<id>/`、`chroma/versions/`+`current.json`、`models/`、`registry.json` |
| `tmp/` | 临时证据与评测输出（不入库、上线前清理） |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/documents` | 上传 PDF 并排队入库（SHA-256 幂等） |
| `GET` | `/documents` | 文档列表：状态 / 页数 / chunk 数 |
| `PUT` | `/documents/{document_id}` | 替换已注册文档 |
| `DELETE` | `/documents/{document_id}` | 删除文档 |
| `POST` | `/documents/{document_id}/retry` | 仅 `failed` 可重试 |
| `GET` | `/tasks/{task_id}` | 入库任务状态 |
| `POST` | `/agent/chat` | **产品主路径**：Agent 对话（ReAct + 证据闸门；评测用同步口径） |
| `POST` | `/agent/chat/stream` | 同一轮对话的 SSE 流式版（事件名：`tool_call` / `tool_result` / `answer_delta` / `answer_reset` / `gate_block` / `turn_end` / `error`） |
| `GET` | `/agent/collections` | 资料集列表 |
| `POST` | `/agent/collections` | 新建资料集 |
| `GET` | `/agent/collections/{collection_id}/documents` | 资料集内文档 |
| `POST` | `/agent/collections/{collection_id}/documents` | 向资料集添加文档 |
| `GET` | `/agent/sessions` | 会话列表 |
| `POST` | `/agent/sessions` | 新建会话 |
| `GET` | `/agent/sessions/{session_id}/messages` | 会话历史 |
| `POST` | `/agent/read` | 按 chunk_id 或 文档+页 展开片段（前端引用预览） |
| `GET` | `/agent/memory` | 读取当前记忆 |
| `GET` | `/agent/memory/candidates` | 待确认记忆候选 |
| `POST` | `/agent/memory/confirm` | 确认候选（须显式传 `candidate_ids`） |
| `POST` | `/agent/memory/reject` | 拒绝候选 |
| `GET` | `/health` | 索引是否加载、文档/chunk 计数、LLM 是否配置 |
| `POST` | `/indexes/rebuild` | 基于全部 `ordered.json` 全量重建索引 |

## 评测

| 层 | 测什么 | 入口 |
|----|--------|------|
| L1 | 检索质量：目标页排名 / recall@k / top-5 噪声（不调答案模型，秒级） | `python -m src.evaluation.retrieval_probe --variants baseline,routed --k 50` |
| L0–L4 | Agent 主路径：上游漂移预检 / 轨迹 / 答案 / 拒答三态 | `python -m eval.agent_eval`（见 `eval/README.md`） |

容器内跑评测（compose 已挂载 `data`/`src`/`config`，需补挂 `eval`/`test_data`/`tmp`）：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/eval:/app/eval" \
  -v "$PWD/test_data:/app/test_data" \
  -v "$PWD/tmp:/app/tmp" \
  api python -m eval.agent_eval \
    --questions test_data/evaluation_questions.jsonl,test_data/evaluation_questions_refusal.jsonl \
    --out tmp/baselines/agent_eval_$(date +%Y%m%d).jsonl
```

测试门禁（需绝对路径，本机简写 `-v tests` 会报 `mount path must be absolute`）：

```bash
docker compose run --rm --no-deps -e HF_HUB_OFFLINE=1 \
  -v "$PWD/tests:/app/tests" \
  -v "$PWD/eval:/app/eval" \
  -v "$PWD/test_data:/app/test_data" \
  -v "$PWD/pytest.ini:/app/pytest.ini" \
  api sh -lc 'pip install -q pytest; python -m pytest -q'
```

当前记录基线：容器内全量 **149 passed**（2026-09-16，含 SSE 批次）；L1 可复现基线 recall@5=1.0、MRR@5≈0.8451（正式索引 195 chunks / 3 文档）。

## 运维与已知边界

- **门禁前须批准**：本仓库规则要求测试 / 冒烟 / 真实 API 验收先经用户确认；`eval/` 也**只写 `tmp/` 或显式 `--out`，永不写 `data/`**（`outcomes.assert_writable_output()` 强制），且每次运行把 `data/app.db` 复制到副本（`AGENT_DB_PATH`），不污染产品库。
- **`data/chroma/.build.lock`** 全程持有：重建过程中被强杀会让后续重建恒报 `index_build_busy`，需手工删锁（台账 D1，未根治）。
- **Agent 路径无充分性门控**：`RETRIEVAL_MIN_SCORE` 在 retriever / adapter / agent 三处零引用，`search` 永远返回 top-k，闸门只看「有没有搜」；离题问题依赖模型自然语言拒答，不触发结构拒答（台账 A4，待产品决策）。
- **模型只收本轮 Context**：本机不做云端托管向量库，也不把原始 PDF 交给模型供应商。
- **Docker 镜像勿轻删**：`document-qa-assistant-api` / `-worker` 体积大但层完全共享，重建代价为「数十分钟 + 数 GB 下载」；磁盘吃紧优先 `docker builder prune -f`。
- **容器以 root 运行（D50，2026-09-21 更正）**：镜像里**已准备好**非 root 用户 `appuser`（uid/gid 1000）与 `/app/data` 的 `chown 1000:1000`，但 **`USER appuser` 被刻意注释掉、未启用** —— **实测启用即坏**：
  在 Docker Desktop for Windows 上，宿主 `./data` 绑定挂载后一律呈现为 `root:root`，于是
  `/app/data`（`drwxrwxrwx`）可写、但**已存在的 `app.db`（`-rw-r--r--`）不可写** → 以 uid 1000 跑第一次写库就 `sqlite3.OperationalError: attempt to write a readonly database`。
  本节此前写的「`sudo chown -R 1000:1000 ./data` 即可修」**在 Windows 上不成立**（D 盘没有可改的 POSIX 属主）。
  因此**当前工作配置就是 root**；要真正启用非 root，需要**入口脚本以 root 修属主后降权**（gosu/setpriv），属未做事项（实测：以非 root 运行会因宿主挂载权限导致数据库只读，故刻意保持 root）。
  缓解：服务端口已绑 `127.0.0.1`（D40），不对局域网暴露。
- **健康检查（D50）**：`api` 每 30s 打 `http://localhost:8000/health`；`worker` 每 30s ping Redis 连通性（**不用** `celery inspect ping`：`--pool=solo` 下长解析任务会占满唯一工作槽，ping 会假红并可能让容器在入库中途进入 `unhealthy`）。
- **本地 CI（D47）**：`bash scripts/ci.sh`（容器内 pytest → `npx tsc --noEmit` → `npm run build`，任一失败即非零退出）；`scripts/ci.sh pytest` / `frontend` 可只跑一半。pre-push hook 模板在 `scripts/pre-push`，**需手动安装**（仓库不会自动改你的 git 配置）：`cp scripts/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push`。
- 未提交：仓库目前只有 initial commit，`src/` `tests/` `frontend/` 等尚未提交（台账 F2，提交策略待定）。
- 本机存在并发写者风险（Claude Code / Codex 宿主同写仓库），改 `src/api/` 与方案文件前先约定唯一写入方。

## 明确不做（当前主链）

- 图片 VLM / ColPali 页级多向量
- 把 Docling HybridChunker 当主切刀
- 云端托管向量库、FAISS、自研 Paddle 表/版式栈
- LangGraph 作首版主链
- 多用户 / 鉴权 / 云端部署（本地单用户定位）

## 已知未完成

前端 F1/F2 已完成（B0 路由补丁、同步版前端、SSE 流式、会话侧栏），F3 打磨未开工；可视化点击冒烟与真 LLM 端到端验收欠账、以及全部待修缺陷编号与修法，记录在维护者本地的内部台账中（不在本仓库）。