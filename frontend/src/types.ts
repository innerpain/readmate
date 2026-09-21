// API types mirroring src/models/agent_schemas.py + src/models/schemas.py (F1).

export type IngestionStatus = "queued" | "processing" | "completed" | "failed";

// D8/D11: 解析质量摘要（后端 ``GET /documents/{id}`` 与
// ``/agent/collections/{id}/documents`` 的条目携带）。
//
// 后端从 ``data/parsed/<id>/quality_report.json`` + ``ordered.json`` 读出；两者缺失
// 时整个 ``quality`` 为 ``null``，所以前端**必须容错**：所有字段都是可选的，
// 渲染前先做 null 检查（见 LibraryView 的 QualityBadges）。旧快照/旧接口的响应里
// 这些键可能完全不存在，那不是错误。
export interface DroppedElements {
  total: number;
  by_role: Record<string, number>;
}

export interface ParseQuality {
  parse_quality: string;
  /** 超过单包 token 上限的表格分片数（后端 D15 之前恒为 0，不要当权威）。 */
  table_packs_over_cap: number;
  tables_truncated: boolean;
  /** 按 role 统计的被丢弃元素；后端算不出时该键为 null。 */
  dropped_elements: DroppedElements | null;
}

export interface DocumentRecord {
  document_id: string;
  original_filename: string;
  stored_filename: string;
  size_bytes?: number;
  page_count?: number | null;
  status: IngestionStatus;
  stage: string; // queued|parsing|parsed|chunking|embedding|indexing|publishing|completed|failed
  task_id?: string | null;
  failure_code?: string | null;
  chunk_count?: number;
  /** FE-2: display name; empty means "use original_filename". */
  alias?: string;
  /** FE-2: retrieval switch.  A disabled document stays in the library but is
   *  excluded from every search path (see CollectionService.document_ids). */
  enabled?: boolean;
  created_at?: string;
  updated_at?: string;
  /** D8: 解析降级（表格/公式未能可靠结构化）——true 时显示黄标 + user_notice。 */
  degraded?: boolean;
  /** D8: 给用户看的降级说明；后端未生成时为 null。 */
  user_notice?: string | null;
  /** D11: 解析质量摘要；无 quality_report.json 时为 null/缺失。 */
  quality?: ParseQuality | null;
}

export interface AgentCitation {
  chunk_id: string;
  document_id: string;
  filename: string;
  page: number | null;
  // A3: last page the cited passage covers; equal to `page` (or absent) when it
  // does not straddle a page break.
  page_end?: number | null;
  quote: string;
}

export interface AgentChatResponse {
  session_id: string;
  mode: string;
  answer: string;
  citations: AgentCitation[];
  non_source: string[];
  warnings: string[];
  tool_trace: ToolTraceEntry[];
  gate: { blocks?: number; searched?: boolean; read?: boolean };
  refused: boolean;
  failure: string | null;
  rounds: RoundRecord[];
  dropped_citations: string[];
}

export interface ToolTraceEntry {
  tool: string;
  /** Live trace: an object.  Historical trace (restored from
   *  ``payload.tool_trace_digest``): a JSON *string* capped by the backend
   *  digester.  Renderers must accept both (see TracePanel.tsx). */
  arguments: Record<string, unknown> | string;
  ok: boolean;
  error: string | null;
  /** C3-b: the backend ``RouteDecision.as_dict()`` (route_planner.py:121-122)
   *  emits ``{track, provider, reason}``; the old renderers read ``mode``/``route``
   *  which never matched, so the badge was always empty (two real bugs).  Renderers
   *  now read ``track``.  ``mode``/``route`` are kept ONLY because the frozen
   *  ``turnState.ts`` (pairing logic, off-limits this batch) still types
   *  ``LiveCall.route`` with them — they carry no runtime value. */
  route?: { track?: string; provider?: string; reason?: string; mode?: string; route?: string } | null;
  round?: number;
  offset_s?: number;
  duration_s?: number;
}

export interface RoundRecord {
  round: number;
  /** C3-b: backend ``_round_record`` emits ``tool_calls`` + ``tool_call_count``,
   *  not ``calls``.  The frontend previously declared ``calls`` but never read
   *  it, so both names are silently wrong until now. */
  tool_calls: string[];
  tool_call_count?: number;
  started_offset_s: number;
  duration_s: number;
  final_answer: boolean;
  gate_block?: string | null;
}

export interface CollectionRow {
  id: string;
  name: string;
  updated_at: string;
  document_count: number;
}

export interface SessionRow {
  id: string;
  title: string;
  mode: string;
  collection_id: string | null;
  created_at: string;
  updated_at: string;
  /** Collections locked on the first turn (问题.md 第 8 条); null while undecided. */
  scope_ids?: string[] | null;
  scope_locked?: boolean;
}

export interface MessageRow {
  id: number;
  session_id: string;
  role: string;
  content: string;
  tool_name: string | null;
  payload_json: string | null;
  /** C4-a: R7-3 decodes ``payload_json`` server-side into this object
   *  (rounds / tool_trace_digest / gate / citations / failure / observed chunks).
   *  Kept optional; callers fall back to parsing ``payload_json`` when absent. */
  payload?: Record<string, unknown> | null;
  created_at: string;
}

export interface MemoryCandidate {
  id: number;
  session_id: string;
  kind: string;
  key: string;
  value: string;
  status: string;
  created_at: string;
}

export interface DocInfo {
  document_id: string;
  filename: string;
  status: string;
  page_count: number | null;
  chunk_count: number;
  failure_code: string | null;
  /** D8/D11: same quality fields as DocumentRecord (this row type is what
   *  ``/agent/collections/{id}/documents`` returns). */
  degraded?: boolean;
  user_notice?: string | null;
  quality?: ParseQuality | null;
}

/** The quality-bearing subset both document row types share (D8/D11).
 *  ``DocumentRecord`` and ``DocInfo`` are structurally compatible with it, so
 *  renderers can take one argument shape instead of two. */
export interface QualityFields {
  degraded?: boolean;
  user_notice?: string | null;
  quality?: ParseQuality | null;
}

export interface QualityNotice {
  /** yellow = degraded（后端明确说了降级）; warn = 仅表格被截断. */
  tone: "yellow" | "warn";
  text: string;
}

/** D8/D11 display rule, kept out of the component so it is unit-testable and
 *  has exactly one implementation.
 *
 *  * ``degraded === true`` → yellow badge, ``user_notice`` as the copy (a
 *    degraded document with no notice still gets a generic line — a silent
 *    degradation is the very bug D8 exists to kill).
 *  * ``table_packs_over_cap > 0 || tables_truncated`` → 「有表格被截断」.
 *  * ``quality`` null/absent and ``degraded`` falsy → **no badge at all**.
 *
 *  Every field is read defensively: an older backend (or a snapshot written
 *  before D15) may omit ``quality`` entirely, and that must not throw.
 */
export function parseQualityNotice(doc: QualityFields | null | undefined): QualityNotice | null {
  if (!doc) return null;
  const quality = doc.quality ?? null;
  const overCap = typeof quality?.table_packs_over_cap === "number" && quality.table_packs_over_cap > 0;
  const truncated = quality?.tables_truncated === true;
  if (doc.degraded === true) {
    const notice = typeof doc.user_notice === "string" ? doc.user_notice.trim() : "";
    const fallback = overCap || truncated ? "解析降级，且部分表格被截断" : "解析降级：部分内容未能可靠结构化";
    return { tone: "yellow", text: notice || fallback };
  }
  if (overCap || truncated) {
    const detail: string[] = [];
    if (overCap && quality) detail.push(`${quality.table_packs_over_cap} 个表格分片超出上限`);
    if (truncated) detail.push("存在被截断的表格");
    return { tone: "warn", text: `有表格被截断（${detail.join("；")}）` };
  }
  return null;
}

export interface ReadResult {
  document_id: string;
  chunk_id: string | null;
  page: number | null;
  // A3: last page of the read; absent when it stays on one page.
  page_end?: number | null;
  section: string;
  text: string;
  chunk_type: string | null;
  truncated: boolean;
}

export interface HealthInfo {
  status: string;
  index_loaded: boolean;
  document_count: number;
  chunk_count: number;
  index_error: string | null;
  llm_configured: boolean;
}
