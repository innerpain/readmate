import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError, SseError } from "../api";
import type { AgentChatResponse, MessageRow, RoundRecord, ToolTraceEntry } from "../types";
import { applyTurnEvent, extractStreamAnswer, initialTurnState, type TurnLiveState } from "../turnState";
import { useAppStore } from "../store";
import { compactNoteFor } from "../compactNote";
import MarkdownAnswer from "./MarkdownAnswer";
import CitationCards from "./CitationCards";
import AnswerMeta from "./AnswerMeta";
import ActionFlow from "./ActionFlow";

interface Turn {
  id: number;
  role: "user" | "assistant";
  text: string;
  result?: AgentChatResponse; // assistant turns only
  /** C4-a: restored from a persisted message (not produced live this session). */
  historical?: boolean;
}

function parsePayload(row: MessageRow): Record<string, unknown> | null {
  if (row.payload && typeof row.payload === "object") return row.payload;
  if (!row.payload_json) return null;
  try {
    const value = JSON.parse(row.payload_json);
    return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export function restoreTurn(row: MessageRow): Turn {
  if (row.role === "assistant") {
    const payload = parsePayload(row);
    // R1/R7-3 persist rounds + tool_trace_digest + gate + failure on every
    // assistant row (success AND failure), so a refreshed/switched session
    // rebuilds the real trajectory instead of an empty one.
    const citations = (Array.isArray(payload?.citations) ? payload!.citations : []) as AgentChatResponse["citations"];
    const warnings = (Array.isArray(payload?.warnings) ? payload!.warnings : []) as string[];
    const rounds = (Array.isArray(payload?.rounds) ? payload!.rounds : []) as RoundRecord[];
    const toolTrace = (Array.isArray(payload?.tool_trace_digest) ? payload!.tool_trace_digest : []) as ToolTraceEntry[];
    const gate = (payload?.gate && typeof payload.gate === "object" ? payload.gate : {}) as AgentChatResponse["gate"];
    const failure = typeof payload?.failure === "string" ? (payload!.failure as string) : null;
    return {
      id: row.id,
      role: "assistant",
      text: row.content,
      historical: true,
      result: {
        session_id: row.session_id, mode: "", answer: row.content, citations, non_source: [],
        warnings, tool_trace: toolTrace, gate, refused: failure != null, failure,
        rounds, dropped_citations: [],
      },
    };
  }
  return { id: row.id, role: "user", text: row.content };
}

/**
 * C3-a(c): fold freshly fetched server rows over the optimistic turns.  Match is
 * by (role, text) in order; a server row replaces the optimistic one (real DB id
 * for EvidencePanel grouping).  Anything on the optimistic side that the server
 * has not written yet is kept on the tail so a bubble never flickers away.
 */
function reconcileTurns(optimistic: Turn[], restored: Turn[]): Turn[] {
  if (restored.length === 0) return optimistic; // never blank out a live turn
  const matched = new Set<number>();
  const merged: Turn[] = restored.map((server) => {
    const index = optimistic.findIndex((local, position) => !matched.has(position) && local.role === server.role && local.text === server.text);
    if (index >= 0) matched.add(index);
    return server;
  });
  for (let position = 0; position < optimistic.length; position += 1) {
    if (!matched.has(position)) merged.push(optimistic[position]);
  }
  return merged;
}

const TOTAL_BUDGET_MS = 180000; // SSE needs more wall time than sync: fine, deltas keep it honest

export default function ChatPanel() {
  const queryClient = useQueryClient();
  const { sessionId, collectionIds, mode, streaming, setSession, setStreaming, setLastResult, quotedText, setQuotedText } = useAppStore();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [live, setLive] = useState<TurnLiveState>(initialTurnState());
  // 批 D 阶段 2 (D32): the manual "compress this session" action.  Its note is kept
  // separate from `error` because "there was nothing to fold in" is not an error.
  const [compacting, setCompacting] = useState(false);
  const [compactNote, setCompactNote] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // history回填 when the session changes; the sidebar list (F3) switches via setSession.
  useEffect(() => {
    setError(null);
    // C3-a(a): mid-turn the new-session flow sets sessionId (:128-129) while the
    // optimistic user bubble is already on screen and the server has NOT yet
    // written it (runtime appends inside chatStream).  Refetching now would
    // overwrite `turns` with [] and erase the bubble — the exact "first message
    // vanishes" bug.  `setStreaming(true)` runs before `setSession`, and we read
    // it imperatively so it is not a dependency and does not retrigger the fetch.
    if (useAppStore.getState().streaming) return;
    if (!sessionId) {
      setTurns([]);
      return;
    }
    let cancelled = false;
    api.sessionMessages(sessionId).then((data) => {
      if (cancelled) return;
      const restored = data.messages.filter((row) => row.role === "user" || row.role === "assistant").map(restoreTurn);
      setTurns(restored);
      const newest = [...restored].reverse().find((turn) => turn.result);
      useAppStore.getState().setLastResult(newest?.result ?? null);
    }).catch(() => {
      if (!cancelled) setTurns([]);
    });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    if (!streaming) return;
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [streaming]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, streaming, live]);

  const canSend = draft.trim().length > 0 && !streaming;

  /**
   * 批 D 阶段 2 (D32): compress this session's older turns into a pointer summary now.
   *
   * The background job already does this once a window overflows; the button exists
   * for the user who wants it done before the next question (and for the session
   * that is *just* over the line).  Every outcome gets a sentence -- including the
   * non-errors, because a silent button reads as a broken one.
   */
  async function compact() {
    if (!sessionId || compacting) return;
    setCompacting(true);
    setCompactNote(null);
    try {
      setCompactNote(compactNoteFor(await api.compactSession(sessionId)));
    } catch (cause) {
      setCompactNote(cause instanceof ApiError ? `压缩失败：${describeError(cause) || cause.message}` : "压缩失败");
    } finally {
      setCompacting(false);
    }
  }

  function describeError(cause: unknown): string {
    const code =
      cause instanceof ApiError && cause.detail && typeof cause.detail === "object" && "code" in cause.detail
        ? String((cause.detail as { code?: unknown }).code)
        : cause instanceof SseError
          ? cause.code
          : "";
    // Adapter codes: src/adapter/contracts.py
    if (cause instanceof DOMException && cause.name === "AbortError") return "";
    if (code === "knowledge_base_empty") return "索引尚未入库：请先在左栏完成 PDF 入库后再提问";
    if (code === "embedding_incompatible") return "当前索引与嵌入模型不匹配：需重建索引";
    if (code === "collection_empty") return "本资料集还没有可用文档：请在左栏添加已入库的 PDF";
    return `请求失败（${cause instanceof Error ? cause.message : "未知错误"}），输入已保留`;
  }

  async function reconcile(sid: string, optimistic: Turn[]) {
    try {
      const data = await api.sessionMessages(sid);
      const restored = data.messages.filter((row) => row.role === "user" || row.role === "assistant").map(restoreTurn);
      const merged = reconcileTurns(optimistic, restored);
      // If the fetch raced the very last write and returned FEWER rows than we
      // already have on screen, keep the richer optimistic list (never flash out).
      setTurns((current) => (merged.length >= current.length ? merged : current));
    } catch {
      /* keep optimistic turns */
    }
  }

  async function send() {
    const message = draft.trim();
    if (!message || streaming) return;
    // FE-4 (引用): an answer quoted into the box is sent as a markdown blockquote, so
    // the model sees exactly what the user pointed at and the transcript stays honest
    // about where the follow-up came from.
    const outgoing = quotedText ? `> 引用上一条回答：${quotedText}\n\n${message}` : message;
    setDraft("");
    setQuotedText("");
    setError(null);
    setElapsed(0);
    setLive(initialTurnState());
    setStreaming(true);
    setTurns((prev) => [...prev, { id: Date.now(), role: "user", text: outgoing }]);
    const controller = new AbortController();
    abortRef.current = controller;
    const watchdog = window.setTimeout(() => controller.abort(), TOTAL_BUDGET_MS);
    let finalized = false;
    try {
      // Pre-create the session when none is bound.  C1 is fixed backend-side
      // (chat no longer upserts a collection by id), and the turn's scope now
      // rides on ``collection_ids`` (C-①), so a multi-collection turn simply
      // binds no single home collection.
      let activeSession = sessionId;
      if (!activeSession) {
        const home = collectionIds.length === 1 ? collectionIds[0] : null;
        activeSession = (await api.createSession(home, mode, controller.signal)).session_id;
        setSession(activeSession);
      }
      const finish = (result: AgentChatResponse) => {
        finalized = true;
        setSession(result.session_id);
        setLastResult(result); // evidence panel follows the newest turn
        setTurns((prev) => {
          const withAssistant = [...prev, { id: Date.now() + 1, role: "assistant" as const, text: result.answer, result }];
          // C3-a(c): reconcile optimistic ids with the server rows (real DB ids
          // for EvidencePanel grouping).  Fire-and-forget; on any miss the
          // optimistic tail stays, so nothing flickers.
          void reconcile(result.session_id, withAssistant);
          return withAssistant;
        });
        queryClient.invalidateQueries({ queryKey: ["sessions"] });
        queryClient.invalidateQueries({ queryKey: ["messages", result.session_id] });
      };
      try {
        for await (const frame of api.chatStream(outgoing, activeSession, collectionIds, mode, controller.signal)) {
          setLive((prev) => applyTurnEvent(prev, frame.event, frame.data));
          if (frame.event === "turn_end") finish(frame.data as AgentChatResponse);
        }
      } catch (cause) {
        const aborted = cause instanceof DOMException && cause.name === "AbortError";
        const streamUnsupported = cause instanceof ApiError && (cause.status === 404 || cause.status === 405 || cause.status === 500);
        if (aborted) throw cause;
        if (finalized) {
          // error mid-stream after turn_end — cosmetic; result is already shown
          return;
        }
        if (streamUnsupported || cause instanceof SseError || cause instanceof TypeError) {
          // V1 (plan §7): SSE broken/disconnected → auto-degrade to sync once.
          const result = await api.chat(outgoing, activeSession, collectionIds, mode, controller.signal);
          finish(result);
        } else {
          throw cause;
        }
      }
    } catch (cause) {
      const aborted = cause instanceof DOMException && cause.name === "AbortError";
      if (aborted) {
        setError(finalized ? null : "已中断显示；后台答案仍会保存，切换会话/刷新可见");
      } else {
        setError(describeError(cause));
        setDraft((current) => current || message); // restore text if box untouched
      }
    } finally {
      window.clearTimeout(watchdog);
      abortRef.current = null;
      setStreaming(false);
    }
  }

  const stepCount = useMemo(() => turns.filter((turn) => turn.role === "assistant").length, [turns]);
  // The streamed text is the raw model output: a JSON envelope whose `answer`
  // value is what belongs on screen (see extractStreamAnswer).
  const liveAnswer = extractStreamAnswer(live.deltas);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-3">
        {turns.length === 0 && !streaming && (
          <div className="mt-16 text-center text-sm text-gray-400">
            {collectionIds.length ? "选择或创建一个会话，向 ReadMate 提问你的资料。" : "在左栏勾选资料集，或直接通用提问。"}
          </div>
        )}
        {turns.map((turn) => (
          <div key={turn.id} className={turn.role === "user" ? "flex justify-end" : ""}>
            <div className={turn.role === "user" ? "max-w-[80%] rounded-lg bg-blue-600 px-3 py-2 text-sm text-white" : "max-w-[90%]"}>
              {turn.role === "user" ? (
                <span className="whitespace-pre-wrap">{turn.text}</span>
              ) : (
                <div className="rounded-lg border border-gray-200 bg-white px-3 py-2">
                  {turn.text ? <MarkdownAnswer text={turn.text} /> : <span className="text-sm text-gray-500">（本轮无答案文本）</span>}
                  {turn.result && <AnswerMeta result={turn.result} />}
                  {turn.result && <CitationCards citations={turn.result.citations} />}
                  <div className="mt-1 flex items-center gap-2 text-[11px] text-gray-400">
                    {turn.result && turn.result.rounds.length > 0 && (
                      <span>
                        {turn.result.rounds.length} 轮 · {turn.result.tool_trace.length} 次工具调用
                      </span>
                    )}
                    {/* FE-4 (引用): quote this answer into the input box. */}
                    {turn.text && (
                      <button
                        className="ml-auto text-blue-600 hover:underline disabled:text-gray-300"
                        disabled={streaming}
                        title="把这条回答引用到输入框，接着追问"
                        onClick={() => setQuotedText(turn.text.slice(0, 800))}
                      >
                        引用
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {streaming && (
          <div className="max-w-[90%]">
            <ActionFlow calls={live.calls} gateBlocks={live.gateBlocks} elapsed={elapsed} onAbort={() => abortRef.current?.abort()} />
            {liveAnswer && (
              <div className="mt-1 rounded-lg border border-dashed border-gray-300 bg-white/60 px-3 py-2 opacity-80">
                <MarkdownAnswer text={liveAnswer} />
              </div>
            )}
            {liveAnswer === "" && live.calls.length === 0 && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-500" />
                ReadMate 思考中… {elapsed}s（第 {stepCount + 1} 问，基线中位约 23s）
              </div>
            )}
          </div>
        )}
        {error && <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-gray-200 bg-white px-4 py-3">
        {quotedText && (
          <div className="mb-2 flex items-start gap-2 rounded border border-blue-200 bg-blue-50 px-2 py-1.5 text-[11px] text-blue-800">
            <span className="shrink-0">引用上一条回答：</span>
            <span className="min-w-0 flex-1 line-clamp-2">{quotedText}</span>
            <button className="shrink-0 text-blue-500 hover:text-blue-700" title="取消引用" onClick={() => setQuotedText("")}>
              ×
            </button>
          </div>
        )}
        <div className="mb-2 flex items-center gap-2">
          <div className="flex overflow-hidden rounded-md border border-gray-300 text-xs">
            <ModeButton active={mode === "deep"} label="精读" onClick={() => useAppStore.getState().setMode("deep")} />
            <ModeButton active={mode === "chat"} label="闲聊" onClick={() => useAppStore.getState().setMode("chat")} />
          </div>
          <span className="text-xs text-gray-400">{mode === "deep" ? "回答必须先经检索取证" : "相关问题仍会检索；纯闲聊可跳过"}</span>
          <button
            className="ml-auto rounded-md border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-50"
            onClick={compact}
            disabled={!sessionId || streaming || compacting}
            title="把更早的轮次折叠成要点摘要（文档名/页码/结论），之后的提问仍记得这些线索"
          >
            {compacting ? "压缩中…" : "压缩本会话"}
          </button>
          <button
            className="rounded-md border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
            onClick={() => setSession(null)}
            disabled={streaming}
          >
            新会话
          </button>
        </div>
        {compactNote && <div className="mb-2 text-[11px] text-gray-500">{compactNote}</div>}
        <div className="flex items-end gap-2">
          <textarea
            className="max-h-40 min-h-[42px] flex-1 resize-y rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
            placeholder={collectionIds.length ? "关于资料的问题，Enter 发送 / Shift+Enter 换行" : "在左栏勾选资料集后提问，或直接通用提问"}
            value={draft}
            maxLength={4000}
            disabled={streaming}
            rows={1}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (canSend) send();
              }
            }}
          />
          <button
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-300"
            disabled={!canSend}
            onClick={send}
          >
            发送
          </button>
        </div>
        <div className="mt-1 text-right text-[11px] text-gray-400">{draft.length}/4000</div>
      </div>
    </div>
  );
}

function ModeButton({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button className={`px-3 py-1 ${active ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`} onClick={onClick}>
      {label}
    </button>
  );
}
