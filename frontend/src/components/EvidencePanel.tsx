import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { MessageRow } from "../types";
import { restoreTurn } from "./ChatPanel";
import TracePanel from "./TracePanel";
import MemoryPanel from "./MemoryPanel";
import { useAppStore } from "../store";
import { pageLabelZh } from "../pageLabel";

type RestoredTurn = ReturnType<typeof restoreTurn>;

// Right column: 引用 / 轨迹 / 记忆 tabs.  Citations stay tied to the newest
// turn (plan §3/§6); the trace tab is C-④: one collapsible group per
// persisted assistant turn, fold state kept in localStorage so refreshed or
// re-opened sessions come back as you left them.
export default function EvidencePanel() {
  const tab = useAppStore((s) => s.evidenceTab);
  const setTab = useAppStore((s) => s.setEvidenceTab);
  const lastResult = useAppStore((s) => s.lastResult);
  return (
    <div className="flex h-full w-full flex-col">
      <div className="flex border-b border-gray-200 text-xs">
        {(
          [
            ["citations", `引用${lastResult ? ` (${lastResult.citations.length})` : ""}`],
            ["trace", "轨迹"],
            ["memory", "记忆"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            className={`flex-1 border-b-2 px-2 py-2 ${tab === key ? "border-blue-600 font-medium text-blue-700" : "border-transparent text-gray-500 hover:bg-gray-50"}`}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {tab === "citations" &&
          (lastResult ? (
            lastResult.citations.length > 0 ? (
              <div className="space-y-2">
                <GateChip gate={lastResult.gate} />
                <CitationList />
              </div>
            ) : (
              <Empty>本轮回答没有引用（拒答或纯闲聊）。</Empty>
            )
          ) : (
            <Empty>提问后，这里显示该轮全部证据引用。</Empty>
          ))}
        {tab === "trace" && <TraceGroups />}
        {tab === "memory" && <MemoryPanel />}
      </div>
    </div>
  );
}

/**
 * C-④: trajectory grouped by conversation turn.  Same source of truth as the
 * chat history (``["messages", sessionId]`` — ChatPanel.finish() invalidates
 * this key after every turn), rows rebuilt via the exported ``restoreTurn`` so
 * persisted rounds/tool_trace_digest/gate render exactly like the live ones.
 */
function TraceGroups() {
  const sessionId = useAppStore((s) => s.sessionId);
  const streaming = useAppStore((s) => s.streaming);
  const lastResult = useAppStore((s) => s.lastResult);
  const evidenceTurnId = useAppStore((s) => s.evidenceTurnId);
  const setEvidenceTurn = useAppStore((s) => s.setEvidenceTurn);

  const messages = useQuery({
    queryKey: ["messages", sessionId],
    queryFn: () => api.sessionMessages(sessionId as string),
    enabled: Boolean(sessionId),
  });

  const turns = useMemo<RestoredTurn[]>(() => {
    const rows: MessageRow[] = messages.data?.messages ?? [];
    return rows.filter((row) => row.role === "user" || row.role === "assistant").map(restoreTurn);
  }, [messages.data]);

  // assistant turns with their question number and the question text
  const groups = useMemo(() => {
    const out: { turn: RestoredTurn; questionNo: number; question: string }[] = [];
    let questionNo = 0;
    let lastQuestion = "";
    for (const turn of turns) {
      if (turn.role === "user") {
        questionNo += 1;
        lastQuestion = turn.text;
      } else if (turn.result) {
        out.push({ turn, questionNo: questionNo || 1, question: lastQuestion });
      }
    }
    return out;
  }, [turns]);

  const newestId = groups.length ? groups[groups.length - 1].turn.id : null;

  // localStorage-backed fold map: { [messageId]: collapsed }.  Absent key =
  // default (newest group open, older ones closed).
  const [collapsedMap, setCollapsedMap] = useState<Record<string, boolean>>({});
  useEffect(() => {
    if (!sessionId) {
      setCollapsedMap({});
      return;
    }
    try {
      const raw = localStorage.getItem(`readmate.trace.collapsed.${sessionId}`);
      const parsed = raw ? JSON.parse(raw) : {};
      setCollapsedMap(parsed && typeof parsed === "object" ? parsed : {});
    } catch {
      setCollapsedMap({});
    }
  }, [sessionId]);

  const isCollapsed = (id: number) => collapsedMap[String(id)] ?? id !== newestId;
  const toggle = (id: number) => {
    const next = !isCollapsed(id);
    setCollapsedMap((prev) => {
      const updated = { ...prev, [String(id)]: next };
      if (sessionId) {
        try {
          localStorage.setItem(`readmate.trace.collapsed.${sessionId}`, JSON.stringify(updated));
        } catch {
          /* storage full / disabled: fold state is best-effort */
        }
      }
      return updated;
    });
    setEvidenceTurn(id);
  };

  /** FE-1 (R4): fold or unfold every 问 at once, persisted with the same key as a
   *  single toggle so a refresh keeps whatever the user last chose. */
  const persistCollapsed = (map: Record<string, boolean>) => {
    setCollapsedMap(map);
    if (!sessionId) return;
    try {
      localStorage.setItem(`readmate.trace.collapsed.${sessionId}`, JSON.stringify(map));
    } catch {
      /* storage full / disabled: fold state is best-effort */
    }
  };
  const setAllCollapsed = (collapsed: boolean) => {
    const next: Record<string, boolean> = {};
    for (const group of groups) next[String(group.turn.id)] = collapsed;
    persistCollapsed(next);
  };

  if (!sessionId) return <Empty>提问后，这里按对话轮显示工具调用与耗时。</Empty>;

  // The turn that just finished may not be in the query cache yet (finish()
  // invalidates asynchronously).  Show a live placeholder for lastResult when
  // it is not (yet) among the persisted rows, so the panel never blanks out.
  const liveAlreadyPersisted =
    lastResult != null && groups.some((group) => group.turn.text === lastResult.answer);
  const showLive = streaming && lastResult != null && !liveAlreadyPersisted;

  return (
    <div className="text-xs">
      {groups.length === 0 && !showLive && <Empty>本会话还没有已保存的回答轮次。</Empty>}
      {groups.length > 1 && (
        <div className="mb-1.5 flex items-center justify-end gap-2 text-[11px] text-gray-400">
          <button className="hover:text-blue-600" onClick={() => setAllCollapsed(false)}>
            全部展开
          </button>
          <span className="text-gray-200">|</span>
          <button className="hover:text-blue-600" onClick={() => setAllCollapsed(true)}>
            全部折叠
          </button>
        </div>
      )}
      {groups.map((group) => (
        <TurnGroup
          key={group.turn.id}
          turn={group.turn}
          questionNo={group.questionNo}
          question={group.question}
          collapsed={isCollapsed(group.turn.id)}
          anchored={evidenceTurnId === group.turn.id}
          onToggle={() => toggle(group.turn.id)}
        />
      ))}
      {showLive && (
        <div className="mb-1.5 rounded border border-dashed border-blue-200 bg-blue-50/40">
          <div className="flex items-center gap-2 px-2 py-1.5 text-gray-500">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-blue-400" />
            本轮（进行中）…
          </div>
          {lastResult && (
            <div className="border-t border-blue-100 px-2 py-1">
              <TracePanel result={lastResult} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TurnGroup({
  turn,
  questionNo,
  question,
  collapsed,
  anchored,
  onToggle,
}: {
  turn: RestoredTurn;
  questionNo: number;
  question: string;
  collapsed: boolean;
  anchored: boolean;
  onToggle: () => void;
}) {
  const result = turn.result!;
  const failed = result.failure != null;
  const preview = question.length > 24 ? `${question.slice(0, 24)}…` : question || "（无提问文本）";
  return (
    <div className={`mb-1.5 rounded border bg-white ${anchored ? "border-blue-300" : "border-gray-200"}`}>
      <button className="flex w-full items-center gap-2 px-2 py-1.5 text-left" onClick={onToggle}>
        <span className="shrink-0 font-mono text-gray-400">问{questionNo}</span>
        <span className="min-w-0 flex-1 truncate text-gray-700" title={question}>
          {preview}
        </span>
        {failed && <span className="shrink-0 rounded bg-red-50 px-1 py-0.5 text-[10px] text-red-600">失败轮</span>}
        <span className="shrink-0 text-gray-400">
          {result.rounds.length}轮 · {result.tool_trace.length}调用 {collapsed ? "▸" : "▾"}
        </span>
      </button>
      {!collapsed && (
        <div className="border-t border-gray-100 px-2 py-1.5">
          {turn.historical && (
            <div className="mb-1 text-[10px] text-gray-400">历史轨迹仅含轮次与调用，不含实时耗时（耗时为后端记录的轮级数据）。</div>
          )}
          <TracePanel result={result} />
        </div>
      )}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="mt-8 px-2 text-center text-xs text-gray-400">{children}</div>;
}

function GateChip({ gate }: { gate: Record<string, unknown> }) {
  const searched = Boolean(gate.searched);
  const blocks = Number(gate.blocks ?? 0);
  if (searched && blocks === 0) return <div className="rounded bg-green-50 px-2 py-1 text-xs text-green-700">✓ 已取证（search/read 成功）</div>;
  if (searched) return <div className="rounded bg-yellow-50 px-2 py-1 text-xs text-yellow-700">闸门拦截 {blocks} 次后取证</div>;
  return <div className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-500">本轮未取证</div>;
}

function CitationList() {
  const lastResult = useAppStore((s) => s.lastResult);
  const setPreview = useAppStore((s) => s.setPreview);
  if (!lastResult) return null;
  return (
    <div className="space-y-2">
      {lastResult.citations.map((citation, index) => {
        const label = citation.filename || citation.document_id.slice(0, 8) || "未知文档";
        return (
          <button
            key={`${citation.chunk_id}-${index}`}
            className="block w-full rounded border border-gray-200 bg-white p-2 text-left text-xs hover:border-blue-300"
            onClick={() => setPreview(citation.chunk_id)}
          >
            <div className="font-medium text-blue-700">
              [{index + 1}] {label}
              {citation.page != null ? ` · ${pageLabelZh(citation.page, citation.page_end)}` : ""}
            </div>
            {citation.quote && <div className="mt-1 line-clamp-3 text-gray-600">{citation.quote.slice(0, 160)}</div>}
          </button>
        );
      })}
    </div>
  );
}
