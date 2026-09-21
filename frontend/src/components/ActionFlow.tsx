// FE-1: 动作流 —— 「模型在干什么」压成一行一句，点开才看细节。
//
// Replaces LiveTrace's chip grid (superseded, file removed): a chip said
// `search ✓ 1.2s`, which never answered the two questions the user actually has
// while waiting — *what did it search* and *what came back*.  Each row here reads
//
//     🔍 检索「3.2.1 的公式」        6 命中 · 1.2s   ▸
//     📖 精读 attention.pdf · p4     1 命中 · 0.6s   ▸
//
// and expanding a row shows the raw arguments + error.  Round grouping is kept as
// a leading `#n` marker so "what ran in parallel" is still readable.
//
// Data source is ``LiveCall`` (turnState.ts), NOT ``ToolTraceEntry``: only the live
// shape carries ``hits``, which is what makes the summary line worth reading.

import { useMemo, useState } from "react";
import type { LiveCall } from "../turnState";

const TOOL_META: Record<string, { icon: string; label: string }> = {
  search: { icon: "🔍", label: "检索" },
  read: { icon: "📖", label: "精读" },
  plan: { icon: "🧭", label: "规划" },
};

export default function ActionFlow({
  calls,
  gateBlocks,
  elapsed,
  onAbort,
}: {
  calls: LiveCall[];
  gateBlocks: string[];
  elapsed: number;
  onAbort: () => void;
}) {
  const [openKey, setOpenKey] = useState<string | null>(null);

  const rows = useMemo(
    () => calls.map((call, index) => ({ call, key: `${call.round}:${call.call_index}:${index}` })),
    [calls],
  );
  const rounds = calls.length ? Math.max(...calls.map((call) => call.round)) + 1 : 0;
  const running = calls.some((call) => !isSettled(call));

  return (
    <div className="rounded-lg border border-gray-200 bg-white/70 px-3 py-2 text-xs">
      <div className="mb-1.5 flex items-center gap-2 text-gray-500">
        <span className={`inline-block h-2 w-2 rounded-full ${running ? "animate-pulse bg-blue-500" : "bg-green-500"}`} aria-hidden />
        <span>{calls.length ? `${rounds} 轮 · ${calls.length} 个动作` : "已连接，等待首个动作…"}</span>
        <span className="ml-auto tabular-nums">{elapsed}s</span>
        <button className="rounded-full border border-gray-300 px-2 py-0.5 text-gray-500 hover:bg-gray-50" onClick={onAbort}>
          停止显示
        </button>
      </div>

      <ol className="space-y-1">
        {rows.map(({ call, key }, position) => {
          const meta = TOOL_META[call.tool] ?? { icon: "⚙", label: call.tool };
          const outcome = outcomeOf(call);
          const showRound = position === 0 || rows[position - 1].call.round !== call.round;
          const open = openKey === key;
          return (
            <li key={key} className="rounded border border-gray-100 bg-white">
              <button
                className="flex w-full items-center gap-2 px-2 py-1 text-left hover:bg-gray-50"
                onClick={() => setOpenKey(open ? null : key)}
                title={call.error ?? undefined}
              >
                <span className="w-5 shrink-0 font-mono text-[10px] text-gray-300">{showRound ? `#${call.round + 1}` : ""}</span>
                <span className="shrink-0" aria-hidden>{meta.icon}</span>
                <span className="shrink-0 text-gray-700">{meta.label}</span>
                <span className="min-w-0 flex-1 truncate text-gray-500">{summaryOf(call)}</span>
                {call.route?.track ? (
                  <span className="shrink-0 rounded bg-gray-100 px-1 text-[10px] text-gray-500" title={call.route.provider ? `provider=${call.route.provider}` : undefined}>
                    {call.route.track}
                  </span>
                ) : null}
                <span
                  className={`shrink-0 tabular-nums ${
                    outcome.tone === "bad" ? "text-red-600" : outcome.tone === "run" ? "text-blue-600" : "text-green-700"
                  }`}
                >
                  {outcome.text}
                </span>
                <span className="shrink-0 text-gray-300">{open ? "▴" : "▾"}</span>
              </button>
              {open && (
                <div className="border-t border-gray-100 px-2 py-1">
                  <div className="break-all text-[10px] text-gray-500">{formatArguments(call.arguments)}</div>
                  {call.error && <div className="mt-0.5 text-[10px] text-red-600">{call.error}</div>}
                  {call.hits != null && <div className="mt-0.5 text-[10px] text-gray-400">命中 {call.hits} 条</div>}
                </div>
              )}
            </li>
          );
        })}
      </ol>

      {gateBlocks.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {gateBlocks.map((block, index) => (
            <span key={`g${index}`} className="rounded-full border border-yellow-200 bg-yellow-50 px-2 py-0.5 text-yellow-700">
              闸门：{block}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function isSettled(call: LiveCall): boolean {
  return call.done || call.duration_s != null || call.error != null || call.ok === true || call.ok === false;
}

/** One line describing what the call asked for (not what it returned). */
function summaryOf(call: LiveCall): string {
  const args = call.arguments ?? {};
  if (call.tool === "search") {
    const query = asText(args.query ?? args.q ?? args.query_text);
    return query ? `「${truncate(query, 30)}」` : "查询";
  }
  if (call.tool === "read") {
    const target = asText(args.filename ?? args.document_id ?? args.chunk_id);
    const page = args.page != null ? `p${String(args.page)}` : "";
    return [target ? truncate(target, 24) : "", page].filter(Boolean).join(" · ") || "片段";
  }
  const keys = Object.keys(args);
  return keys.length ? truncate(JSON.stringify(args), 44) : "";
}

/** Right-hand status: 运行中 / 失败 / 命中数 + 耗时. */
function outcomeOf(call: LiveCall): { text: string; tone: "ok" | "bad" | "run" } {
  if (!isSettled(call)) return { text: "运行中…", tone: "run" };
  if (call.ok === false) return { text: truncate(call.error || "失败", 18), tone: "bad" };
  const parts: string[] = [];
  if (call.hits != null) parts.push(`${call.hits} 命中`);
  if (call.duration_s != null) parts.push(`${call.duration_s.toFixed(1)}s`);
  return { text: parts.join(" · ") || "完成", tone: "ok" };
}

function asText(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function truncate(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > limit ? `${flat.slice(0, limit)}…` : flat;
}

function formatArguments(args: LiveCall["arguments"]): string {
  if (!args) return "（无参数）";
  try {
    return JSON.stringify(args);
  } catch {
    return "";
  }
}
