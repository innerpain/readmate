import { useState } from "react";
import type { AgentChatResponse } from "../types";

// Plan §6 TracePanel: per-round fold with calls, duration, gate blocks.
// Failed tool calls render red — this is the observation surface for the
// Q9 (document_not_found) / Q17 (repeated read) trajectory defects.
export default function TracePanel({ result }: { result: AgentChatResponse }) {
  const trace = result.tool_trace;
  const rounds = result.rounds;
  const byRound = new Map<number, typeof trace>();
  for (const entry of trace) {
    const key = entry.round ?? 0;
    byRound.set(key, [...(byRound.get(key) ?? []), entry]);
  }
  const total = rounds.reduce((sum, round) => sum + (round.duration_s ?? 0), 0);
  return (
    <div className="text-xs">
      <div className="mb-2 flex justify-between text-gray-500">
        <span>{rounds.length} 轮 / 预算内</span>
        <span>总计 {total.toFixed(1)}s</span>
      </div>
      {rounds.map((round) => (
        <RoundRow key={round.round} index={round.round} round={round} calls={byRound.get(round.round) ?? []} />
      ))}
      {rounds.length === 0 && <div className="text-gray-400">本轮无轨迹数据。</div>}
    </div>
  );
}

function RoundRow({
  index,
  round,
  calls,
}: {
  index: number;
  round: AgentChatResponse["rounds"][number];
  calls: AgentChatResponse["tool_trace"];
}) {
  const [open, setOpen] = useState(false);
  const failed = calls.filter((call) => !call.ok).length;
  return (
    <div className="mb-1.5 rounded border border-gray-200 bg-white">
      <button className="flex w-full items-center gap-2 px-2 py-1.5 text-left" onClick={() => setOpen(!open)}>
        <span className="font-mono text-gray-400">#{index}</span>
        <span className="text-gray-700">{round.final_answer ? "作答" : calls.map((call) => call.tool).join("+") || "闸门"}</span>
        {round.gate_block && <span className="rounded bg-yellow-50 px-1 py-0.5 text-[10px] text-yellow-700">{round.gate_block}</span>}
        {failed > 0 && <span className="rounded bg-red-50 px-1 py-0.5 text-[10px] text-red-600">{failed} 失败</span>}
        <span className="ml-auto text-gray-400">{(round.duration_s ?? 0).toFixed(1)}s {open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <div className="border-t border-gray-100 px-2 py-1">
          {calls.map((call, position) => (
            <div key={position} className="mb-1">
              <div className={call.ok ? "text-gray-600" : "text-red-600"}>
                {call.tool} {call.ok ? "✓" : `✗ ${call.error ?? ""}`}
                {call.route?.track ? (
                  <span className="ml-1 text-gray-400" title={call.route.provider ? `provider=${call.route.provider}` : undefined}>
                    [{call.route.track}]
                  </span>
                ) : null}
                {call.duration_s != null ? <span className="ml-1 text-gray-400">{call.duration_s.toFixed(1)}s</span> : null}
              </div>
              <div className="break-all pl-3 text-[10px] text-gray-400">{formatArguments(call.arguments)}</div>
            </div>
          ))}
          {calls.length === 0 && <div className="text-gray-400">（本轮无工具调用）</div>}
        </div>
      )}
    </div>
  );
}

/** Historical trace digests store arguments as a pre-serialized string; live
 *  turns carry an object.  Render whichever we get. */
function formatArguments(args: AgentChatResponse["tool_trace"][number]["arguments"]): string {
  if (typeof args === "string") return args;
  try {
    return JSON.stringify(args ?? {});
  } catch {
    return "";
  }
}
