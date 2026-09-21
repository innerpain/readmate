// SSE turn reducer: events → live turn state (plan §5/§7: pure, testable).

import type { AgentChatResponse, ToolTraceEntry } from "./types";

export interface LiveCall {
  round: number;
  call_index: number;
  tool: string;
  arguments?: Record<string, unknown>;
  ok?: boolean;
  error?: string | null;
  hits?: number | null;
  duration_s?: number;
  /** C3-b: the backend's ``RouteDecision.as_dict()`` is ``{track, provider, reason}``.
   *  ``mode``/``route`` are kept only because older snapshots/tests referenced them;
   *  they carry no runtime value (types.ts documents the same mismatch). */
  route?: { track?: string; provider?: string; reason?: string; mode?: string; route?: string } | null;
  done: boolean;
}

export interface TurnLiveState {
  /** optimistic answer text of the CURRENT llm round (reset between rounds) */
  deltas: string;
  calls: LiveCall[];
  gateBlocks: string[];
  sessionId: string | null;
  mode: string | null;
  errorPayload: { code: string; message: string } | null;
  done: AgentChatResponse | null;
}

export function initialTurnState(): TurnLiveState {
  return { deltas: "", calls: [], gateBlocks: [], sessionId: null, mode: null, errorPayload: null, done: null };
}

export function applyTurnEvent(state: TurnLiveState, event: string, data: any): TurnLiveState {
  switch (event) {
    case "turn_start":
      return { ...state, sessionId: data.session_id ?? state.sessionId, mode: data.mode ?? state.mode };
    case "answer_delta":
      return { ...state, deltas: state.deltas + String(data.text ?? "") };
    case "answer_reset":
      return { ...state, deltas: "" };
    case "tool_call":
      return {
        ...state,
        calls: [...state.calls, { round: data.round, call_index: data.call_index, tool: data.tool, arguments: data.arguments, done: false }],
      };
    case "tool_result":
      return {
        ...state,
        calls: state.calls.map((call) =>
          call.round === data.round && call.call_index === data.call_index
            ? { ...call, ok: data.ok, error: data.error, hits: data.hits, duration_s: data.duration_s, route: data.route, done: true }
            : call,
        ),
      };
    case "gate_block":
      return { ...state, gateBlocks: [...state.gateBlocks, String(data.block ?? "")] };
    case "turn_end":
      return { ...state, done: data as AgentChatResponse, deltas: "" };
    case "error":
      return { ...state, errorPayload: { code: String(data.code ?? ""), message: String(data.message ?? "") } };
    default:
      return state;
  }
}

/** live calls → tool_trace-shaped rows so TracePanel can render them pre-turn_end. */
export function callsToTrace(calls: LiveCall[]): ToolTraceEntry[] {
  return calls.map((call) => ({
    tool: call.tool,
    arguments: call.arguments ?? {},
    ok: Boolean(call.ok),
    error: call.error ?? null,
    route: call.route ?? null,
    round: call.round,
    duration_s: call.duration_s,
  }));
}

/** Recover the answer text from a partially streamed JSON envelope.
 *
 * The final round streams the raw wire format -- `{"answer": "...",
 * "citations": [...], "non_source": []}` -- so rendering `deltas` directly put
 * the envelope keys into the bubble while the turn was still in flight (and any
 * markdown inside the string, tables included, was replaced again at `turn_end`).
 * Show the recovered `answer` value instead.
 *
 * Non-envelope text (chat-mode prose) passes through unchanged.  An envelope whose
 * `answer` value has not arrived yet yields "" so the caller keeps its "thinking"
 * state instead of flashing JSON at the user.
 */
export function extractStreamAnswer(raw: string): string {
  const text = raw.trimStart();
  if (!text.startsWith("{")) return raw;
  const match = /"answer"\s*:\s*"/.exec(text);
  if (!match) return "";
  return unescapePartialJsonString(text.slice(match.index + match[0].length));
}

function unescapePartialJsonString(fragment: string): string {
  const escapes: Record<string, string> = {
    n: "\n",
    t: "\t",
    r: "\r",
    b: "\b",
    f: "\f",
    '"': '"',
    "\\": "\\",
    "/": "/",
  };
  let out = "";
  for (let i = 0; i < fragment.length; i += 1) {
    const ch = fragment[i];
    if (ch === "\\") {
      const next = fragment[i + 1];
      if (next === undefined) break; // the stream stopped inside an escape sequence
      if (next === "u") {
        const hex = fragment.slice(i + 2, i + 6);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) break; // incomplete \uXXXX at the tail
        out += String.fromCharCode(parseInt(hex, 16));
        i += 5;
        continue;
      }
      out += escapes[next] ?? next;
      i += 1;
      continue;
    }
    if (ch === '"') break; // end of the value
    out += ch;
  }
  return out;
}
