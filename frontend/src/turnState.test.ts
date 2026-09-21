import { describe, expect, it } from "vitest";
import { applyTurnEvent, callsToTrace, extractStreamAnswer, initialTurnState, type TurnLiveState } from "./turnState";
import type { AgentChatResponse } from "./types";

/** Fold a list of [event, data] frames through the reducer, the way ChatPanel does. */
function run(frames: [string, any][], start: TurnLiveState = initialTurnState()): TurnLiveState {
  return frames.reduce((state, [event, data]) => applyTurnEvent(state, event, data), start);
}

describe("initialTurnState", () => {
  it("starts empty", () => {
    expect(initialTurnState()).toEqual({
      deltas: "",
      calls: [],
      gateBlocks: [],
      sessionId: null,
      mode: null,
      errorPayload: null,
      done: null,
    });
  });

  it("returns a fresh object each call (no shared mutable state)", () => {
    const first = initialTurnState();
    first.calls.push({ round: 1, call_index: 0, tool: "search", done: true });
    expect(initialTurnState().calls).toEqual([]);
  });
});

describe("applyTurnEvent", () => {
  it("binds session_id and mode on turn_start", () => {
    const state = run([["turn_start", { session_id: "s1", mode: "deep" }]]);
    expect(state.sessionId).toBe("s1");
    expect(state.mode).toBe("deep");
  });

  it("keeps the previous session_id when turn_start omits it", () => {
    const state = run([
      ["turn_start", { session_id: "s1", mode: "deep" }],
      ["turn_start", { mode: "chat" }],
    ]);
    expect(state.sessionId).toBe("s1");
    expect(state.mode).toBe("chat");
  });

  it("accumulates answer_delta text", () => {
    const state = run([
      ["answer_delta", { text: "你" }],
      ["answer_delta", { text: "好" }],
    ]);
    expect(state.deltas).toBe("你好");
  });

  it("treats a missing delta text as empty instead of 'undefined'", () => {
    expect(run([["answer_delta", {}]]).deltas).toBe("");
  });

  it("clears deltas on answer_reset (new llm round)", () => {
    const state = run([
      ["answer_delta", { text: "第一轮草稿" }],
      ["answer_reset", {}],
    ]);
    expect(state.deltas).toBe("");
  });

  it("appends a live call on tool_call", () => {
    const state = run([["tool_call", { round: 2, call_index: 1, tool: "search", arguments: { query: "q" } }]]);
    expect(state.calls).toEqual([{ round: 2, call_index: 1, tool: "search", arguments: { query: "q" }, done: false }]);
  });

  it("pairs tool_result with its call by (round, call_index) and marks it done", () => {
    const state = run([
      ["tool_call", { round: 1, call_index: 0, tool: "search" }],
      ["tool_call", { round: 1, call_index: 1, tool: "read" }],
      ["tool_result", { round: 1, call_index: 1, ok: true, hits: 3, duration_s: 0.5, route: { track: "retrieval" } }],
    ]);
    expect(state.calls[0].done).toBe(false);
    expect(state.calls[1]).toMatchObject({ done: true, ok: true, hits: 3, duration_s: 0.5, route: { track: "retrieval" } });
  });

  it("ignores a tool_result for an unknown call", () => {
    const state = run([
      ["tool_call", { round: 1, call_index: 0, tool: "search" }],
      ["tool_result", { round: 9, call_index: 9, ok: true }],
    ]);
    expect(state.calls).toHaveLength(1);
    expect(state.calls[0].done).toBe(false);
  });

  it("does not confuse calls from different rounds with the same call_index", () => {
    const state = run([
      ["tool_call", { round: 1, call_index: 0, tool: "search" }],
      ["tool_call", { round: 2, call_index: 0, tool: "read" }],
      ["tool_result", { round: 2, call_index: 0, ok: true }],
    ]);
    expect(state.calls[0]).toMatchObject({ tool: "search", done: false });
    expect(state.calls[1]).toMatchObject({ tool: "read", done: true });
  });

  it("collects gate blocks", () => {
    const state = run([
      ["gate_block", { block: "缺少证据" }],
      ["gate_block", {}],
    ]);
    expect(state.gateBlocks).toEqual(["缺少证据", ""]);
  });

  it("stores the final payload on turn_end and clears the live deltas", () => {
    const result = { session_id: "s1", answer: "答案", citations: [] } as unknown as AgentChatResponse;
    const state = run([
      ["answer_delta", { text: "流式草稿" }],
      ["turn_end", result],
    ]);
    expect(state.done).toBe(result);
    expect(state.deltas).toBe("");
  });

  it("records an error payload", () => {
    const state = run([["error", { code: "knowledge_base_empty", message: "索引为空" }]]);
    expect(state.errorPayload).toEqual({ code: "knowledge_base_empty", message: "索引为空" });
  });

  it("stringifies a missing error code/message", () => {
    expect(run([["error", {}]]).errorPayload).toEqual({ code: "", message: "" });
  });

  it("returns the same state object for an unknown event", () => {
    const start = initialTurnState();
    expect(applyTurnEvent(start, "something_new", { x: 1 })).toBe(start);
  });

  it("does not mutate the state it was given", () => {
    const start = initialTurnState();
    applyTurnEvent(start, "tool_call", { round: 1, call_index: 0, tool: "search" });
    applyTurnEvent(start, "answer_delta", { text: "x" });
    expect(start.calls).toEqual([]);
    expect(start.deltas).toBe("");
  });

  it("replays a whole turn into a coherent live state", () => {
    const state = run([
      ["turn_start", { session_id: "s7", mode: "deep" }],
      ["tool_call", { round: 1, call_index: 0, tool: "search", arguments: { query: "注意力" } }],
      ["tool_result", { round: 1, call_index: 0, ok: true, hits: 5, duration_s: 1.2 }],
      ["gate_block", { block: "证据不足，重试" }],
      ["answer_reset", {}],
      ["answer_delta", { text: '{"answer":"最终' }],
      ["answer_delta", { text: '答案"}' }],
      ["turn_end", { session_id: "s7", answer: "最终答案", citations: [] }],
    ]);
    expect(state.sessionId).toBe("s7");
    expect(state.mode).toBe("deep");
    expect(state.calls).toHaveLength(1);
    expect(state.calls[0]).toMatchObject({ tool: "search", ok: true, hits: 5, done: true });
    expect(state.gateBlocks).toEqual(["证据不足，重试"]);
    expect(state.done).toMatchObject({ answer: "最终答案" });
    expect(state.deltas).toBe("");
  });
});

describe("callsToTrace", () => {
  it("maps live calls to tool_trace-shaped rows", () => {
    const rows = callsToTrace([{ round: 2, call_index: 0, tool: "read", arguments: { chunk_id: "c1" }, ok: true, duration_s: 0.3, done: true }]);
    expect(rows).toEqual([{ tool: "read", arguments: { chunk_id: "c1" }, ok: true, error: null, route: null, round: 2, duration_s: 0.3 }]);
  });

  it("normalizes missing fields (TracePanel renders these before turn_end)", () => {
    const rows = callsToTrace([{ round: 1, call_index: 0, tool: "search", done: false }]);
    expect(rows[0]).toEqual({ tool: "search", arguments: {}, ok: false, error: null, route: null, round: 1, duration_s: undefined });
  });

  it("keeps a failure visible", () => {
    const rows = callsToTrace([{ round: 1, call_index: 0, tool: "read", ok: false, error: "not_found", done: true }]);
    expect(rows[0]).toMatchObject({ ok: false, error: "not_found" });
  });

  it("passes the route decision through", () => {
    const rows = callsToTrace([{ round: 1, call_index: 0, tool: "search", route: { track: "retrieval", provider: "chroma" }, done: true }]);
    expect(rows[0].route).toEqual({ track: "retrieval", provider: "chroma" });
  });

  it("returns [] for no calls", () => {
    expect(callsToTrace([])).toEqual([]);
  });
});

describe("extractStreamAnswer", () => {
  it("passes plain prose through unchanged", () => {
    expect(extractStreamAnswer("这是普通回答")).toBe("这是普通回答");
  });

  it("recovers the answer value from a partial envelope", () => {
    expect(extractStreamAnswer('{"answer":"已经到达的部分')).toBe("已经到达的部分");
  });

  it("does not leak the envelope keys while the answer is still pending", () => {
    // The bubble must keep its "thinking" state instead of showing JSON.
    expect(extractStreamAnswer('{"citations":[')).toBe("");
  });

  it("handles leading whitespace before the envelope", () => {
    expect(extractStreamAnswer('  \n {"answer":"ok"}')).toBe("ok");
  });

  it("stops at the closing quote of the answer value", () => {
    expect(extractStreamAnswer('{"answer":"end","citations":[]}')).toBe("end");
  });

  it("unescapes JSON escape sequences as they arrive", () => {
    expect(extractStreamAnswer('{"answer":"第一行\\n第二行"}')).toBe("第一行\n第二行");
    expect(extractStreamAnswer('{"answer":"带\\"引号\\"的答案"}')).toBe('带"引号"的答案');
    expect(extractStreamAnswer('{"answer":"反斜杠\\\\"}')).toBe("反斜杠\\");
  });

  it("decodes \\uXXXX escapes, including CJK", () => {
    expect(extractStreamAnswer('{"answer":"\\u4f60\\u597d"}')).toBe("你好");
  });

  it("drops a half-received escape sequence at the tail", () => {
    // The stream is mid-escape: better to lose the byte than to print the backslash.
    expect(extractStreamAnswer('{"answer":"abc\\')).toBe("abc");
    expect(extractStreamAnswer('{"answer":"abc\\u4f')).toBe("abc");
  });

  it("returns empty for text that starts with a brace but has no answer key", () => {
    // Documented behavior: `{`-leading text with no `"answer"` is treated as an
    // envelope whose answer has not arrived, so the caller keeps its "thinking"
    // state (extractStreamAnswer, turnState.ts:94-100).
    expect(extractStreamAnswer("{not json at all")).toBe("");
    expect(extractStreamAnswer('{"citations":[{"quote":"x"}]}')).toBe("");
  });

  it("returns an empty answer for an empty answer value", () => {
    expect(extractStreamAnswer('{"answer":""}')).toBe("");
  });
});
