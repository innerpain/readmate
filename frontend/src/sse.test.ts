import { describe, expect, it } from "vitest";
import { parseSseStream, type SseEvent } from "./sse";

/** Feed the parser byte-by-byte slices, so frame boundaries land mid-line the way
 *  a real network stream delivers them. */
function streamOf(chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<SseEvent[]> {
  const events: SseEvent[] = [];
  for await (const event of parseSseStream(stream)) events.push(event);
  return events;
}

describe("parseSseStream", () => {
  it("parses an event name and a JSON payload", async () => {
    const events = await collect(streamOf(['event: answer_delta\ndata: {"text":"你好"}\n\n']));
    expect(events).toEqual([{ event: "answer_delta", data: { text: "你好" } }]);
  });

  it("parses several frames from one chunk", async () => {
    const events = await collect(
      streamOf(['event: turn_start\ndata: {"session_id":"s1","mode":"deep"}\n\nevent: answer_delta\ndata: {"text":"a"}\n\n']),
    );
    expect(events.map((event) => event.event)).toEqual(["turn_start", "answer_delta"]);
    expect(events[0].data).toEqual({ session_id: "s1", mode: "deep" });
  });

  it("reassembles a frame split across chunks", async () => {
    // The boundary lands inside the event name AND inside the JSON body.
    const events = await collect(streamOf(['event: ans', 'wer_delta\ndata: {"te', 'xt":"split"}\n', "\n"]));
    expect(events).toEqual([{ event: "answer_delta", data: { text: "split" } }]);
  });

  it("splits on the frame boundary, not on \\n\\n inside a JSON string", async () => {
    // A literal newline inside the payload is escaped by JSON.stringify, so the
    // parser must not see it; the \\n\\n inside the *string* is two characters.
    const events = await collect(streamOf(['event: answer_delta\ndata: {"text":"a\\n\\nb"}\n\n']));
    expect(events).toEqual([{ event: "answer_delta", data: { text: "a\n\nb" } }]);
  });

  it("joins multiple data: lines of one frame", async () => {
    const events = await collect(streamOf(['event: turn_end\ndata: {"a":1,\ndata: "b":2}\n\n']));
    expect(events[0].data).toEqual({ a: 1, b: 2 });
  });

  it("defaults the event name to 'message' when no event: line is present", async () => {
    const events = await collect(streamOf(['data: {"x":1}\n\n']));
    expect(events).toEqual([{ event: "message", data: { x: 1 } }]);
  });

  it("skips heartbeat comments", async () => {
    const events = await collect(streamOf([": ping\n\n", 'event: answer_delta\ndata: {"text":"ok"}\n\n']));
    expect(events).toEqual([{ event: "answer_delta", data: { text: "ok" } }]);
  });

  it("skips frames with no data lines", async () => {
    const events = await collect(streamOf(["event: answer_delta\n\n", 'data: {"x":1}\n\n']));
    expect(events).toEqual([{ event: "message", data: { x: 1 } }]);
  });

  it("skips a malformed JSON frame instead of throwing", async () => {
    // turn_end stays authoritative, so a bad frame must not kill the turn.
    const events = await collect(streamOf(["event: answer_delta\ndata: {not json}\n\n", 'event: answer_delta\ndata: {"text":"after"}\n\n']));
    expect(events).toEqual([{ event: "answer_delta", data: { text: "after" } }]);
  });

  it("drops an unterminated trailing frame", async () => {
    // The stream ended mid-frame: nothing complete to emit.
    const events = await collect(streamOf(['event: answer_delta\ndata: {"text":"cut"}\n']));
    expect(events).toEqual([]);
  });

  it("decodes a multi-byte character split across chunks", async () => {
    // "好" is 3 UTF-8 bytes; the decoder must hold them together (stream: true).
    const encoder = new TextEncoder();
    const frame = encoder.encode('event: answer_delta\ndata: {"text":"好"}\n\n');
    const cut = frame.indexOf(0xe5) + 1; // just after the first byte of 好
    const events = await collect(streamOf([frame.slice(0, cut), frame.slice(cut)]));
    expect(events).toEqual([{ event: "answer_delta", data: { text: "好" } }]);
  });

  it("yields nothing for an empty stream", async () => {
    expect(await collect(streamOf([]))).toEqual([]);
  });
});
