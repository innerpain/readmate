// SSE parser: byte stream → (event name, payload) pairs (plan §5).
// Exported separately so the reducer is unit-testable in F3's Vitest batch.

export interface SseEvent {
  event: string;
  data: unknown;
}

export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    for (;;) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary < 0) break;
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (raw.startsWith(":")) continue; // heartbeat
      let name = "message";
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      try {
        yield { event: name, data: JSON.parse(dataLines.join("\n")) };
      } catch {
        /* malformed frame: skip, turn_end still authoritative */
      }
    }
  }
}
