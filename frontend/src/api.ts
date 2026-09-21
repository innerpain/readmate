// Thin fetch layer over the same-origin API (dev: vite proxy → :8000).

import type {
  AgentChatResponse,
  CollectionRow,
  DocInfo,
  DocumentRecord,
  HealthInfo,
  MemoryCandidate,
  MessageRow,
  ReadResult,
  SessionRow,
} from "./types";
import { parseSseStream } from "./sse";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = await response.json();
    } catch {
      /* body not JSON */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

/** Error carrying the HTTP status and FastAPI `detail` payload (e.g. the
 *  409 {code: knowledge_base_empty} from /agent/chat). */
export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(`HTTP ${status}`);
  }
}

/** Thrown for an SSE `error` event (backend surfaced an AdapterError mid-stream). */
export class SseError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** C2-b payload shape from ``GET /agent/documents/{id}/figures`` (defined here,
 *  not in types.ts, so this batch adds nothing to the shared type module). */
export interface FiguresResponse {
  document_id: string;
  base: string;
  figures: { name: string; url: string }[];
}

/** ``GET /agent/memory`` (defined here, not in types.ts, so this batch adds
 *  nothing to the shared type module). */
export interface MemoryResponse {
  digest: string;
  profile: Record<string, string>;
  preferences: Record<string, string>;
  progress: Record<string, unknown> | null;
}

/** 批 D 阶段 2 (D32): the outcome of a manual session compaction.  ``reason`` is the
 *  honest one from the server — ``ok`` / ``nothing_to_summarize`` /
 *  ``generation_failed`` / ``empty_summary`` / ``breaker_open``. */
export interface CompactSessionResult {
  stored: boolean;
  reason: string;
  upto_message_id: number | null;
  chars?: number;
  failures?: number;
}

export const api = {
  health: () => request<HealthInfo>("/health"),

  listDocuments: () => request<{ documents: DocumentRecord[] }>("/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ document: DocumentRecord }>("/documents", { method: "POST", body: form });
  },
  retryDocument: (id: string) => request<{ document: DocumentRecord }>(`/documents/${id}/retry`, { method: "POST" }),
  deleteDocument: (id: string) => request<unknown>(`/documents/${id}`, { method: "DELETE" }),
  /** FE-2: rename (alias) and/or switch a document off.  Separate from the PUT
   *  route, which replaces the PDF and re-runs ingestion. */
  updateDocumentMeta: (id: string, patch: { alias?: string; enabled?: boolean }) =>
    request<{ document: DocumentRecord }>(`/documents/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  listCollections: () => request<{ collections: CollectionRow[] }>("/agent/collections"),
  /** FE-2: 分区改名. */
  renameCollection: (id: string, name: string) =>
    request<{ renamed: boolean; collection_id: string; name: string }>(`/agent/collections/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  /** FE-2: move one document out of one collection (membership only). */
  removeCollectionDocument: (collectionId: string, documentId: string) =>
    request<{ removed: boolean; document_count: number }>(
      `/agent/collections/${encodeURIComponent(collectionId)}/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" },
    ),
  createCollection: (name: string, document_ids: string[]) =>
    postJson<{ collection_id: string; document_count: number }>("/agent/collections", { name, document_ids }),
  /** FE-2: delete one collection (documents and index are untouched). */
  deleteCollection: (id: string) =>
    request<{ deleted: boolean; collection_id: string; detached_sessions: number }>(
      `/agent/collections/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  collectionDocuments: (id: string) => request<{ collection_id: string; documents: DocInfo[] }>(`/agent/collections/${id}/documents`),

  /** C2-b: the read-only figure listing for a document (plan §C-② / P3).  The
   *  backend returns a consumable ``base`` prefix and one ready-to-use ``url``
   *  per figure, so the frontend never has to reconstruct on-disk paths. */
  documentFigures: (id: string) =>
    request<FiguresResponse>(`/agent/documents/${encodeURIComponent(id)}/figures`),

  listSessions: (collection_id?: string | null, limit?: number, offset?: number) => {
    const params = new URLSearchParams();
    if (collection_id) params.set("collection_id", collection_id);
    if (limit != null) params.set("limit", String(limit));
    if (offset != null) params.set("offset", String(offset));
    const query = params.toString();
    return request<{ sessions: SessionRow[] }>(`/agent/sessions${query ? `?${query}` : ""}`);
  },
  createSession: (collection_id: string | null, mode: string, signal?: AbortSignal) =>
    postJson<{ session_id: string }>("/agent/sessions", { collection_id, mode }, signal),
  sessionMessages: (id: string) => request<{ session_id: string; messages: MessageRow[] }>(`/agent/sessions/${id}/messages`),
  /** C4: rename a session (``create_session``/``list_sessions`` always carried a
   *  title; only the write route was missing). */
  renameSession: (id: string, title: string) =>
    request<{ session_id: string; title: string }>(`/agent/sessions/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  /** C4: delete a session and its transcript.  Documents and the index are not
   *  touched -- the confirmation copy says so. */
  deleteSession: (id: string) =>
    request<{ deleted: boolean; session_id: string; deleted_messages: number }>(
      `/agent/sessions/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  /** 批 D 阶段 2 (D32): fold the older turns of this session into a pointer summary
   *  now, instead of waiting for the background job.  Synchronous — the caller gets
   *  the honest reason when there was nothing to compress (`nothing_to_summarize`)
   *  or when the model failed (`generation_failed`), so the UI can say which. */
  compactSession: (id: string) =>
    request<CompactSessionResult>(`/agent/sessions/${encodeURIComponent(id)}/compact`, {
      method: "POST",
    }),

  /**
   * C-① (D-5b): ``chat`` takes a list of collection ids.  A single-id turn also
   * sends the legacy ``collection_id`` so the backend can bind the session home;
   * multi-id turns send only ``collection_ids`` (message-scope, per plan §R7-1).
   */
  chat: (
    message: string,
    session_id: string | null,
    collection_ids: string[] | null,
    mode: string,
    signal?: AbortSignal,
  ) => {
    const ids = (collection_ids ?? []).filter(Boolean);
    return postJson<AgentChatResponse>(
      "/agent/chat",
      {
        message,
        session_id,
        collection_id: ids.length === 1 ? ids[0] : null,
        collection_ids: ids.length ? ids : null,
        mode,
      },
      signal,
    );
  },

  /** SSE turn (plan §5).  Yields every parsed event; throws SseError on an
   *  `error` frame, AbortError on signal.  Collection scope follows the same
   *  rule as ``chat`` (C-①). */
  chatStream: async function* (
    message: string,
    session_id: string | null,
    collection_ids: string[] | null,
    mode: string,
    signal?: AbortSignal,
  ): AsyncGenerator<{ event: string; data: any }> {
    const ids = (collection_ids ?? []).filter(Boolean);
    const response = await fetch("/agent/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id,
        collection_id: ids.length === 1 ? ids[0] : null,
        collection_ids: ids.length ? ids : null,
        mode,
      }),
      signal,
    });
    if (!response.ok) {
      let detail: unknown = null;
      try {
        detail = await response.json();
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(response.status, detail);
    }
    if (!response.body) throw new ApiError(200, "empty stream body");
    for await (const frame of parseSseStream(response.body)) {
      if (frame.event === "error") throw new SseError(String((frame.data as any).code ?? ""), String((frame.data as any).message ?? ""));
      yield frame;
    }
  },

  read: (chunk_id: string) => postJson<ReadResult>("/agent/read", { chunk_id }),

  /** FE-1: the memory overview page reads the same endpoint the right-hand
   *  "记忆" tab used; ``collection_id`` scopes only the learning-progress block. */
  memory: (collection_id?: string | null) =>
    request<MemoryResponse>(`/agent/memory${collection_id ? `?collection_id=${encodeURIComponent(collection_id)}` : ""}`),

  /** FE-3: the editable view of long-term memory (confirmed rows by default). */
  memoryEntries: (status = "confirmed") =>
    request<{ entries: MemoryCandidate[] }>(`/agent/memory/entries?status=${encodeURIComponent(status)}`),
  updateMemoryEntry: (id: number, patch: { key?: string; value?: string }) =>
    request<{ entry: MemoryCandidate }>(`/agent/memory/entries/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  deleteMemoryEntry: (id: number) =>
    request<{ deleted: number; entry_id: number }>(`/agent/memory/entries/${id}`, { method: "DELETE" }),

  /** FE-3: learner profile + preferences. */
  updateProfile: (patch: { identity?: string; major?: string; goal?: string }) =>
    request<{ profile: Record<string, string> }>("/agent/profile", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  setPreference: (key: string, value: string) =>
    request<{ preferences: Record<string, string> }>(`/agent/preferences/${encodeURIComponent(key)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    }),
  deletePreference: (key: string) =>
    request<{ deleted: number; key: string }>(`/agent/preferences/${encodeURIComponent(key)}`, { method: "DELETE" }),
  /** FE-3: learning progress is per collection (that is how memory writes it). */
  updateProgress: (collectionId: string, patch: { last_focus?: string; open_questions?: string; next_step?: string }) =>
    request<{ progress: Record<string, unknown> }>(`/agent/progress/${encodeURIComponent(collectionId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  listCandidates: (session_id: string) =>
    request<{ candidates: MemoryCandidate[] }>(`/agent/memory/candidates?session_id=${encodeURIComponent(session_id)}`),
  confirmCandidates: (session_id: string, candidate_ids: number[]) => postJson<{ confirmed: number }>("/agent/memory/confirm", { session_id, candidate_ids }),
  rejectCandidates: (session_id: string, candidate_ids: number[]) => postJson<{ rejected: number }>("/agent/memory/reject", { session_id, candidate_ids }),
};
