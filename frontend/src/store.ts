import type { AgentChatResponse } from "./types";

// Pure UI state (plan §4: backend responses are the only data source).

import { create } from "zustand";

export type Mode = "deep" | "chat";
export type EvidenceTab = "citations" | "trace" | "memory";

interface AppState {
  /** C-① (D-5b): the turn's scope is a *set* of collections, chosen in the
   *  chat panel and sent as ``collection_ids``.  Switching it never clears the
   *  active session — a conversation survives a scope change. */
  collectionIds: string[];
  /** Derived single-select convenience for legacy callers (SourcesPanel header,
   *  session filter): the lone collection when exactly one is picked, else null. */
  collectionId: string | null;
  sessionId: string | null;
  mode: Mode;
  streaming: boolean;
  evidenceTab: EvidenceTab;
  /** full response of the newest assistant turn; drives the evidence panel */
  lastResult: AgentChatResponse | null;
  /** chunk_id of the citation whose preview modal is open */
  previewChunkId: string | null;
  /** C2-b: document_id of the figure-gallery modal (mutually exclusive with chunk preview; see PreviewModal). */
  previewDocId: string | null;
  /** Optional human label (original filename) so the modal header reads naturally. */
  previewDocLabel: string | null;
  /** C4-b: which server message row the right column is anchored to
   *  (null = newest turn).  Grouping/persistence lives in EvidencePanel. */
  evidenceTurnId: number | null;
  setCollection: (id: string | null) => void;
  toggleCollection: (id: string) => void;
  /** Replace the picker wholesale -- used when reopening a session whose scope is
   * already locked (问题.md 第 8 条), where per-id toggling would be a no-op race. */
  setCollectionIds: (ids: string[]) => void;
  /** FE-D3: which collection the 资料管理 page opens focused on (null = 全部).
   *  Set by the left rail's 「去管理」 link, read by LibraryView's filter chips. */
  focusCollectionId: string | null;
  setFocusCollection: (id: string | null) => void;
  /** FE-4 (引用): an answer quoted into the input box — "针对这条继续问". */
  quotedText: string;
  setQuotedText: (text: string) => void;
  setSession: (id: string | null) => void;
  setMode: (mode: Mode) => void;
  setStreaming: (value: boolean) => void;
  setEvidenceTab: (tab: EvidenceTab) => void;
  setLastResult: (result: AgentChatResponse | null) => void;
  setPreview: (chunkId: string | null) => void;
  setPreviewDoc: (docId: string | null, label?: string | null) => void;
  setEvidenceTurn: (id: number | null) => void;
}

function derive(collectionIds: string[]): { collectionIds: string[]; collectionId: string | null } {
  return { collectionIds, collectionId: collectionIds.length === 1 ? collectionIds[0] : null };
}

export const useAppStore = create<AppState>((set) => ({
  collectionIds: [],
  collectionId: null,
  sessionId: null,
  mode: "deep", // Agent 册 §12 风险2：默认精读，一键切闲聊
  streaming: false,
  evidenceTab: "citations",
  lastResult: null,
  previewChunkId: null,
  previewDocId: null,
  previewDocLabel: null,
  evidenceTurnId: null,
  // C-①: selecting a collection must NOT drop the session (was `sessionId: null`).
  setCollection: (id) => set(derive(id ? [id] : [])),
  toggleCollection: (id) =>
    set((state) => {
      const picked = state.collectionIds.includes(id)
        ? state.collectionIds.filter((item) => item !== id)
        : [...state.collectionIds, id];
      return derive(picked);
    }),
  setCollectionIds: (ids) => set(derive(ids)),
  focusCollectionId: null,
  setFocusCollection: (id) => set({ focusCollectionId: id }),
  quotedText: "",
  setQuotedText: (text) => set({ quotedText: text }),
  setSession: (id) => set({ sessionId: id }),
  setMode: (mode) => set({ mode }),
  setStreaming: (streaming) => set({ streaming }),
  setEvidenceTab: (tab) => set({ evidenceTab: tab }),
  setLastResult: (result) => set({ lastResult: result }),
  setPreview: (chunkId) => set({ previewChunkId: chunkId }),
  // C2-b additive: opening a figure gallery closes any open chunk preview, and
  // vice-versa (the modal decides which one to render).  ``setPreview`` is
  // untouched so C-①'s callers are unaffected.
  setPreviewDoc: (docId, label = null) =>
    set(docId ? { previewDocId: docId, previewDocLabel: label, previewChunkId: null } : { previewDocId: null, previewDocLabel: null }),
  // C4-b additive: anchor the right column to one persisted turn (null = newest).
  setEvidenceTurn: (id: number | null) => set({ evidenceTurnId: id }),
}));
