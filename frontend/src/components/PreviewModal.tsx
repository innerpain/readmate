import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, api } from "../api";
import { useAppStore } from "../store";
import { pageLabelZh } from "../pageLabel";

/**
 * PreviewModal doubles as the app's single dialog.  Two modes:
 *   1) chunk   — a citation's underlying chunk (POST /agent/read).  Kept exactly
 *                as before so C-①'s callers are unchanged.
 *   2) figures — a document's extracted images (GET /agent/documents/{id}/figures,
 *                added in R7-2 / plan C-②).  The response already carries
 *                browser-ready ``url`` values (P3); we never reconstruct paths.
 *
 * Mode selection: if a chunk id is set, chunk wins; otherwise if a document id
 * is set, figures wins.  Both setters are cleared together on close so a stale
 * document state cannot reopen the modal after the user has dismissed it.
 */
export default function PreviewModal() {
  const chunkId = useAppStore((s) => s.previewChunkId);
  const docId = useAppStore((s) => s.previewDocId);
  const docLabel = useAppStore((s) => s.previewDocLabel);
  const setPreview = useAppStore((s) => s.setPreview);
  const setPreviewDoc = useAppStore((s) => s.setPreviewDoc);

  const close = () => {
    setPreview(null);
    setPreviewDoc(null, null);
  };

  const open = Boolean(chunkId || docId);

  // FE-5: Esc closes the app's single dialog (keyboard parity with the × button).
  // Declared before the early return so the hook order never changes.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/30 p-4" onClick={close}>
      <div
        className="max-h-[80vh] w-[560px] max-w-full overflow-y-auto rounded-lg bg-white p-4 shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        {chunkId ? <ChunkPreview chunkId={chunkId} onClose={close} /> : <FiguresPreview docId={docId!} label={docLabel} onClose={close} />}
      </div>
    </div>
  );
}

function ModalHeader({ title, onClose }: { title: React.ReactNode; onClose: () => void }) {
  return (
    <div className="mb-2 flex items-start justify-between">
      <div className="text-sm font-medium">{title}</div>
      <button className="text-gray-400 hover:text-gray-600" onClick={onClose}>
        ✕
      </button>
    </div>
  );
}

function ChunkPreview({ chunkId, onClose }: { chunkId: string; onClose: () => void }) {
  const query = useQuery({
    queryKey: ["read", chunkId],
    queryFn: () => api.read(chunkId),
  });
  const data = query.data;
  return (
    <>
      <ModalHeader
        title={
          <>
            原文片段
            {data?.page != null && <span className="ml-2 text-xs font-normal text-gray-500">{pageLabelZh(data.page, data.page_end)}</span>}
            {data?.chunk_type && <span className="ml-2 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500">{data.chunk_type}</span>}
          </>
        }
        onClose={onClose}
      />
      {query.isLoading && <div className="py-8 text-center text-xs text-gray-400">读取中…</div>}
      {query.isError && <div className="py-8 text-center text-xs text-red-600">片段读取失败（chunk 可能已随索引重建失效）</div>}
      {data && (
        <>
          {data.section && <div className="mb-1 text-xs text-gray-400">{data.section}</div>}
          <div className="whitespace-pre-wrap rounded bg-gray-50 p-3 text-xs leading-relaxed">{data.text || "（空片段）"}</div>
          {data.truncated && <div className="mt-1 text-[11px] text-amber-600">片段较长，已截断显示</div>}
        </>
      )}
    </>
  );
}

function FiguresPreview({ docId, label, onClose }: { docId: string; label: string | null; onClose: () => void }) {
  const query = useQuery({
    queryKey: ["figures", docId],
    queryFn: () => api.documentFigures(docId),
    staleTime: 60_000,
  });
  // Toggle the enlarged image; clicking the same one collapses it back.
  const [focused, setFocused] = useState<string | null>(null);

  const errorMessage = query.isError ? describeFigureError(query.error) : null;
  const figures = query.data?.figures ?? [];

  return (
    <>
      <ModalHeader
        title={
          <>
            插图 · {label || docId.slice(0, 8)}
            {query.data && <span className="ml-2 text-xs font-normal text-gray-500">{figures.length} 张</span>}
          </>
        }
        onClose={onClose}
      />
      {query.isLoading && <div className="py-8 text-center text-xs text-gray-400">加载插图列表…</div>}
      {errorMessage && (
        <div className="py-6 text-center text-xs text-red-600">
          {errorMessage}
          <div className="mt-1 text-[11px] text-gray-400">（后端返回码 <code>{errorCode(query.error)}</code>，可稍后重试或换文档）</div>
        </div>
      )}
      {query.data && figures.length === 0 && <div className="py-8 text-center text-xs text-gray-400">该文档暂无插图。</div>}
      {figures.length > 0 && (
        <div className={`grid gap-2 ${focused ? "grid-cols-1" : "grid-cols-2 sm:grid-cols-3"}`}>
          {focused ? (
            <div className="text-center">
              <img src={focused} alt="" className="mx-auto max-h-[60vh] w-auto max-w-full rounded border border-gray-200 bg-gray-50 object-contain" />
              <button className="mt-2 rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50" onClick={() => setFocused(null)}>
                ← 返回列表
              </button>
            </div>
          ) : (
            figures.map((item) => (
              <button
                key={item.url}
                type="button"
                className="group block rounded border border-gray-200 bg-gray-50 p-1 text-left hover:border-blue-300"
                title={item.name}
                onClick={() => setFocused(item.url)}
              >
                <img
                  src={item.url}
                  alt={item.name}
                  loading="lazy"
                  className="mx-auto h-24 w-full object-contain"
                  onError={(event) => {
                    (event.currentTarget as HTMLImageElement).style.opacity = "0.3";
                  }}
                />
                <div className="mt-1 truncate text-[10px] text-gray-500 group-hover:text-gray-700">{item.name}</div>
              </button>
            ))
          )}
        </div>
      )}
    </>
  );
}

function errorCode(cause: unknown): string {
  if (cause instanceof ApiError && cause.detail && typeof cause.detail === "object" && "code" in cause.detail) {
    return String((cause.detail as { code?: unknown }).code ?? "");
  }
  return "";
}

function describeFigureError(cause: unknown): string {
  const code = errorCode(cause);
  if (code === "document_not_found") return "文档不存在，或已被删除。";
  if (code === "figure_not_found") return "该插图文件已不存在（可能随索引重建失效）。";
  if (code === "unsupported_type" || code === "invalid_path") return "插图请求被后端安全策略拒绝。";
  if (cause instanceof Error) return `插图加载失败：${cause.message}`;
  return "插图加载失败。";
}
