// FE-2: 资料管理页 —— 文档 + 分区（资料集）的完整管理面。
//
// Everything 问题.md asked for lives here: 起别名 / 启用·取消 / 上传（多文件，带进度）
// / 选择分区 / 分区改名 / 资料移动位置.  Two rules shape the code:
//
//   * 别名只改显示（FE-D4）：索引里的文件名不动，所以引用卡仍显示原文件名——卡片上
//     因此同时标出原名，避免"别名没生效"的误会。
//   * 停用是软禁用（FE-D5）：文档留在库里、留在分区里，只是不再被检索；后端在
//     CollectionService.document_ids() 一处收口，前端不重复过滤。

import { useMemo, useRef, useState } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api";
import { useAppStore } from "../../store";
import { parseQualityNotice, type DocumentRecord, type QualityNotice } from "../../types";

const STAGES = ["queued", "parsing", "chunking", "embedding", "indexing", "completed"] as const;

const STAGE_LABEL: Record<string, string> = {
  queued: "排队",
  parsing: "解析",
  parsed: "解析",
  ocr: "OCR",
  structure: "结构化",
  chunking: "切块",
  embedding: "向量",
  indexing: "建索引",
  publishing: "发布",
  completed: "完成",
  failed: "失败",
};

export default function LibraryView() {
  const queryClient = useQueryClient();
  const focusCollectionId = useAppStore((s) => s.focusCollectionId);
  const setFocusCollection = useAppStore((s) => s.setFocusCollection);
  const setPreviewDoc = useAppStore((s) => s.setPreviewDoc);
  const [creating, setCreating] = useState(false);
  const [renamingDoc, setRenamingDoc] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });

  const documents = useQuery({
    queryKey: ["documents"],
    queryFn: api.listDocuments,
    refetchInterval: (query) => {
      const rows = query.state.data?.documents ?? [];
      return rows.some((doc) => doc.status === "queued" || doc.status === "processing") ? 3000 : false;
    },
  });

  const membership = useQueries({
    queries: (collections.data?.collections ?? []).map((row) => ({
      queryKey: ["collectionDocuments", row.id],
      queryFn: () => api.collectionDocuments(row.id),
    })),
  });

  // document_id -> the collections it belongs to (id + name), so a card can render
  // removable chips and a "move to…" picker without extra requests.
  const membershipByDoc = useMemo(() => {
    const map = new Map<string, { id: string; name: string }[]>();
    (collections.data?.collections ?? []).forEach((row, index) => {
      for (const doc of membership[index]?.data?.documents ?? []) {
        map.set(doc.document_id, [...(map.get(doc.document_id) ?? []), { id: row.id, name: row.name }]);
      }
    });
    return map;
  }, [collections.data, membership]);

  const invalidateAll = () => {
    queryClient.invalidateQueries({ queryKey: ["documents"] });
    queryClient.invalidateQueries({ queryKey: ["collectionDocuments"] });
  };

  const upload = useMutation({ mutationFn: (file: File) => api.uploadDocument(file), onSuccess: invalidateAll });
  const retry = useMutation({ mutationFn: (id: string) => api.retryDocument(id), onSuccess: invalidateAll });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteDocument(id),
    onSuccess: () => {
      invalidateAll();
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });

  // FE-2 mutations ---------------------------------------------------------
  const meta = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: { alias?: string; enabled?: boolean } }) =>
      api.updateDocumentMeta(id, patch),
    onSuccess: invalidateAll,
  });

  const renameCollection = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => api.renameCollection(id, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      invalidateAll();
    },
  });

  const removeCollection = useMutation({
    mutationFn: (id: string) => api.deleteCollection(id),
    onSuccess: (_data, id) => {
      if (focusCollectionId === id) setFocusCollection(null);
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      invalidateAll();
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  const move = useMutation({
    mutationFn: ({ collectionId, documentId }: { collectionId: string; documentId: string }) =>
      api.removeCollectionDocument(collectionId, documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      invalidateAll();
    },
  });

  // "移动到…" = 加进目标分区.  There is no dedicated add-one route: POST
  // /agent/collections upserts *by name*, so naming an existing collection adds the
  // document to it.  Kept in one named mutation so the trick is not repeated inline.
  const moveTo = useMutation({
    mutationFn: ({ name, documentId }: { name: string; documentId: string }) => api.createCollection(name, [documentId]),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      invalidateAll();
    },
  });

  const rows = useMemo(() => {
    const all = documents.data?.documents ?? [];
    if (!focusCollectionId) return all;
    const index = (collections.data?.collections ?? []).findIndex((row) => row.id === focusCollectionId);
    const inCollection = new Set((membership[index]?.data?.documents ?? []).map((doc) => doc.document_id));
    return all.filter((doc) => inCollection.has(doc.document_id));
  }, [documents.data, focusCollectionId, collections.data, membership]);

  const processing = rows.filter((doc) => doc.status === "queued" || doc.status === "processing").length;
  const failed = rows.filter((doc) => doc.status === "failed").length;
  const disabled = rows.filter((doc) => doc.enabled === false).length;
  // D8: 有解析降级/表格截断的篇数——只在真的 >0 时进标题行，正常库里不添噪。
  const qualityFlagged = rows.filter((doc) => parseQualityNotice(doc) !== null).length;
  const focused = (collections.data?.collections ?? []).find((row) => row.id === focusCollectionId) ?? null;

  const startRename = (doc: DocumentRecord) => {
    setRenamingDoc(doc.document_id);
    setRenameDraft(doc.alias || doc.original_filename);
  };

  const commitRename = (doc: DocumentRecord) => {
    const next = renameDraft.trim();
    setRenamingDoc(null);
    if (next === (doc.alias || doc.original_filename)) return;
    // An alias equal to the original filename is stored as "" so the two stay in sync
    // if the file is ever replaced.
    meta.mutate({ id: doc.document_id, patch: { alias: next === doc.original_filename ? "" : next } });
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-gray-200 bg-white px-4 py-2">
        <span className="text-sm font-medium">资料管理</span>
        <span className="text-xs text-gray-400">
          {rows.length} 篇{focusCollectionId ? "（已按分区筛选）" : ""}
          {processing ? ` · ${processing} 篇入库中` : ""}
          {failed ? ` · ${failed} 篇失败` : ""}
          {disabled ? ` · ${disabled} 篇已停用` : ""}
          {qualityFlagged ? ` · ${qualityFlagged} 篇解析有提示` : ""}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <button
            className="rounded-md border border-gray-300 px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50"
            onClick={() => setCreating(true)}
          >
            ＋ 新建分区
          </button>
          <button
            className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:bg-gray-300"
            disabled={upload.isPending}
            onClick={() => fileRef.current?.click()}
          >
            {upload.isPending ? "上传中…" : "上传 PDF"}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            className="hidden"
            onChange={(event) => {
              for (const file of event.target.files ?? []) upload.mutate(file);
              event.target.value = "";
            }}
          />
        </div>
      </div>

      {upload.isError && (
        <div className="border-b border-red-200 bg-red-50 px-4 py-1.5 text-xs text-red-700">上传失败：请确认是有效 PDF 且小于服务端限额。</div>
      )}

      <div className="flex flex-wrap items-center gap-1.5 border-b border-gray-200 bg-white px-4 py-2 text-xs">
        <span className="text-gray-400">分区：</span>
        <button
          className={`rounded-md px-2 py-1 ${focusCollectionId === null ? "bg-blue-600 text-white" : "border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"}`}
          onClick={() => setFocusCollection(null)}
        >
          全部
        </button>
        {(collections.data?.collections ?? []).map((row) => (
          <button
            key={row.id}
            title={`${row.document_count} 篇文档`}
            className={`rounded-md px-2 py-1 ${focusCollectionId === row.id ? "bg-blue-600 text-white" : "border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"}`}
            onClick={() => setFocusCollection(focusCollectionId === row.id ? null : row.id)}
          >
            {row.name} · {row.document_count}
          </button>
        ))}
        {(collections.data?.collections ?? []).length === 0 && (
          <span className="text-gray-400">还没有分区：先上传 PDF，再「新建分区」把文档归类。</span>
        )}

        {focused && (
          <span className="ml-auto flex items-center gap-2 text-[11px] text-gray-500">
            <span>当前分区：{focused.name}</span>
            <button
              className="rounded border border-gray-300 px-1.5 py-0.5 hover:bg-gray-50"
              onClick={() => {
                const typed = window.prompt("分区改名", focused.name);
                const name = (typed ?? "").trim();
                if (!name || name === focused.name) return;
                renameCollection.mutate({ id: focused.id, name });
              }}
            >
              改名
            </button>
            <button
              className="rounded border border-red-200 px-1.5 py-0.5 text-red-600 hover:bg-red-50"
              onClick={() => {
                if (!window.confirm(`删除分区「${focused.name}」？文档与索引不受影响，只解除归类。`)) return;
                removeCollection.mutate(focused.id);
              }}
            >
              删除分区
            </button>
          </span>
        )}
        {renameCollection.isError && <span className="text-[11px] text-red-600">改名失败：分区名称需唯一</span>}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {rows.length === 0 && (
          <div className="mt-10 text-center text-xs text-gray-400">
            {focusCollectionId ? "该分区内暂无文档。" : "暂无文档，点右上角「上传 PDF」开始。"}
          </div>
        )}
        <div className="space-y-2">
          {rows.map((doc) => {
            const inCollections = membershipByDoc.get(doc.document_id) ?? [];
            const isOff = doc.enabled === false;
            const label = doc.alias || doc.original_filename;
            const candidates = (collections.data?.collections ?? []).filter((row) => !inCollections.some((item) => item.id === row.id));
            return (
              <div key={doc.document_id} className={`rounded-lg border bg-white p-3 ${isOff ? "border-gray-200 opacity-70" : "border-gray-200"}`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    {renamingDoc === doc.document_id ? (
                      <div className="flex items-center gap-1">
                        <input
                          autoFocus
                          className="min-w-0 flex-1 rounded border border-blue-300 px-2 py-1 text-sm"
                          value={renameDraft}
                          maxLength={120}
                          onChange={(event) => setRenameDraft(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") commitRename(doc);
                            if (event.key === "Escape") setRenamingDoc(null);
                          }}
                        />
                        <button className="rounded bg-blue-600 px-2 py-1 text-xs text-white" onClick={() => commitRename(doc)}>
                          保存
                        </button>
                        <button className="rounded px-2 py-1 text-xs text-gray-500 hover:bg-gray-100" onClick={() => setRenamingDoc(null)}>
                          取消
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <span className="break-all text-sm font-medium leading-tight" title={doc.document_id}>
                          {label}
                        </span>
                        <button className="shrink-0 text-[11px] text-blue-600 hover:underline" onClick={() => startRename(doc)}>
                          改名
                        </button>
                      </div>
                    )}
                    {doc.alias && <div className="mt-0.5 truncate text-[11px] text-gray-400">原名：{doc.original_filename}</div>}
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-gray-400">
                      {doc.page_count ? <span>{doc.page_count} 页</span> : null}
                      {doc.chunk_count ? <span>{doc.chunk_count} chunks</span> : null}
                      {doc.size_bytes ? <span>{(doc.size_bytes / 1024 / 1024).toFixed(1)} MB</span> : null}
                      {doc.updated_at ? <span>· {doc.updated_at.slice(0, 16).replace("T", " ")}</span> : null}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    {isOff && <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[11px] text-amber-700">已停用</span>}
                    <StatusBadge status={doc.status} />
                  </div>
                </div>

                {(doc.status === "queued" || doc.status === "processing") && <StageBar stage={doc.stage} />}

                <QualityBadges doc={doc} />

                {doc.status === "failed" && (
                  <div className="mt-1.5 flex items-center gap-2 text-xs">
                    <span className="text-red-600">{doc.failure_code ?? "入库失败"}</span>
                    <button className="rounded border border-red-200 px-1.5 py-0.5 text-red-600 hover:bg-red-50" onClick={() => retry.mutate(doc.document_id)}>
                      重试
                    </button>
                  </div>
                )}

                {/* 分区成员：× 移出，select 加入（FE-2 资料移动位置） */}
                <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
                  <span className="text-gray-400">分区：</span>
                  {inCollections.length === 0 && <span className="text-gray-300">未归类</span>}
                  {inCollections.map((item) => (
                    <span key={item.id} className="flex items-center gap-1 rounded-full border border-gray-200 bg-gray-50 px-2 py-0.5 text-gray-600">
                      {item.name}
                      <button
                        className="text-gray-400 hover:text-red-500"
                        title="从该分区移出（文档与索引不受影响）"
                        onClick={() => move.mutate({ collectionId: item.id, documentId: doc.document_id })}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  {candidates.length > 0 && (
                    <select
                      className="rounded border border-gray-300 px-1 py-0.5 text-[11px] text-gray-600"
                      value=""
                      onChange={(event) => {
                        const collectionId = event.target.value;
                        if (!collectionId) return;
                        const name = candidates.find((row) => row.id === collectionId)?.name ?? "";
                        moveTo.mutate({ name, documentId: doc.document_id });
                      }}
                    >
                      <option value="">移动到…</option>
                      {candidates.map((row) => (
                        <option key={row.id} value={row.id}>
                          {row.name}
                        </option>
                      ))}
                    </select>
                  )}
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px]">
                  <DocChip
                    label="原文"
                    title={doc.status === "completed" ? "在新窗口打开 PDF 原文" : "入库完成后可查看"}
                    disabled={doc.status !== "completed"}
                    onClick={() => window.open(`/agent/documents/${encodeURIComponent(doc.document_id)}/file`, "_blank", "noopener")}
                  />
                  <DocChip
                    label="插图"
                    title={doc.status === "completed" ? "查看该文档的插图" : "入库完成后可查看"}
                    disabled={doc.status !== "completed"}
                    onClick={() => setPreviewDoc(doc.document_id, label)}
                  />
                  <button
                    className={`rounded-full border px-2 py-0.5 ${
                      isOff ? "border-green-200 text-green-700 hover:bg-green-50" : "border-amber-200 text-amber-700 hover:bg-amber-50"
                    }`}
                    title={isOff ? "重新参与检索" : "停用后不再参与检索（文件与索引保留）"}
                    onClick={() => meta.mutate({ id: doc.document_id, patch: { enabled: isOff } })}
                  >
                    {isOff ? "启用" : "停用"}
                  </button>
                  <button
                    className="ml-auto rounded-full border border-gray-200 px-2 py-0.5 text-gray-400 hover:border-red-200 hover:text-red-500"
                    onClick={() => window.confirm(`删除《${label}》并重建索引？`) && remove.mutate(doc.document_id)}
                  >
                    删除
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {creating && <CreateCollectionModal onClose={() => setCreating(false)} />}
    </div>
  );
}

function DocChip({ label, title, disabled, onClick }: { label: string; title: string; disabled: boolean; onClick: () => void }) {
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      className="rounded-full border border-gray-200 px-2 py-0.5 text-gray-600 hover:border-blue-300 hover:text-blue-600 disabled:cursor-not-allowed disabled:border-gray-100 disabled:text-gray-300"
    >
      {label}
    </button>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    completed: "bg-green-50 text-green-700",
    processing: "bg-blue-50 text-blue-700",
    queued: "bg-gray-100 text-gray-500",
    failed: "bg-red-50 text-red-700",
  };
  const label: Record<string, string> = { completed: "已入库", processing: "入库中", queued: "排队", failed: "失败" };
  return <span className={`shrink-0 rounded px-1.5 py-0.5 text-[11px] ${map[status] ?? "bg-gray-100 text-gray-500"}`}>{label[status] ?? status}</span>;
}

/** D8/D11: 解析质量徽标。判定规则在 `parseQualityNotice`（types.ts，纯函数、有测试），
 *  这里只负责把它画出来——`null` 就什么都不渲染，所以字段缺失的后端响应不会崩、
 *  也不会在正常文档上多出一个空壳。
 *
 *  黄标 = 后端明确 `degraded`（文案取 `user_notice`，这就是 D8 说的「降级提示到不了
 *  UI」的落点）；琥珀标 = 仅表格被截断（`table_packs_over_cap` / `tables_truncated`）。 */
function QualityBadges({ doc }: { doc: DocumentRecord }) {
  const notice: QualityNotice | null = parseQualityNotice(doc);
  if (!notice) return null;
  const tone =
    notice.tone === "yellow"
      ? "border-amber-300 bg-amber-100 text-amber-900"
      : "border-amber-200 bg-amber-50 text-amber-700";
  return (
    <div className={`mt-1.5 flex items-start gap-1.5 rounded border px-2 py-1 text-[11px] ${tone}`}>
      <span className="shrink-0" aria-hidden="true">
        ⚠
      </span>
      <span className="min-w-0 break-words">{notice.text}</span>
      {doc.quality?.parse_quality ? <span className="ml-auto shrink-0 opacity-70">质量：{doc.quality.parse_quality}</span> : null}
    </div>
  );
}

function StageBar({ stage }: { stage: string }) {
  const index = STAGES.indexOf(stage as (typeof STAGES)[number]);
  const current = index >= 0 ? index : 1;
  return (
    <div className="mt-1.5 flex items-center gap-1">
      {STAGES.slice(1, -1).map((item, position) => {
        const state = position + 1 < current ? "done" : position + 1 === current ? "active" : "todo";
        return (
          <span key={item} className="flex items-center gap-1">
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] ${
                state === "done" ? "bg-green-100 text-green-700" : state === "active" ? "animate-pulse bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-400"
              }`}
            >
              {STAGE_LABEL[item] ?? item}
            </span>
            {position < 2 && <span className="text-[8px] text-gray-300">▸</span>}
          </span>
        );
      })}
      {!STAGES.includes(stage as (typeof STAGES)[number]) && <span className="text-[10px] text-gray-400">{STAGE_LABEL[stage] ?? stage}</span>}
    </div>
  );
}

function CreateCollectionModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.listDocuments });
  const done = (documents.data?.documents ?? []).filter((doc) => doc.status === "completed");
  const create = useMutation({
    mutationFn: () => api.createCollection(name.trim(), picked),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      onClose();
    },
  });
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/30" onClick={onClose}>
      <div className="w-96 max-w-[90vw] rounded-lg bg-white p-4 shadow-xl" onClick={(event) => event.stopPropagation()}>
        <div className="mb-2 text-sm font-medium">新建分区</div>
        <input
          className="mb-3 w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
          placeholder="名称，如「Transformer 论文集」"
          maxLength={120}
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
        <div className="mb-1 text-xs text-gray-500">勾选已入库文档（{done.length} 可选）</div>
        <div className="max-h-48 overflow-y-auto rounded border border-gray-200">
          {done.length === 0 && <div className="p-2 text-xs text-gray-400">暂无 completed 文档</div>}
          {done.map((doc) => (
            <label key={doc.document_id} className="flex cursor-pointer items-center gap-2 px-2 py-1 text-xs hover:bg-gray-50">
              <input
                type="checkbox"
                checked={picked.includes(doc.document_id)}
                onChange={(event) =>
                  setPicked((prev) => (event.target.checked ? [...prev, doc.document_id] : prev.filter((id) => id !== doc.document_id)))
                }
              />
              <span className="truncate">{doc.alias || doc.original_filename}</span>
              <span className="ml-auto shrink-0 text-gray-400">{doc.page_count ? `${doc.page_count}p` : ""}</span>
            </label>
          ))}
        </div>
        {create.isError && <div className="mt-2 text-xs text-red-600">创建失败：名称需唯一且非空</div>}
        <div className="mt-3 flex justify-end gap-2">
          <button className="rounded px-3 py-1.5 text-sm text-gray-500 hover:bg-gray-100" onClick={onClose}>
            取消
          </button>
          <button
            className="rounded bg-blue-600 px-3 py-1.5 text-sm text-white disabled:bg-gray-300"
            disabled={!name.trim() || picked.length === 0 || create.isPending}
            onClick={() => create.mutate()}
          >
            创建
          </button>
        </div>
      </div>
    </div>
  );
}
