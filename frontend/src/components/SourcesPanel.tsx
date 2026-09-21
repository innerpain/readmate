// FE-1: 左栏三段 —— 资料集分类（不列具体资料）/ 管理入口 / 会话。
//
// R2 asked for three things this file now does:
//   * 上方显示资料集分类，**不显示具体资料**（文档列表搬去 资料管理 页）；
//   * 点击资料集 = 展开该集的文档名（只读，仅用于确认"这个集里有什么"）
//     并给一个「去管理」链接跳到 资料管理 页按该集筛选（FE-D3）；
//   * 下方显示会话列表（改名/删除，C4）。
//
// The scope checkboxes stay here because 问题.md R1 puts the *selection* in the
// dialogue view — and 第二轮第 8 条 locks it after the first turn, so the whole
// block becomes read-only with a hint (batch 8 behaviour, unchanged).

import { useMemo, useState } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useAppStore } from "../store";
import { goToView } from "../router";
import { exportSessionMarkdown } from "../exportSession";

export default function SourcesPanel() {
  const queryClient = useQueryClient();
  const { collectionIds, toggleCollection, sessionId, setSession, setCollectionIds, streaming } = useAppStore();
  const setFocusCollection = useAppStore((s) => s.setFocusCollection);
  const [expanded, setExpanded] = useState<string[]>([]);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const collectionName = useMemo(() => {
    const map = new Map<string, string>();
    for (const row of collections.data?.collections ?? []) map.set(row.id, row.name);
    return map;
  }, [collections.data]);

  const [sessionLimit, setSessionLimit] = useState(12);
  const sessions = useQuery({
    // Keyed with the page size: "加载更多" widens the same list instead of adding a
    // second cache entry.  ChatPanel still invalidates ["sessions"] by prefix.
    queryKey: ["sessions", sessionLimit],
    queryFn: () => api.listSessions(null, sessionLimit),
    refetchInterval: 30000,
  });

  // 问题.md 第二轮第 8 条: the material scope is decided *before* the conversation
  // starts.  Once a session has locked one, the picker is read-only -- changing the
  // material means starting a new session (decided: 1a).
  const activeSession = useMemo(
    () => (sessions.data?.sessions ?? []).find((row) => row.id === sessionId) ?? null,
    [sessions.data, sessionId],
  );
  const scopeLocked = Boolean(activeSession?.scope_locked);
  const scopeLabel = useMemo(() => {
    const ids = activeSession?.scope_ids ?? [];
    if (!scopeLocked) return "";
    if (ids.length === 0) return "（全库）";
    return `（${ids.map((id) => collectionName.get(id) ?? id).join("、")}）`;
  }, [activeSession, scopeLocked, collectionName]);

  // Only expanded collections fetch their membership -- the rail must not pull the
  // whole document list just to render counts (it already has ``document_count``).
  const membership = useQueries({
    queries: expanded.map((id) => ({
      queryKey: ["collectionDocuments", id],
      queryFn: () => api.collectionDocuments(id),
      enabled: true,
    })),
  });
  const membershipById = useMemo(() => {
    const map = new Map<string, string[]>();
    expanded.forEach((id, index) => {
      map.set(id, (membership[index]?.data?.documents ?? []).map((doc) => doc.filename));
    });
    return map;
  }, [expanded, membership]);

  const renameSession = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.renameSession(id, title),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sessions"] }),
  });

  const removeSession = useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: (_report, id) => {
      if (id === sessionId) setSession(null);
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  const startRename = (row: { id: string; title: string }) => {
    const current = row.title || "";
    const typed = window.prompt("重命名会话", current);
    if (typed == null) return;
    const title = typed.trim();
    if (!title || title === current) return;
    renameSession.mutate({ id: row.id, title });
  };

  const confirmRemove = (row: { id: string; title: string }) => {
    const label = row.title || `会话 ${row.id.slice(4, 10)}`;
    if (!window.confirm(`删除「${label}」及其全部对话记录？文档与索引不受影响。`)) return;
    removeSession.mutate(row.id);
  };

  const openLibrary = (collectionId?: string) => {
    setFocusCollection(collectionId ?? null);
    goToView("library");
  };

  return (
    <div className="flex h-full flex-col text-sm">
      <div className="flex items-center justify-between border-b border-gray-200 px-3 py-2">
        <span className="font-medium">资料集</span>
        <button className="text-xs text-gray-400 hover:text-blue-600" onClick={() => openLibrary()} title="打开资料管理">
          资料管理 ›
        </button>
      </div>

      <div className="max-h-64 shrink-0 overflow-y-auto px-2 py-1">
        {collections.isLoading && <div className="px-2 py-1 text-xs text-gray-400">加载中…</div>}
        {(collections.data?.collections ?? []).length === 0 && (
          <div className="px-2 py-1 text-xs text-gray-400">还没有资料集：到「资料管理」上传 PDF 并新建资料集。</div>
        )}
        {(collections.data?.collections ?? []).map((row) => {
          const checked = collectionIds.includes(row.id);
          const isOpen = expanded.includes(row.id);
          const docs = membershipById.get(row.id);
          return (
            <div key={row.id} className={`mb-0.5 rounded ${checked ? "bg-blue-50" : ""}`}>
              <div className="flex items-center gap-1.5 px-1.5 py-1">
                <input
                  type="checkbox"
                  className="shrink-0"
                  checked={checked}
                  disabled={scopeLocked}
                  title={scopeLocked ? "本会话资料集已锁定，换集请新建会话" : "选入本会话的检索范围"}
                  onChange={() => toggleCollection(row.id)}
                />
                <button
                  className="min-w-0 flex-1 truncate text-left text-xs hover:text-blue-600"
                  title={isOpen ? "收起该集文档" : "展开该集文档（只读）"}
                  onClick={() => setExpanded((prev) => (isOpen ? prev.filter((id) => id !== row.id) : [...prev, row.id]))}
                >
                  <span className="mr-1 text-gray-300">{isOpen ? "▾" : "▸"}</span>
                  <span className={checked ? "text-blue-700" : "text-gray-700"}>{row.name}</span>
                </button>
                <span className="shrink-0 text-[10px] text-gray-400">{row.document_count} 篇</span>
              </div>
              {isOpen && (
                <div className="mb-1 ml-6 mr-1.5 rounded border border-gray-100 bg-white px-2 py-1">
                  {docs === undefined && <div className="text-[11px] text-gray-400">读取中…</div>}
                  {docs?.length === 0 && <div className="text-[11px] text-gray-400">该集暂无文档。</div>}
                  {(docs ?? []).map((filename, index) => (
                    <div key={`${row.id}-${index}`} className="truncate text-[11px] text-gray-500" title={filename}>
                      · {filename}
                    </div>
                  ))}
                  <button className="mt-1 text-[11px] text-blue-600 hover:underline" onClick={() => openLibrary(row.id)}>
                    去管理 ›
                  </button>
                </div>
              )}
            </div>
          );
        })}
        {scopeLocked && (
          <div className="mt-1 rounded border border-blue-200 bg-blue-50 px-2 py-1 text-[11px] text-blue-700">
            本会话资料集已锁定{scopeLabel}。要换资料集，请新建会话。
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-1 border-y border-gray-200 px-2 py-2 text-xs">
        <button className="rounded border border-gray-300 px-2 py-1 text-gray-600 hover:bg-gray-50" onClick={() => openLibrary()}>
          📚 资料管理
        </button>
        <button className="rounded border border-gray-300 px-2 py-1 text-gray-600 hover:bg-gray-50" onClick={() => goToView("memory")}>
          🧠 记忆管理
        </button>
      </div>

      <div className="flex items-center justify-between bg-gray-50/60 px-3 py-1.5 text-xs font-medium text-gray-600">
        <span>会话</span>
        <button
          className="text-blue-600 hover:underline disabled:text-gray-300"
          disabled={streaming}
          onClick={() => {
            if (streaming || !window.confirm("新建会话？当前会话仍保留在列表中")) return;
            setSession(null);
          }}
        >
          ＋ 新建
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-1">
        {(sessions.data?.sessions ?? []).map((row) => (
          <div key={row.id} className="mb-0.5 flex items-center gap-1">
            <button
              className={`flex min-w-0 flex-1 items-center gap-1 rounded px-2 py-1 text-left text-xs ${
                row.id === sessionId ? "bg-blue-50 text-blue-700" : "text-gray-600 hover:bg-gray-50"
              }`}
              disabled={streaming}
              onClick={() => {
                if (row.scope_locked) setCollectionIds(row.scope_ids ?? []);
                setSession(row.id);
              }}
            >
              <span className="min-w-0 flex-1 truncate">
                {row.title || `${row.mode === "deep" ? "精读" : "闲聊"}会话 ${row.id.slice(4, 10)}`}
              </span>
              {row.scope_locked && (row.scope_ids?.length ?? 0) > 1 && (
                <span className="shrink-0 rounded bg-gray-100 px-1 py-0.5 text-[10px] text-gray-500" title={row.scope_ids?.join("、")}>
                  {row.scope_ids?.length} 集
                </span>
              )}
              {row.scope_locked && (row.scope_ids?.length ?? 0) === 0 && (
                <span className="shrink-0 rounded bg-gray-100 px-1 py-0.5 text-[10px] text-gray-500">全库</span>
              )}
              {(row.scope_ids?.length ?? 0) <= 1 && row.collection_id && collectionName.get(row.collection_id) && (
                <span className="shrink-0 rounded bg-gray-100 px-1 py-0.5 text-[10px] text-gray-500">{collectionName.get(row.collection_id)}</span>
              )}
              <span className="shrink-0 text-[10px] text-gray-400">{row.updated_at.slice(5, 10)}</span>
            </button>
            {/* C4: session actions.  Kept outside the row button (nested buttons are
                invalid) and bounded to the two things a session owns. */}
            <button
              className="shrink-0 rounded px-1 py-0.5 text-[10px] text-gray-400 hover:bg-gray-100 hover:text-blue-600 disabled:text-gray-200"
              disabled={streaming}
              title="重命名会话"
              onClick={() => startRename(row)}
            >
              改名
            </button>
            <button
              className="shrink-0 rounded px-1 py-0.5 text-[10px] text-gray-400 hover:bg-gray-100 hover:text-blue-600 disabled:text-gray-200"
              disabled={streaming}
              title="导出该会话为 markdown"
              onClick={() => {
                void exportSessionMarkdown(row.id, row.title || `会话 ${row.id.slice(4, 10)}`);
              }}
            >
              导出
            </button>
            <button
              className="shrink-0 rounded px-1 py-0.5 text-[10px] text-gray-400 hover:bg-gray-100 hover:text-red-500 disabled:text-gray-200"
              disabled={streaming}
              title="删除该会话及其对话记录（文档与索引不受影响）"
              onClick={() => confirmRemove(row)}
            >
              删除
            </button>
          </div>
        ))}
        {(sessions.data?.sessions.length ?? 0) >= sessionLimit && (
          <button
            className="mt-1 w-full rounded border border-dashed border-gray-300 px-2 py-1 text-[11px] text-gray-500 hover:bg-gray-50"
            onClick={() => setSessionLimit((current) => current + 12)}
          >
            加载更多会话
          </button>
        )}
        {sessions.data?.sessions.length === 0 && <div className="px-2 py-1 text-xs text-gray-400">还没有历史会话。</div>}
      </div>
    </div>
  );
}
