// FE-1/FE-5: three-view shell (对话 / 资料管理 / 记忆管理) over the hash router,
// plus the small-screen behaviour: below md/lg the rail and the evidence column
// become dismissible overlays instead of disappearing (V5: <768px must not break).

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { useAppStore } from "./store";
import { useView, type View } from "./router";
import SourcesPanel from "./components/SourcesPanel";
import ChatPanel from "./components/ChatPanel";
import EvidencePanel from "./components/EvidencePanel";
import PreviewModal from "./components/PreviewModal";
import LibraryView from "./components/views/LibraryView";
import MemoryView from "./components/views/MemoryView";

const TABS: { key: View; label: string }[] = [
  { key: "chat", label: "对话" },
  { key: "library", label: "资料管理" },
  { key: "memory", label: "记忆管理" },
];

export default function App() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const collectionIds = useAppStore((s) => s.collectionIds);
  const [view, setView] = useView();
  const [pane, setPane] = useState<"none" | "rail" | "evidence">("none");

  // A pane opened on a narrow window must not survive into another view.
  useEffect(() => setPane("none"), [view]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-2">
        <span className="text-base font-semibold">ReadMate</span>
        <nav className="flex overflow-hidden rounded-md border border-gray-300 text-xs">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              className={`px-3 py-1 ${view === tab.key ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}
              onClick={() => setView(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <HealthDot state={health.data} error={health.isError} loading={health.isLoading} />
        {view === "chat" ? (
          <>
            <button
              className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50 md:hidden"
              onClick={() => setPane((current) => (current === "rail" ? "none" : "rail"))}
            >
              资料集
            </button>
            <button
              className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50 lg:hidden"
              onClick={() => setPane((current) => (current === "evidence" ? "none" : "evidence"))}
            >
              证据
            </button>
            <span className="hidden md:inline">
              {collectionIds.length ? <CollectionBadges ids={collectionIds} /> : <span className="text-xs text-gray-400">未选资料集</span>}
            </span>
          </>
        ) : null}
      </header>

      <div className="flex min-h-0 flex-1">
        {view === "chat" && (
          <>
            <aside className="hidden w-64 shrink-0 flex-col border-r border-gray-200 bg-white md:flex">
              <SourcesPanel />
            </aside>
            <main className="flex min-w-0 flex-1 flex-col">
              <ChatPanel />
            </main>
            <aside className="hidden w-80 shrink-0 border-l border-gray-200 bg-white lg:flex">
              <EvidencePanel />
            </aside>
          </>
        )}
        {view === "library" && (
          <main className="min-w-0 flex-1 bg-gray-50/40">
            <LibraryView />
          </main>
        )}
        {view === "memory" && (
          <main className="min-w-0 flex-1 bg-gray-50/40">
            <MemoryView />
          </main>
        )}
      </div>

      {/* FE-5: small-screen overlays.  The buttons above only exist below md/lg, so
          on a wide window these never open. */}
      {view === "chat" && pane !== "none" && (
        <div className="fixed inset-0 z-20 bg-black/20" onClick={() => setPane("none")} />
      )}
      {view === "chat" && pane === "rail" && (
        <div className="fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-gray-200 bg-white shadow-xl md:hidden">
          <button className="self-end px-2 py-1 text-xs text-gray-400 hover:text-gray-600" onClick={() => setPane("none")}>
            关闭 ×
          </button>
          <SourcesPanel />
        </div>
      )}
      {view === "chat" && pane === "evidence" && (
        <div className="fixed inset-y-0 right-0 z-30 flex w-80 flex-col border-l border-gray-200 bg-white shadow-xl lg:hidden">
          <button className="self-end px-2 py-1 text-xs text-gray-400 hover:text-gray-600" onClick={() => setPane("none")}>
            关闭 ×
          </button>
          <EvidencePanel />
        </div>
      )}

      <PreviewModal />
    </div>
  );
}

function HealthDot({ state, error, loading }: { state?: { index_loaded: boolean }; error: boolean; loading?: boolean }) {
  const cls = loading ? "bg-gray-300" : error ? "bg-gray-400" : state?.index_loaded ? "bg-green-500" : "bg-yellow-400";
  const tip = loading ? "连接中…" : error ? "API 不可达" : state?.index_loaded ? "索引就绪" : "索引未入库：提问将无证据";
  return (
    <span title={tip} className="flex items-center gap-1 text-xs text-gray-500">
      <span className={`inline-block h-2 w-2 rounded-full ${cls}`} />
      {tip}
    </span>
  );
}

function CollectionBadges({ ids }: { ids: string[] }) {
  const { data } = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const rows = (data?.collections ?? []).filter((item) => ids.includes(item.id));
  if (rows.length === 0) return null;
  return (
    <>
      {rows.map((row) => (
        <span key={row.id} className="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700">
          {row.name} · {row.document_count} 篇
        </span>
      ))}
    </>
  );
}
