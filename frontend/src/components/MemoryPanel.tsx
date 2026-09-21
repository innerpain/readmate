import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useAppStore } from "../store";

const KIND_LABEL: Record<string, string> = { profile: "画像", preference: "偏好", progress: "进度" };

// Plan §6 MemoryPanel: per-candidate confirm/reject, plus an explicit-select
// batch action.  NO "confirm all" request — the backend treats omitted
// candidate_ids as confirm-everything (§9 gap #2), so we always send an explicit
// id list; 「全选」 only fills that list locally (2026-09-19: 用户要"只需一次确认").
export default function MemoryPanel() {
  const queryClient = useQueryClient();
  const sessionId = useAppStore((s) => s.sessionId);
  const [checked, setChecked] = useState<number[]>([]);
  const [failed, setFailed] = useState(false);

  const candidates = useQuery({
    queryKey: ["candidates", sessionId],
    queryFn: () => api.listCandidates(sessionId!),
    enabled: Boolean(sessionId),
    refetchInterval: 8000, // memory_note may arrive during a turn
  });

  async function decide(ids: number[], approve: boolean) {
    if (!sessionId || ids.length === 0) return;
    try {
      if (approve) await api.confirmCandidates(sessionId, ids);
      else await api.rejectCandidates(sessionId, ids);
      setFailed(false);
    } catch {
      setFailed(true);
      return;
    }
    setChecked((prev) => prev.filter((id) => !ids.includes(id)));
    queryClient.invalidateQueries({ queryKey: ["candidates", sessionId] });
  }

  if (!sessionId) return <div className="mt-8 text-center text-xs text-gray-400">先在会话中提问；模型提议的记忆会出现在这里。</div>;

  const rows = candidates.data?.candidates ?? [];
  return (
    <div className="text-xs">
      <div className="mb-2 flex items-center justify-between text-gray-500">
        <span>待确认记忆候选</span>
        <span className="flex items-center gap-2">
          <span>{rows.length} 条</span>
          {rows.length > 1 && (
            <button
              className="rounded border border-gray-300 px-1.5 py-0.5 text-[11px] text-gray-600 hover:bg-gray-50"
              onClick={() => setChecked(checked.length === rows.length ? [] : rows.map((row) => row.id))}
            >
              {checked.length === rows.length ? "取消全选" : "全选"}
            </button>
          )}
        </span>
      </div>
      {rows.length === 0 && <div className="px-1 py-4 text-center text-gray-400">暂无候选。你自己说「记住…」时模型会直接写入长期库（不在此列）；模型自己推断出的偏好会先问你一次，确认后才写。</div>}
      <div className="space-y-2">
        {rows.map((row) => (
          <div key={row.id} className="rounded border border-gray-200 bg-white p-2">
            <label className="flex items-start gap-2">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={checked.includes(row.id)}
                onChange={(event) => setChecked((prev) => (event.target.checked ? [...prev, row.id] : prev.filter((id) => id !== row.id)))}
              />
              <span className="min-w-0 flex-1">
                <span className="mr-1 rounded bg-gray-100 px-1 py-0.5 text-[10px] text-gray-500">{KIND_LABEL[row.kind] ?? row.kind}</span>
                <span className="text-gray-400">{row.key}</span>
                <div className="mt-0.5 break-words text-gray-700">{row.value}</div>
              </span>
            </label>
            <div className="mt-1 flex justify-end gap-2">
              <button className="rounded border border-gray-300 px-2 py-0.5 text-gray-500 hover:bg-gray-50" onClick={() => decide([row.id], false)}>
                拒绝
              </button>
              <button className="rounded bg-blue-600 px-2 py-0.5 text-white hover:bg-blue-700" onClick={() => decide([row.id], true)}>
                确认
              </button>
            </div>
          </div>
        ))}
      </div>
      {checked.length > 0 && (
        <div className="sticky bottom-0 mt-2 flex gap-2 bg-white pt-2">
          <button className="flex-1 rounded border border-gray-300 px-2 py-1 text-gray-600 hover:bg-gray-50" onClick={() => decide(checked, false)}>
            批量拒绝（{checked.length}）
          </button>
          <button className="flex-1 rounded bg-blue-600 px-2 py-1 text-white hover:bg-blue-700" onClick={() => decide(checked, true)}>
            批量确认（{checked.length}）
          </button>
        </div>
      )}
      {failed && <div className="mt-2 text-red-600">操作失败，请重试</div>}
    </div>
  );
}
