// FE-3: 记忆管理页 —— 看得见、改得动的长期记忆。
//
// 问题.md asked for "显示记忆文件，用户可自由编辑".  This project's memory is not a
// file: it lives in SQLite (``user_profile`` / ``user_preferences`` /
// ``learning_progress`` / ``memory_candidates``).  So the page shows those four in
// editable form, and offers a one-way **导出 md** for the "文件" mental model --
// deliberately no import (parse + conflict merge is a different feature, FE-D6).
//
// 条目 CRUD works on the confirmed rows of ``memory_candidates``: that IS the memory
// table, so editing an entry is a plain UPDATE rather than a second store.

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api";
import { useAppStore } from "../../store";
import MemoryPanel from "../MemoryPanel";

const KIND_LABEL: Record<string, string> = { profile: "画像", preference: "偏好", progress: "进度" };

export default function MemoryView() {
  const queryClient = useQueryClient();
  const collectionIds = useAppStore((s) => s.collectionIds);
  const [collectionId, setCollectionId] = useState<string | null>(collectionIds.length === 1 ? collectionIds[0] : null);

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  const memory = useQuery({ queryKey: ["memory", collectionId], queryFn: () => api.memory(collectionId) });
  const entries = useQuery({ queryKey: ["memoryEntries"], queryFn: () => api.memoryEntries("confirmed") });

  const profile = memory.data?.profile ?? {};
  const preferences = memory.data?.preferences ?? {};
  const progress = (memory.data?.progress ?? null) as Record<string, string> | null;

  // --- 档案 ---------------------------------------------------------------
  const [draft, setDraft] = useState({ identity: "", major: "", goal: "" });
  useEffect(() => {
    setDraft({
      identity: String(profile.identity ?? ""),
      major: String(profile.major ?? ""),
      goal: String(profile.goal ?? ""),
    });
  }, [profile.identity, profile.major, profile.goal]);

  const saveProfile = useMutation({
    mutationFn: () => api.updateProfile(draft),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memory"] }),
  });

  // --- 进度 ---------------------------------------------------------------
  const [progressDraft, setProgressDraft] = useState({ last_focus: "", open_questions: "", next_step: "" });
  useEffect(() => {
    setProgressDraft({
      last_focus: String(progress?.last_focus ?? ""),
      open_questions: String(progress?.open_questions ?? ""),
      next_step: String(progress?.next_step ?? ""),
    });
  }, [progress?.last_focus, progress?.open_questions, progress?.next_step]);

  const saveProgress = useMutation({
    mutationFn: () => api.updateProgress(collectionId as string, progressDraft),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memory"] }),
  });

  // --- 偏好 ---------------------------------------------------------------
  const [newPreference, setNewPreference] = useState({ key: "", value: "" });
  const savePreference = useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) => api.setPreference(key, value),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memory"] }),
  });
  const dropPreference = useMutation({
    mutationFn: (key: string) => api.deletePreference(key),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memory"] }),
  });

  // --- 条目 ---------------------------------------------------------------
  const [editingEntry, setEditingEntry] = useState<number | null>(null);
  const [entryDraft, setEntryDraft] = useState({ key: "", value: "" });
  const saveEntry = useMutation({
    mutationFn: ({ id, key, value }: { id: number; key: string; value: string }) => api.updateMemoryEntry(id, { key, value }),
    onSuccess: () => {
      setEditingEntry(null);
      queryClient.invalidateQueries({ queryKey: ["memoryEntries"] });
    },
  });
  const dropEntry = useMutation({
    mutationFn: (id: number) => api.deleteMemoryEntry(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["memoryEntries"] }),
  });

  const preferenceRows = Object.entries(preferences);
  const entryRows = entries.data?.entries ?? [];

  const markdown = useMemo(() => {
    const lines: string[] = ["# ReadMate 记忆导出", "", `导出时间：${new Date().toLocaleString()}`, ""];
    lines.push("## 档案", "", `- 身份：${profile.identity || "（空）"}`, `- 专业：${profile.major || "（空）"}`, `- 目标：${profile.goal || "（空）"}`, "");
    lines.push("## 偏好", "");
    lines.push(...(preferenceRows.length ? preferenceRows.map(([key, value]) => `- ${key}：${value}`) : ["（无）"]));
    lines.push("", `## 学习进度${collectionId ? `（${collections.data?.collections.find((row) => row.id === collectionId)?.name ?? collectionId}）` : ""}`, "");
    lines.push(
      ...(progress
        ? [`- 最近关注：${progress.last_focus || "（空）"}`, `- 未解问题：${progress.open_questions || "（空）"}`, `- 下一步：${progress.next_step || "（空）"}`]
        : ["（未选择资料集）"]),
    );
    lines.push("", "## 记忆条目", "");
    lines.push(...(entryRows.length ? entryRows.map((row) => `- [${KIND_LABEL[row.kind] ?? row.kind}] ${row.key}：${row.value}`) : ["（无）"]));
    return lines.join("\n");
  }, [profile, preferenceRows, progress, entryRows, collectionId, collections.data]);

  const exportMarkdown = () => {
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `readmate-memory-${new Date().toISOString().slice(0, 10)}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-gray-200 bg-white px-4 py-2">
        <span className="text-sm font-medium">记忆管理</span>
        <span className="text-xs text-gray-400">模型长期记住的内容；候选经你确认才会写入，已确认的条目可直接编辑</span>
        <div className="ml-auto flex items-center gap-2 text-xs">
          <select
            className="rounded border border-gray-300 px-2 py-1 text-xs"
            value={collectionId ?? ""}
            onChange={(event) => setCollectionId(event.target.value || null)}
          >
            <option value="">学习进度范围：（不按资料集）</option>
            {(collections.data?.collections ?? []).map((row) => (
              <option key={row.id} value={row.id}>
                {row.name}
              </option>
            ))}
          </select>
          <button className="rounded-md border border-gray-300 px-3 py-1.5 text-gray-700 hover:bg-gray-50" onClick={exportMarkdown}>
            导出 md
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <div className="grid gap-3 lg:grid-cols-3">
          <Card title="档案" hint="模型对你的基本了解">
            <Field label="身份" value={draft.identity} onChange={(value) => setDraft((prev) => ({ ...prev, identity: value }))} />
            <Field label="专业" value={draft.major} onChange={(value) => setDraft((prev) => ({ ...prev, major: value }))} />
            <Field label="目标" value={draft.goal} onChange={(value) => setDraft((prev) => ({ ...prev, goal: value }))} />
            <SaveRow pending={saveProfile.isPending} onClick={() => saveProfile.mutate()} error={saveProfile.isError} />
          </Card>

          <Card title="偏好" hint={`${preferenceRows.length} 条`}>
            {preferenceRows.length === 0 && <Muted>还没有记录偏好。</Muted>}
            {preferenceRows.map(([key, value]) => (
              <PreferenceRow
                key={key}
                name={key}
                value={value}
                onSave={(next) => savePreference.mutate({ key, value: next })}
                onDelete={() => dropPreference.mutate(key)}
              />
            ))}
            <div className="mt-2 flex items-end gap-1 border-t border-gray-100 pt-2">
              <label className="min-w-0 flex-1">
                <span className="block text-[10px] text-gray-400">新偏好</span>
                <input
                  className="w-full rounded border border-gray-300 px-1.5 py-1 text-[11px]"
                  value={newPreference.key}
                  placeholder="键，如 解释风格"
                  onChange={(event) => setNewPreference((prev) => ({ ...prev, key: event.target.value }))}
                />
              </label>
              <label className="min-w-0 flex-1">
                <span className="block text-[10px] text-gray-400">值</span>
                <input
                  className="w-full rounded border border-gray-300 px-1.5 py-1 text-[11px]"
                  value={newPreference.value}
                  placeholder="值，如 先给数据流"
                  onChange={(event) => setNewPreference((prev) => ({ ...prev, value: event.target.value }))}
                />
              </label>
              <button
                className="rounded bg-blue-600 px-2 py-1 text-[11px] text-white disabled:bg-gray-300"
                disabled={!newPreference.key.trim() || !newPreference.value.trim() || savePreference.isPending}
                onClick={() => {
                  savePreference.mutate({ key: newPreference.key.trim(), value: newPreference.value.trim() });
                  setNewPreference({ key: "", value: "" });
                }}
              >
                添加
              </button>
            </div>
          </Card>

          <Card title="学习进度" hint={collectionId ? "按所选资料集" : "未选资料集"}>
            {!collectionId ? (
              <Muted>选择资料集后可查看与编辑该集的进度。</Muted>
            ) : (
              <>
                <Field label="最近关注" value={progressDraft.last_focus} onChange={(value) => setProgressDraft((prev) => ({ ...prev, last_focus: value }))} />
                <Field label="未解问题" value={progressDraft.open_questions} onChange={(value) => setProgressDraft((prev) => ({ ...prev, open_questions: value }))} />
                <Field label="下一步" value={progressDraft.next_step} onChange={(value) => setProgressDraft((prev) => ({ ...prev, next_step: value }))} />
                <SaveRow pending={saveProgress.isPending} onClick={() => saveProgress.mutate()} error={saveProgress.isError} />
              </>
            )}
          </Card>
        </div>

        <div className="mt-4 rounded-lg border border-gray-200 bg-white p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-medium">已确认的记忆条目</span>
            <span className="text-[11px] text-gray-400">{entryRows.length} 条 · 来自候选确认，可直接编辑</span>
          </div>
          {entries.isLoading && <Muted>加载中…</Muted>}
          {!entries.isLoading && entryRows.length === 0 && <Muted>还没有已确认的记忆条目：在会话里让模型「记住…」，再到下方确认。</Muted>}
          <div className="space-y-2">
            {entryRows.map((row) => (
              <div key={row.id} className="rounded border border-gray-200 p-2 text-xs">
                {editingEntry === row.id ? (
                  <div className="flex flex-wrap items-end gap-1">
                    <input
                      className="w-40 rounded border border-gray-300 px-1.5 py-1 text-[11px]"
                      value={entryDraft.key}
                      onChange={(event) => setEntryDraft((prev) => ({ ...prev, key: event.target.value }))}
                    />
                    <input
                      className="min-w-0 flex-1 rounded border border-gray-300 px-1.5 py-1 text-[11px]"
                      value={entryDraft.value}
                      onChange={(event) => setEntryDraft((prev) => ({ ...prev, value: event.target.value }))}
                    />
                    <button
                      className="rounded bg-blue-600 px-2 py-1 text-[11px] text-white"
                      onClick={() => saveEntry.mutate({ id: row.id, key: entryDraft.key.trim(), value: entryDraft.value.trim() })}
                    >
                      保存
                    </button>
                    <button className="rounded px-2 py-1 text-[11px] text-gray-500 hover:bg-gray-100" onClick={() => setEditingEntry(null)}>
                      取消
                    </button>
                  </div>
                ) : (
                  <div className="flex items-start gap-2">
                    <span className="rounded bg-gray-100 px-1 py-0.5 text-[10px] text-gray-500">{KIND_LABEL[row.kind] ?? row.kind}</span>
                    <span className="text-gray-400">{row.key}</span>
                    <span className="min-w-0 flex-1 break-words text-gray-700">{row.value}</span>
                    <button
                      className="shrink-0 text-[11px] text-blue-600 hover:underline"
                      onClick={() => {
                        setEditingEntry(row.id);
                        setEntryDraft({ key: row.key, value: row.value });
                      }}
                    >
                      编辑
                    </button>
                    <button
                      className="shrink-0 text-[11px] text-gray-400 hover:text-red-500"
                      onClick={() => window.confirm("删除这条记忆？模型将不再记得它。") && dropEntry.mutate(row.id)}
                    >
                      删除
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-4 rounded-lg border border-gray-200 bg-white p-3">
          <div className="mb-2 text-sm font-medium">待确认候选</div>
          <MemoryPanel />
        </div>
      </div>
    </div>
  );
}

function Card({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-gray-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium">{title}</span>
        {hint && <span className="text-[11px] text-gray-400">{hint}</span>}
      </div>
      {children}
    </section>
  );
}

function Field({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="mb-1.5 flex items-center gap-2 text-xs">
      <span className="w-20 shrink-0 truncate text-gray-400" title={label}>
        {label}
      </span>
      <input
        className="min-w-0 flex-1 rounded border border-gray-300 px-1.5 py-1 text-[11px]"
        value={value}
        placeholder="（空）"
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function PreferenceRow({
  name,
  value,
  onSave,
  onDelete,
}: {
  name: string;
  value: string;
  onSave: (value: string) => void;
  onDelete: () => void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <div className="mb-1.5 flex items-center gap-2 text-xs">
      <span className="w-20 shrink-0 truncate text-gray-400" title={name}>
        {name}
      </span>
      <input
        className="min-w-0 flex-1 rounded border border-gray-300 px-1.5 py-1 text-[11px]"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <button
        className="shrink-0 text-[11px] text-blue-600 hover:underline disabled:text-gray-300"
        disabled={draft === value}
        onClick={() => onSave(draft)}
      >
        保存
      </button>
      <button className="shrink-0 text-[11px] text-gray-400 hover:text-red-500" onClick={onDelete}>
        删除
      </button>
    </div>
  );
}

function SaveRow({ pending, onClick, error }: { pending: boolean; onClick: () => void; error: boolean }) {
  return (
    <div className="mt-1 flex items-center gap-2">
      <button className="rounded bg-blue-600 px-2 py-1 text-[11px] text-white disabled:bg-gray-300" disabled={pending} onClick={onClick}>
        {pending ? "保存中…" : "保存"}
      </button>
      {error && <span className="text-[11px] text-red-600">保存失败</span>}
    </div>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return <div className="py-2 text-xs text-gray-400">{children}</div>;
}
