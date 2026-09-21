import { useState } from "react";
import type { AgentChatResponse } from "../types";

const WARNING_TEXT: Record<string, string> = {
  citations_dropped_unobserved: "模型引用了本轮未检索到的片段，已剔除",
  no_citation: "本条回答没有可用引用",
  unverified_no_search: "闲聊模式：未检索，内容未经资料验证",
  deep_mode_requires_evidence: "精读模式闸门拦截：需先取证",
  chat_mode_requires_search: "问题与资料相关但未检索，已要求补检索",
  max_steps_without_answer: "步数预算用尽，未能给出答案",
  no_evidence_after_gate: "多次取证未果，按闸门拒绝作答",
  adapter_unavailable: "检索服务不可用",
  answer_not_json: "模型输出未走回答协议",
  answer_recovered_partial: "模型输出格式不完整，已尽力恢复答案",
  // A12: the model itself declined (material does not answer the question).
  model_refused: "模型判断资料里没有答案，已明确拒答",
  // 第 8 条: the request asked for a different material than the session is locked
  // to; the session's own scope was used.  Shown in Chinese -- the raw code was
  // unreadable in the answer bar (user-reported, 2026-09-19).
  scope_overridden: "本轮请求的资料集与本会话锁定范围不同，已按本会话范围检索",
  memory_auto_written: "你明确要求记住 → 已直接写入长期记忆（无需确认）",
  unverified_nonmaterial: "精读模式：问题与资料无关，未取证作答",
};

function warnText(code: string): string {
  return WARNING_TEXT[code] ?? code;
}

// Plan §6: NonSourceBadge + WarningBar + GateChip (right panel / under answer).
export default function AnswerMeta({ result }: { result: AgentChatResponse }) {
  const [open, setOpen] = useState(false);
  const failed = Boolean(result.failure) || result.refused;
  return (
    <div className="mt-2 space-y-1.5">
      {failed && (
        <div className="rounded border border-gray-200 bg-gray-100 px-2 py-1 text-xs text-gray-600">
          证据不足，未能作答{result.failure ? `（${warnText(result.failure)}）` : ""}
        </div>
      )}
      {result.non_source.length > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-700">
          <button onClick={() => setOpen(!open)} className="font-medium">
            ⚠ 非资料补充（{result.non_source.length}）{open ? " ▴" : " ▾"}
          </button>
          {open && (
            <ul className="mt-1 list-disc pl-4">
              {result.non_source.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {result.warnings.length > 0 && (
        <div className="rounded border border-yellow-300 bg-yellow-50 px-2 py-1 text-xs text-yellow-800">
          {result.warnings.map((item) => warnText(item)).join("；")}
        </div>
      )}
      {result.dropped_citations.length > 0 && (
        <div className="text-xs text-gray-400">剔除的幻觉引用：{result.dropped_citations.join(", ")}</div>
      )}
    </div>
  );
}
