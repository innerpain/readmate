import type { CompactSessionResult } from "./api";

/**
 * 批 D 阶段 2 (D32): what to tell the user after a manual compaction.
 *
 * The server answers with a `reason` on *every* path -- including the ones that are
 * not errors, like "there was nothing older to fold in". Mapping each reason to its
 * own sentence is the difference between a button the user trusts and one that
 * reports "成功" while nothing happened. The failure reasons all say the same
 * important thing: the conversation was left untouched.
 */
export function compactNoteFor(result: CompactSessionResult): string {
  switch (result.reason) {
    case "ok":
      return result.chars
        ? `已压缩：旧轮次折叠为 ${result.chars} 字的要点摘要`
        : "已压缩";
    case "nothing_to_summarize":
      return "无需压缩：还没有更早的轮次需要折叠";
    case "empty_summary":
      return "压缩未生效：模型没有返回内容，会话保持原样";
    case "generation_failed":
      return "压缩失败：模型调用出错，会话保持原样";
    case "breaker_open":
      return "压缩已暂停：连续失败多次，稍后再试";
    default:
      return `压缩结果：${result.reason}`;
  }
}
