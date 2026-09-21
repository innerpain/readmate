import type { AgentCitation } from "../types";
import { useAppStore } from "../store";
import { pageLabel } from "../pageLabel";

// Plan §6 CitationCard: filename (fallback = doc_id prefix, gap #7) + p{page}
// + quote≤120 chars; click → chunk preview via POST /agent/read.
export default function CitationCards({ citations }: { citations: AgentCitation[] }) {
  const setPreview = useAppStore((s) => s.setPreview);
  const setEvidenceTab = useAppStore((s) => s.setEvidenceTab);
  if (citations.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {citations.map((citation, index) => {
        const label = citation.filename || citation.document_id.slice(0, 8) || "未知文档";
        return (
          <button
            key={`${citation.chunk_id}-${index}`}
            className="rounded border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-xs text-blue-700 hover:bg-blue-100"
            title={citation.quote.slice(0, 120)}
            onClick={() => {
              setEvidenceTab("citations");
              setPreview(citation.chunk_id);
            }}
          >
            [{index + 1}] {label}
            {citation.page != null ? ` · ${pageLabel(citation.page, citation.page_end)}` : ""}
          </button>
        );
      })}
    </div>
  );
}
