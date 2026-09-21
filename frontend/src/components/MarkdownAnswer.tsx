import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// GFM tables + KaTeX (answers quote LaTeX formulas from papers).
export default function MarkdownAnswer({ text }: { text: string }) {
  return (
    <div className="md-answer text-sm">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
