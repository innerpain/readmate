import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy keeps the UI same-origin with the API container (frontend plan §1:
// zero CORS code — in production FastAPI serves dist/ from the same port).
//
// E5: the production build used to be a single ~660KB chunk, so every visit
// re-downloaded the markdown/KaTeX pipeline together with the app.
//
// Only that heavy pipeline is split off.  Splitting React (and the rest of the
// dependencies) away from the modules that initialise against them produced a
// chunk cycle that threw at load time — a blank page, caught by the CDP
// re-check — so everything else stays in rollup's own chunking.
const MARKDOWN_PREFIXES = [
  "react-markdown",
  "remark",
  "rehype",
  "katex",
  "micromark",
  "mdast",
  "hast",
  "unified",
  "unist",
  "vfile",
  "character-entities",
  "decode-named",
  "parse-entities",
  "property-information",
  "html-url-attributes",
  "style-to-object",
  "entities",
  "devlop",
  "bail",
  "trough",
  "zwitch",
  "longest-streak",
  "ccount",
  "markdown-table",
  "escape-string-regexp",
  "extend",
  "space-separated-tokens",
  "comma-separated-tokens",
  "trim-lines",
  "inline-style-parser",
  "web-namespaces",
  "is-decimal",
  "is-hexadecimal",
  "is-alphanumerical",
  "is-alphabetical",
];

/** Package name of a bundled module id, scoped names included. */
function packageOf(id: string): string | null {
  const match = /[\\/]node_modules[\\/](@[^\\/]+[\\/][^\\/]+|[^\\/]+)/.exec(id);
  return match ? match[1] : null;
}

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(rawId: string): string | undefined {
          const pkg = packageOf(rawId);
          if (!pkg) return undefined;
          const isMarkdown = MARKDOWN_PREFIXES.some(
            (prefix) => pkg === prefix || pkg.startsWith(`${prefix}-`) || pkg.startsWith(prefix),
          );
          return isMarkdown ? "vendor-markdown" : undefined;
        },
      },
    },
  },
  server: {
    proxy: {
      "/agent": { target: "http://localhost:8000", changeOrigin: true },
      "/documents": { target: "http://localhost:8000", changeOrigin: true },
      "/indexes": { target: "http://localhost:8000", changeOrigin: true },
      "/tasks": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
