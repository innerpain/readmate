// FE-D1: a ~40-line hash router.  Three views, zero dependencies.
//
// Why not react-router: this is a single SPA with three flat views and no nested
// routes, loaders or data APIs.  The hash is enough to keep the view across a
// refresh and to deep-link ("资料管理" buttons just set the hash), and it keeps the
// bundle free of a router runtime.

import { useEffect, useState } from "react";

export type View = "chat" | "library" | "memory";

const VIEWS = ["chat", "library", "memory"] as const;
const DEFAULT_VIEW: View = "chat";

/** `#/library`, `#library`, `#/library?x=1` → `library`; anything else → chat. */
export function parseView(hash: string): View {
  const raw = hash.replace(/^#\/?/, "").split("?")[0].trim().toLowerCase();
  return (VIEWS as readonly string[]).includes(raw) ? (raw as View) : DEFAULT_VIEW;
}

export function viewHref(view: View): string {
  return `#/${view}`;
}

export function goToView(view: View): void {
  if (parseView(window.location.hash) === view) return;
  window.location.hash = viewHref(view);
}

/** Current view + a navigate function that writes the hash (history-friendly:
 *  the browser back button walks views, and a refresh stays where you were). */
export function useView(): [View, (next: View) => void] {
  const [view, setView] = useState<View>(() => parseView(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setView(parseView(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    // The very first render may have happened before the hash was applied.
    onHashChange();
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return [view, goToView];
}
