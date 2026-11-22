import { useEffect, useState } from "react";
import { api } from "./api";
import type { Health } from "./types";
import Chat from "./pages/Chat";
import Traces from "./pages/Traces";
import Evaluation from "./pages/Evaluation";
import Retrieval from "./pages/Retrieval";

const PAGES = { chat: "Chat", traces: "Traces", eval: "Evaluation", retrieval: "Retrieval experiments" } as const;
type Page = keyof typeof PAGES;

function pageFromHash(): Page {
  const h = window.location.hash.replace("#/", "") as Page;
  return h in PAGES ? h : "chat";
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash());
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth(null));
    load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">🧭 HR Policy Navigator</span>
        <nav>
          {(Object.keys(PAGES) as Page[]).map((p) => (
            <a key={p} href={`#/${p}`} className={p === page ? "active" : ""}>
              {PAGES[p]}
            </a>
          ))}
        </nav>
        <span className="status" title={health?.llm.error ?? ""}>
          <span className={`dot ${health ? (health.status === "ok" ? "ok" : "warn") : "down"}`} />
          {health ? `${health.status} · ${health.retriever} · ${health.storage}` : "backend down"}
        </span>
      </header>
      <main>
        {page === "chat" && <Chat />}
        {page === "traces" && <Traces />}
        {page === "eval" && <Evaluation />}
        {page === "retrieval" && <Retrieval />}
      </main>
    </div>
  );
}
