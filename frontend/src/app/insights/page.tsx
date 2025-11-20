import Link from "next/link";
import { SystemStatusCard } from "@/components/status/SystemStatusCard";

const highlights = [
  { label: "Pages Indexed", value: "266" },
  { label: "Knowledge Chunks", value: "391" },
  { label: "Avg. Similarity", value: "1.18" },
  { label: "Relevance Score", value: "85.7%" },
];

const coverage = [
  { label: "Courses & Curriculum", value: "46 docs" },
  { label: "Admissions & Visas", value: "34 docs" },
  { label: "Student Support", value: "22 docs" },
  { label: "Campus Life", value: "18 docs" },
];

const milestones = [
  "Wire backend `/chat` streaming to UI",
  "Progressive rendering for long answers",
  "Authenticated admin dashboard for reloads",
  "User feedback loop for answer quality",
];

export default function InsightsPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 py-12 text-white">
      <div className="mx-auto w-full max-w-6xl px-4 lg:px-6">
        <header className="space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                DBS Chatbot · System Insights
              </p>
              <h1 className="text-4xl font-semibold text-white lg:text-5xl">
                Knowledge Coverage & Telemetry
              </h1>
              <p className="mt-3 max-w-2xl text-base text-slate-300 lg:text-lg">
                Monitor the health of the RAG stack, review what content is in
                the knowledge base, and track upcoming UI milestones.
              </p>
            </div>
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:border-white/40 hover:text-white"
            >
              Back to chat →
            </Link>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {highlights.map((item) => (
              <div
                key={item.label}
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-center shadow-lg shadow-blue-500/10"
              >
                <p className="text-2xl font-semibold text-white">{item.value}</p>
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  {item.label}
                </p>
              </div>
            ))}
          </div>
        </header>

        <section className="mt-10 grid gap-6 lg:grid-cols-[320px,1fr]">
          <aside className="space-y-4">
            <SystemStatusCard />

            <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-lg shadow-sky-500/5">
              <p className="text-sm font-semibold text-slate-300">
                Knowledge Coverage
              </p>
              <ul className="mt-4 space-y-3 text-sm text-slate-200">
                {coverage.map((item) => (
                  <li
                    key={item.label}
                    className="flex items-center justify-between rounded-2xl bg-slate-900/60 px-4 py-3"
                  >
                    <span>{item.label}</span>
                    <span className="text-slate-400">{item.value}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-lg shadow-sky-500/5">
              <p className="text-sm font-semibold text-slate-300">
                Upcoming Milestones
              </p>
              <ul className="mt-4 space-y-2 text-sm text-slate-200">
                {milestones.map((item) => (
                  <li key={item} className="flex items-start gap-2">
                    <span className="text-sky-400">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </aside>

          <div className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-sky-500/10">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                Data Collection Snapshot
              </p>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Web Crawl
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-white">
                    266 pages
                  </p>
                  <p className="text-sm text-slate-400">
                    Dynamic crawler with depth 2 and deduping.
                  </p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Knowledge Base
                  </p>
                  <p className="mt-2 text-2xl font-semibold text-white">
                    391 chunks
                  </p>
                  <p className="text-sm text-slate-400">
                    Persistent ChromaDB collection with OpenAI embeddings.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-sky-500/10">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                Prompt & Retrieval Guardrails
              </p>
              <ul className="mt-4 list-disc space-y-2 pl-5 text-sm text-slate-200">
                <li>Intent-aware query expansion and entity extraction.</li>
                <li>Similarity thresholding with adaptive source filtering.</li>
                <li>Context deduplication plus metadata-aware reranking.</li>
                <li>Prompt instructions enforce DBS-only, citation-backed answers.</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

