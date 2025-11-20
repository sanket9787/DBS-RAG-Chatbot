import Link from "next/link";
import { ChatPanel } from "@/components/chat/ChatPanel";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 py-12 text-white">
      <div className="mx-auto w-full max-w-4xl px-4 lg:px-0">
        <header className="mb-8 text-center space-y-4">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
            Dublin Business School · Intelligent RAG Assistant
          </p>
          <h1 className="text-4xl font-semibold leading-tight text-white lg:text-5xl">
            Ask Dublin Business School
          </h1>
          <p className="mx-auto max-w-2xl text-base text-slate-300 lg:text-lg">
            Real-time, citation-backed answers about DBS courses, admissions,
            campus life, and student support. Type your question below to start
            a focused conversation.
          </p>
          <div className="flex justify-center">
            <Link
              href="/insights"
              className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/5 px-5 py-2 text-sm text-slate-200 transition hover:border-white/40 hover:text-white"
            >
              View data coverage & system status ↗
            </Link>
          </div>
        </header>

        <ChatPanel />
      </div>
    </main>
  );
}
