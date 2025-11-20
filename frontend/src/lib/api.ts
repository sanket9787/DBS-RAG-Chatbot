export type HealthResponse = {
  status: string;
  vector_store: string;
  collection_count: number;
  rag_service: string;
};

export type StatsResponse = {
  total_documents: number;
  collection_name: string;
  status: string;
};

export type ChatRequest = {
  query: string;
  top_k?: number;
  conversation_history?: { role: "user" | "assistant"; content: string }[];
  filter_metadata?: Record<string, unknown>;
  stream?: boolean;
};

export type ChatResponse = {
  response: string;
  sources: string[];
  context: {
    content: string;
    source: string;
    similarity: number;
  }[];
  model: string;
  tokens_used: number;
  timestamp: string;
  query_info?: {
    intent: string;
    confidence: number;
    entities: Record<string, unknown>;
  };
};

const BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000/api/v1";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || "Unknown API error");
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${BASE_URL}/stats`, { cache: "no-store" });
  return handleResponse<StatsResponse>(res);
}

export async function fetchChat(
  payload: ChatRequest
): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      top_k: 5,
      stream: false,
      ...payload,
    }),
  });
  return handleResponse<ChatResponse>(res);
}

export async function* fetchChatStream(
  payload: ChatRequest
): AsyncGenerator<string, void> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      top_k: 5,
      stream: true,
      ...payload,
    }),
  });

  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || "Unknown API error");
  }

  if (!res.body) {
    throw new Error("Response body is null");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      if (chunk) {
        yield chunk;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

