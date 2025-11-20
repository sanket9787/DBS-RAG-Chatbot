# DBS Chatbot Backend

FastAPI backend for the DBS Intelligent RAG Chatbot.

## 🏗️ Architecture

### Components

1. **FastAPI Application** (`main.py`)
   - Main application entry point
   - Lifespan management for services
   - CORS configuration
   - Error handling

2. **RAG Service** (`services/rag_service.py`)
   - Query processing
   - Vector retrieval
   - Response generation
   - OpenAI integration

3. **Vector Store** (`services/vector_store.py`)
   - ChromaDB integration
   - Document search
   - Similarity matching

4. **API Routes** (`api/routes.py`)
   - `/api/v1/chat` - Main chat endpoint
   - `/api/v1/health` - Health check
   - `/api/v1/stats` - Statistics

## 🚀 Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Server Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
DEBUG=False

# ChromaDB Configuration
CHROMA_COLLECTION_NAME=dbs_documents
CHROMA_PERSIST_DIR=./data/chroma_db

# RAG Configuration
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_LENGTH=4000
```

### 3. Run the Server

```bash
# From backend directory
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

### POST `/api/v1/chat`

Main chat endpoint for RAG queries.

**Request:**
```json
{
  "query": "What courses does DBS offer?",
  "conversation_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ],
  "top_k": 5,
  "stream": false
}
```

**Response:**
```json
{
  "response": "DBS offers a wide range of courses...",
  "sources": ["https://www.dbs.ie/courses/"],
  "context": [
    {
      "content": "DBS offers undergraduate and postgraduate...",
      "source": "https://www.dbs.ie/courses/",
      "similarity": 0.95
    }
  ],
  "model": "gpt-4-turbo-preview",
  "tokens_used": 250,
  "timestamp": "2025-10-28T12:00:00"
}
```

### GET `/api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "vector_store": "connected",
  "collection_count": 391,
  "rag_service": "ready"
}
```

### GET `/api/v1/stats`

Get knowledge base statistics.

**Response:**
```json
{
  "total_documents": 391,
  "collection_name": "dbs_documents",
  "status": "active"
}
```

## 🔄 RAG Pipeline Flow

1. **Query Reception**: User query received via API
2. **Embedding Generation**: Convert query to vector embedding
3. **Vector Search**: Search ChromaDB for similar documents
4. **Context Retrieval**: Get top K relevant documents
5. **Prompt Building**: Construct prompt with context
6. **LLM Generation**: Generate response using OpenAI
7. **Response Return**: Return answer with sources

## 🧪 Testing

### Test Health Endpoint

```bash
curl http://localhost:8000/api/v1/health
```

### Test Chat Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What courses does DBS offer?",
    "top_k": 5
  }'
```

## 📝 Notes

- The backend requires the ChromaDB knowledge base to be built (Phase 2)
- OpenAI API key is required for full functionality
- Without API key, simulated responses will be returned
- Vector store is persisted in `./data/chroma_db/`

