# DBS Intelligent RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot designed specifically for Dublin Business School (DBS) to provide instant, accurate responses to student and prospective student queries about courses, admissions, campus life, and student support services.

## 🎯 Project Overview

This project implements a production-ready RAG (Retrieval-Augmented Generation) system that combines:
- **Vector Search**: Semantic understanding of queries using embeddings
- **Knowledge Base**: Curated DBS-specific information (391+ documents)
- **LLM Integration**: OpenAI GPT-4 Turbo for natural language generation
- **Modern Stack**: FastAPI backend + Next.js frontend

### Why RAG?

Traditional chatbots often provide generic responses or hallucinate information. This RAG system:
- ✅ **Accurate**: Grounded in real DBS data, reducing hallucinations
- ✅ **Relevant**: Context-aware responses based on actual DBS content
- ✅ **Transparent**: Cites sources for verification
- ✅ **Scalable**: Easy to update with new information

## ✨ Features

- 🤖 **Intelligent Query Processing**: Natural language understanding with query expansion
- 🔍 **Semantic Search**: Vector-based similarity search for relevant context
- 💬 **Conversational Interface**: Maintains conversation history and context
- 📚 **Source Attribution**: Provides citations for all responses
- ⚡ **Real-time Responses**: Fast response times (< 3 seconds)
- 🎨 **Modern UI**: Clean, responsive Next.js interface
- 🐳 **Docker Support**: Containerized for easy deployment
- 🧪 **Comprehensive Testing**: Unit, integration, and performance tests

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Python 3.11+**: Modern Python with async support
- **ChromaDB**: Vector database for embeddings
- **OpenAI API**: GPT-4 Turbo and text-embedding-3-large

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Zustand**: State management

### Infrastructure
- **Docker**: Containerization
- **Railway/Render**: Backend hosting
- **Vercel**: Frontend hosting

## 📋 Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Docker** (optional, for containerized deployment)
- **Git**

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sanket9787/DBS-RAG-Chatbot.git
cd DBS-RAG-Chatbot
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

**Backend:**
```bash
# Copy example env file
cp ../env.example ../.env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

**Frontend:**
```bash
# Create .env.local in frontend directory
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000/api/v1" > .env.local
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
# Server runs on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# App runs on http://localhost:3000
```

Visit `http://localhost:3000` to use the chatbot!

## 📁 Project Structure

```
DBS-RAG-Chatbot/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes
│   ├── models/             # Pydantic models
│   ├── services/           # RAG service, vector store, query processor
│   ├── tests/              # Test suite
│   ├── main.py            # Application entry point
│   ├── config.py          # Configuration settings
│   └── requirements.txt   # Python dependencies
│
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app router pages
│   │   ├── components/    # React components
│   │   ├── lib/           # Utilities and API client
│   │   └── store/         # State management
│   └── package.json       # Node dependencies
│
├── scripts/                # Data collection and processing scripts
│   ├── scrape_data.py     # Web scraping
│   ├── build_knowledge_base.py  # Knowledge base construction
│   └── ...
│
├── data/                   # Data storage
│   └── chroma_db/         # ChromaDB vector store
│
├── docs/                   # Documentation
├── .env.example           # Environment variables template
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

See `env.example` for all available configuration options. Key variables:

**Required:**
- `OPENAI_API_KEY`: Your OpenAI API key

**Optional (with defaults):**
- `BACKEND_HOST`: Server host (default: `127.0.0.1`)
- `BACKEND_PORT`: Server port (default: `8000`)
- `CHROMA_COLLECTION_NAME`: Vector store collection (default: `dbs_documents`)
- `OPENAI_MODEL`: LLM model (default: `gpt-4-turbo-preview`)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-large`)

## 📡 API Endpoints

### POST `/api/v1/chat`
Main chat endpoint for RAG queries.

**Request:**
```json
{
  "query": "What courses does DBS offer?",
  "conversation_history": [],
  "top_k": 5
}
```

**Response:**
```json
{
  "response": "DBS offers a wide range of courses...",
  "sources": ["https://www.dbs.ie/courses/"],
  "context": [...],
  "model": "gpt-4-turbo-preview",
  "tokens_used": 250
}
```

### GET `/api/v1/health`
Health check endpoint.

### GET `/api/v1/stats`
Get knowledge base statistics.

Full API documentation available at `http://localhost:8000/docs` when backend is running.

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build backend image
cd backend
docker build -t dbs-chatbot-backend .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/../data/chroma_db:/app/data/chroma_db \
  dbs-chatbot-backend
```

### Docker Compose

```bash
# Run entire stack
docker-compose up -d
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 🚀 Deployment

### Backend (Railway/Render)

1. Connect your GitHub repository
2. Set environment variables in platform dashboard
3. Deploy automatically on push

### Frontend (Vercel)

1. Connect GitHub repository
2. Set `NEXT_PUBLIC_BACKEND_URL` environment variable
3. Deploy automatically

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions.

## 📊 Project Phases

- ✅ **Phase 1**: Research & Planning
- ✅ **Phase 2**: Data Collection (266 pages scraped, 391 documents indexed)
- ✅ **Phase 3**: Backend Development (RAG pipeline, FastAPI)
- ✅ **Phase 4**: Frontend Development (Next.js interface)
- ✅ **Phase 5**: Testing & Evaluation
- ✅ **Phase 6**: Deployment & Optimization

## 📚 Documentation

- [Deployment Guide](./DEPLOYMENT_GUIDE.md)
- [Project Overview](./PROJECT_OVERVIEW.md)
- [Backend README](./backend/README.md)

## 🤝 Contributing

This is a master's thesis project. For questions or suggestions, please contact:

**Author**: Sanket Walunj (20060376)  
**Institution**: Dublin Business School

## 📄 License

This project is for academic purposes only.

## 🙏 Acknowledgments

- Dublin Business School for providing the use case
- OpenAI for GPT-4 and embedding models
- Open source community for excellent tools and frameworks

---

**Status**: ✅ Production Ready  
**Last Updated**: November 2025
# DBS-RAG-Chatbot
# DBS-RAG-Chatbot
# DBS-RAG-Chatbot
# DBS-RAG-Chatbot
# DBS-RAG-Chatbot
# DBS-RAG-Chatbot
