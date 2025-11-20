"""
DBS Intelligent RAG Chatbot - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv
import os

from config import settings
from services.rag_service import RAGService
from services.vector_store import VectorStore
import state

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting DBS Chatbot Backend...")
    
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        state.vector_store = VectorStore(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        
        # Initialize RAG service
        logger.info("Initializing RAG service...")
        state.rag_service = RAGService(
            vector_store=state.vector_store,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        logger.info("✅ Backend services initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down backend services...")
        if state.vector_store:
            state.vector_store.close()


# Create FastAPI app
app = FastAPI(
    title="DBS Intelligent RAG Chatbot API",
    description="RAG-based chatbot for Dublin Business School",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (import after state is set up)
from api.routes import router
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DBS Intelligent RAG Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        if state.vector_store:
            collection_count = state.vector_store.get_collection_count()
        else:
            collection_count = 0
        
        return {
            "status": "healthy",
            "vector_store": "connected" if state.vector_store else "disconnected",
            "collection_count": collection_count,
            "rag_service": "ready" if state.rag_service else "not_ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

