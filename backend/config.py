"""
Configuration settings for DBS Chatbot Backend
"""

from pydantic import BaseModel
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory (parent of backend)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Load .env file from project root
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseModel):
    """Application settings"""
    
    # Server settings
    HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS settings
    # Allow CORS_ORIGINS from environment variable, or use defaults
    _cors_origins_env = os.getenv("CORS_ORIGINS", "")
    _debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    
    if _cors_origins_env:
        if _cors_origins_env.strip() == "*":
            # Allow all origins
            CORS_ORIGINS: List[str] = ["*"]
        else:
            # Parse comma-separated origins from environment
            CORS_ORIGINS: List[str] = [origin.strip() for origin in _cors_origins_env.split(",")]
    elif not _debug_mode:
        # In production (non-debug), allow all origins by default
        CORS_ORIGINS: List[str] = ["*"]
    else:
        # Default origins for development
        CORS_ORIGINS: List[str] = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ]
    
    # In production, allow all origins if CORS_ORIGINS contains "*"
    ALLOW_ALL_ORIGINS: bool = "*" in CORS_ORIGINS
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    
    # ChromaDB settings
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "dbs_documents")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
    
    # RAG settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    
    # Response settings
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "True").lower() == "true"
    RESPONSE_TIMEOUT: int = int(os.getenv("RESPONSE_TIMEOUT", "30"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

