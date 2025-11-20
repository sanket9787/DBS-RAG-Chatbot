"""
Global state for services
Avoids circular imports
"""

from typing import Optional
from services.rag_service import RAGService
from services.vector_store import VectorStore

# Global services (initialized in main.py)
rag_service: Optional[RAGService] = None
vector_store: Optional[VectorStore] = None

