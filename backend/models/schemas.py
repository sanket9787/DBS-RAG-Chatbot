"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation messages"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )


class SourceInfo(BaseModel):
    """Source information model"""
    content: str = Field(..., description="Content snippet")
    source: str = Field(..., description="Source URL")
    similarity: float = Field(..., description="Similarity score")


class QueryInfo(BaseModel):
    """Query processing information"""
    intent: str = Field(..., description="Detected query intent")
    confidence: float = Field(..., description="Intent detection confidence")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Generated response")
    sources: List[str] = Field(default_factory=list, description="Source URLs")
    context: List[SourceInfo] = Field(default_factory=list, description="Retrieved context")
    model: str = Field(..., description="Model used for generation")
    tokens_used: int = Field(default=0, description="Tokens used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    query_info: Optional[QueryInfo] = Field(default=None, description="Query processing information")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    vector_store: str = Field(..., description="Vector store status")
    collection_count: int = Field(..., description="Number of documents in collection")
    rag_service: str = Field(..., description="RAG service status")

