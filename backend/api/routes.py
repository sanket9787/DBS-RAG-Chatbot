"""
API Routes for DBS Chatbot
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import logging
from typing import Dict, Any

from models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ChatMessage
)
import state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chatbot"])


def get_rag_service():
    """Dependency to get RAG service"""
    if state.rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return state.rag_service


def get_vector_store():
    """Dependency to get vector store"""
    if state.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return state.vector_store


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: Any = Depends(get_rag_service)
):
    """
    Chat endpoint - Main RAG query endpoint
    
    Args:
        request: Chat request with query and optional history
        service: RAG service dependency
        
    Returns:
        Chat response with generated answer and sources
    """
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Validate query length
        if len(request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query is too long (max 1000 characters)")
        
        # Convert conversation history if provided
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        # Process query through RAG pipeline
        result = service.process_query(
            query=request.query.strip(),
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            conversation_history=conversation_history,
            stream=request.stream
        )
        
        # Handle streaming response
        if request.stream and "stream" in result:
            import asyncio
            
            async def generate_stream():
                try:
                    # OpenAI streaming response is synchronous, so we need to wrap it
                    stream = result["stream"]
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                yield delta.content
                        # Yield control to event loop periodically
                        await asyncio.sleep(0)
                except Exception as stream_error:
                    logger.error(f"Streaming error: {str(stream_error)}", exc_info=True)
                    yield f"\n\n[Error: Streaming interrupted: {str(stream_error)}]"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Validate result structure
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid response format from RAG service")
        
        # Return regular response
        return ChatResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health(
    store: Any = Depends(get_vector_store),
    service: Any = Depends(get_rag_service)
):
    """
    Health check endpoint
    
    Returns:
        Health status of all services
    """
    try:
        collection_count = store.get_collection_count()
        
        return HealthResponse(
            status="healthy",
            vector_store="connected",
            collection_count=collection_count,
            rag_service="ready"
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            vector_store="error",
            collection_count=0,
            rag_service="error"
        )


@router.get("/stats")
async def get_stats(
    store: Any = Depends(get_vector_store)
):
    """
    Get statistics about the knowledge base
    
    Returns:
        Statistics about documents and collection
    """
    try:
        count = store.get_collection_count()
        
        return {
            "total_documents": count,
            "collection_name": store.collection_name,
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

