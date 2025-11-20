"""
Vector Store Service - ChromaDB Integration
Handles vector database operations for RAG
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store service for ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = "dbs_documents",
        persist_directory: str = "./data/chroma_db"
    ):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "DBS documents for RAG chatbot"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Prepare query
            where_clause = filter_metadata if filter_metadata else None
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else None
                    # Convert distance to similarity (ChromaDB uses L2 distance, normalize to 0-1)
                    # For large distances, use inverse or normalize based on expected range
                    if distance is not None:
                        # Normalize distance to similarity (simple approach: 1 / (1 + distance))
                        # This works better for L2 distances which can be large
                        similarity = 1.0 / (1.0 + distance / 100.0)  # Scale by 100 for better range
                    else:
                        similarity = None
                    
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": distance,
                        "similarity": similarity
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_collection_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0
    
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get documents by IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results['documents']:
                for i, doc_id in enumerate(ids):
                    if doc_id in results['ids']:
                        idx = results['ids'].index(doc_id)
                        formatted_results.append({
                            "id": doc_id,
                            "content": results['documents'][idx],
                            "metadata": results['metadatas'][idx] if results['metadatas'] else {}
                        })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting documents by IDs: {str(e)}")
            return []
    
    def close(self):
        """Close vector store connection"""
        # ChromaDB PersistentClient doesn't need explicit closing
        logger.info("Vector store connection closed")

