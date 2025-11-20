"""
Unit tests for Vector Store Service
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from services.vector_store import VectorStore


class TestVectorStore:
    """Test cases for VectorStore"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test database"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create vector store instance for testing"""
        return VectorStore(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
    
    def test_initialization(self, vector_store):
        """Test vector store initialization"""
        assert vector_store.collection_name == "test_collection"
        assert vector_store.collection is not None
    
    def test_get_collection_count_empty(self, vector_store):
        """Test getting count from empty collection"""
        count = vector_store.get_collection_count()
        assert count == 0
    
    def test_search_empty_collection(self, vector_store):
        """Test searching empty collection"""
        # Create a dummy embedding (3072 dimensions for text-embedding-3-large)
        dummy_embedding = [0.1] * 3072
        
        results = vector_store.search(
            query_embedding=dummy_embedding,
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_get_by_ids_empty(self, vector_store):
        """Test getting documents by IDs from empty collection"""
        results = vector_store.get_by_ids(["non_existent_id"])
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_close(self, vector_store):
        """Test closing vector store"""
        # Should not raise any exceptions
        vector_store.close()
        assert True  # If we get here, close() worked
    
    def test_search_with_metadata_filter(self, vector_store):
        """Test search with metadata filter"""
        dummy_embedding = [0.1] * 3072
        
        results = vector_store.search(
            query_embedding=dummy_embedding,
            top_k=5,
            filter_metadata={"category": "test"}
        )
        
        assert isinstance(results, list)

