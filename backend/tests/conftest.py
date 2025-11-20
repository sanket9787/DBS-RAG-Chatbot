"""
Shared pytest fixtures and configuration
"""
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rag_service import RAGService
from services.vector_store import VectorStore
from services.query_processor import QueryProcessor


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Test response"))],
        usage=Mock(total_tokens=100)
    ))
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search = Mock(return_value=[
        {
            "content": "Test content",
            "metadata": {"source_url": "https://example.com", "title": "Test"},
            "similarity": 0.85
        }
    ])
    mock_store.get_collection_count = Mock(return_value=100)
    return mock_store


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "What courses are available in data analytics?"


@pytest.fixture
def sample_context():
    """Sample context documents for testing"""
    return [
        {
            "content": "DBS offers a Master of Science in Data Analytics...",
            "metadata": {
                "source_url": "https://dbs.ie/courses/data-analytics",
                "title": "Data Analytics Course",
                "category": "course"
            },
            "similarity": 0.92
        },
        {
            "content": "The MSc in Data Analytics requires...",
            "metadata": {
                "source_url": "https://dbs.ie/courses/data-analytics",
                "title": "Data Analytics Requirements",
                "category": "admission"
            },
            "similarity": 0.88
        }
    ]


@pytest.fixture
def test_settings():
    """Test settings configuration"""
    return {
        "OPENAI_MODEL": "gpt-4-turbo-preview",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        "OPENAI_TEMPERATURE": 0.7,
        "OPENAI_MAX_TOKENS": 1000,
        "CHROMA_COLLECTION_NAME": "test_dbs_documents",
        "TOP_K_RESULTS": 5,
        "SIMILARITY_THRESHOLD": 0.7,
        "MAX_CONTEXT_LENGTH": 4000
    }

