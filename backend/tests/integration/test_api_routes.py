"""
Integration tests for API routes
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# Import after path setup
from main import app
import state


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock RAG service and vector store"""
    mock_rag = Mock()
    mock_rag.process_query = Mock(return_value={
        "response": "Test response",
        "sources": ["https://example.com"],
        "context": [],
        "model": "gpt-4-turbo-preview",
        "tokens_used": 100
    })
    
    mock_vector = Mock()
    mock_vector.get_collection_count = Mock(return_value=100)
    mock_vector.collection_name = "test_collection"  # Simple string attribute
    
    # Set in state
    state.rag_service = mock_rag
    state.vector_store = mock_vector
    
    yield mock_rag, mock_vector
    
    # Cleanup
    state.rag_service = None
    state.vector_store = None


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_endpoint(self, client, mock_services):
        """Test GET /api/v1/health"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]


class TestStatsEndpoint:
    """Test stats endpoint"""
    
    def test_stats_endpoint(self, client, mock_services):
        """Test GET /api/v1/stats"""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert isinstance(data["total_documents"], int)


class TestChatEndpoint:
    """Test chat endpoint"""
    
    def test_chat_endpoint_success(self, client, mock_services):
        """Test POST /api/v1/chat with valid query"""
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "what courses are available?",
                "stream": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
    
    def test_chat_endpoint_empty_query(self, client, mock_services):
        """Test POST /api/v1/chat with empty query"""
        response = client.post(
            "/api/v1/chat",
            json={"query": "", "stream": False}
        )
        # FastAPI returns 422 for validation errors (empty string fails validation)
        assert response.status_code in [400, 422]
    
    def test_chat_endpoint_long_query(self, client, mock_services):
        """Test POST /api/v1/chat with query too long"""
        long_query = "a" * 1001
        response = client.post(
            "/api/v1/chat",
            json={"query": long_query, "stream": False}
        )
        # FastAPI returns 422 for validation errors, or 400 from our custom validation
        assert response.status_code in [400, 422]


