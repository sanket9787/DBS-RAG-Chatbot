"""
Performance benchmarks for the DBS chatbot
"""
import pytest
import time
from services.rag_service import RAGService
from services.vector_store import VectorStore


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_query_response_time(self, mock_vector_store, mock_openai_client):
        """Benchmark: Query response time should be < 3 seconds"""
        rag_service = RAGService(
            vector_store=mock_vector_store,
            openai_api_key="test-key"
        )
        rag_service.openai_client = mock_openai_client
        
        start_time = time.time()
        result = rag_service.process_query("what courses are available?")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 3.0, f"Response time {response_time:.2f}s exceeds 3s threshold"
    
    def test_vector_search_performance(self, mock_vector_store):
        """Benchmark: Vector search should be < 200ms"""
        import numpy as np
        
        # Mock embedding
        test_embedding = np.random.rand(1536).tolist()
        
        start_time = time.time()
        results = mock_vector_store.search(
            query_embedding=test_embedding,
            top_k=5
        )
        end_time = time.time()
        
        search_time = end_time - start_time
        assert search_time < 0.2, f"Vector search time {search_time*1000:.2f}ms exceeds 200ms threshold"
    
    def test_concurrent_queries(self, mock_vector_store, mock_openai_client):
        """Benchmark: System should handle concurrent queries"""
        import concurrent.futures
        
        rag_service = RAGService(
            vector_store=mock_vector_store,
            openai_api_key="test-key"
        )
        rag_service.openai_client = mock_openai_client
        
        queries = ["query 1", "query 2", "query 3", "query 4", "query 5"]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(rag_service.process_query, query)
                for query in queries
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_query = total_time / len(queries)
        
        assert len(results) == 5, "Not all queries completed"
        assert avg_time_per_query < 3.0, f"Average time {avg_time_per_query:.2f}s exceeds threshold"

