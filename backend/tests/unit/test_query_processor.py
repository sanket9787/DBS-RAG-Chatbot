"""
Unit tests for QueryProcessor
"""
import pytest
from services.query_processor import QueryProcessor


class TestQueryProcessor:
    """Test cases for QueryProcessor"""
    
    def test_normalize_query(self):
        """Test query normalization"""
        processor = QueryProcessor()
        
        # Test lowercase conversion (question marks are kept)
        assert processor.normalize_query("WHAT COURSES ARE AVAILABLE?") == "what courses are available?"
        
        # Test trimming
        assert processor.normalize_query("  data analytics  ") == "data analytics"
        
        # Test special character handling (apostrophes are kept, other special chars removed)
        result = processor.normalize_query("what's the fee?")
        assert "what" in result and "fee" in result
    
    def test_detect_intent(self):
        """Test intent detection"""
        processor = QueryProcessor()
        
        # Course inquiry
        result = processor.detect_intent("what courses do you offer?")
        assert result["intent"] == "course_inquiry"
        
        # Admission
        result = processor.detect_intent("how do I apply?")
        assert result["intent"] == "admission"
        
        # Support
        result = processor.detect_intent("where is the library?")
        assert result["intent"] == "support"
        
        # General
        result = processor.detect_intent("tell me about dbs")
        assert result["intent"] == "general"
    
    def test_extract_entities(self):
        """Test entity extraction"""
        processor = QueryProcessor()
        
        # Test course level extraction (returns "postgraduate" for masters)
        result = processor.extract_entities("masters in data analytics")
        assert len(result.get("course_level", [])) > 0  # Should extract course level
        
        # Test subject extraction
        result = processor.extract_entities("ai course fees")
        assert len(result.get("subject", [])) > 0  # Should extract subject
        
        # Test location extraction (only extracts "international" or "overseas")
        result = processor.extract_entities("international students")
        assert len(result.get("location", [])) > 0  # Should extract location
    
    def test_process_query(self):
        """Test complete query processing"""
        processor = QueryProcessor()
        
        result = processor.process_query("masters in data analytics fees")
        
        assert "intent" in result
        assert "entities" in result
        assert isinstance(result["intent"], dict)
        assert "intent" in result["intent"]
        assert isinstance(result["entities"], dict)

