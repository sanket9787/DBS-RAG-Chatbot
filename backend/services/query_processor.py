"""
Query Processing Service
Handles query normalization, expansion, and intent detection
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process and enhance queries for better retrieval"""
    
    # Query expansion synonyms (domain-specific)
    SYNONYMS = {
        'course': ['program', 'programme', 'degree', 'qualification', 'study'],
        'program': ['course', 'programme', 'degree', 'qualification'],
        'admission': ['entry', 'enrollment', 'enrolment', 'application', 'apply'],
        'requirement': ['requirement', 'prerequisite', 'criteria', 'eligibility'],
        'support': ['help', 'assistance', 'service', 'resource'],
        'library': ['library', 'libraries', 'learning resource'],
        'international': ['overseas', 'foreign', 'non-EU'],
        'undergraduate': ['bachelor', 'bachelors', 'undergrad'],
        'postgraduate': ['masters', 'master', 'graduate', 'postgrad'],
        'fee': ['fees', 'tuition', 'cost', 'price'],
        'scholarship': ['scholarship', 'bursary', 'grant', 'funding'],
    }
    
    # Intent categories
    INTENT_KEYWORDS = {
        'course_inquiry': ['course', 'program', 'degree', 'study', 'qualification'],
        'admission': ['admission', 'entry', 'requirement', 'apply', 'application'],
        'support': ['support', 'help', 'service', 'library', 'career'],
        'international': ['international', 'overseas', 'visa', 'foreign'],
        'fee': ['fee', 'tuition', 'cost', 'scholarship', 'funding'],
        'general': ['about', 'information', 'what', 'tell me', 'explain']
    }
    
    def __init__(self):
        """Initialize query processor"""
        pass
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query text
        
        Args:
            query: Raw user query
            
        Returns:
            Normalized query
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters (keep alphanumeric, spaces, and common punctuation)
        normalized = re.sub(r'[^\w\s\?\.]', '', normalized)
        
        # Remove trailing punctuation (except question marks)
        normalized = normalized.rstrip('.,!;:')
        
        return normalized
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms
        
        Args:
            query: Normalized query
            
        Returns:
            List of query variations
        """
        query_lower = query.lower()
        variations = [query]  # Original query first
        
        # Find synonyms for words in query
        words = query_lower.split()
        for word in words:
            # Check if word has synonyms
            for key, synonyms in self.SYNONYMS.items():
                if key in word or word in key:
                    # Add variations with synonyms
                    for synonym in synonyms:
                        if synonym != word:
                            variation = query_lower.replace(word, synonym)
                            if variation not in variations:
                                variations.append(variation)
        
        # Limit to top 3 variations to avoid too many queries
        return variations[:3]
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect query intent
        
        Args:
            query: User query
            
        Returns:
            Intent information with category and confidence
        """
        query_lower = query.lower()
        intent_scores = {}
        
        # Score each intent category
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent] / len(query_lower.split())
        else:
            primary_intent = 'general'
            confidence = 0.5
        
        return {
            'intent': primary_intent,
            'confidence': min(confidence, 1.0),
            'all_scores': intent_scores
        }
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query (simple keyword-based)
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted entities
        """
        query_lower = query.lower()
        entities = {
            'course_level': [],
            'subject': [],
            'location': []
        }
        
        # Course level detection
        if any(word in query_lower for word in ['undergraduate', 'bachelor', 'bachelors', 'undergrad']):
            entities['course_level'].append('undergraduate')
        if any(word in query_lower for word in ['postgraduate', 'masters', 'master', 'graduate', 'postgrad']):
            entities['course_level'].append('postgraduate')
        if any(word in query_lower for word in ['diploma', 'certificate']):
            entities['course_level'].append('diploma')
        
        # Subject detection (simplified)
        subjects = [
    'marketing', 'computing', 'business', 'law', 'accounting', 'psychology',
    'artificial intelligence', 'ai', 'data analytics', 'data science',
    'finance', 'fintech', 'human resources', 'hr', 'digital marketing',
    'cybersecurity', 'ethical hacking', 'counselling', 'counseling', 'legal studies'
]
        for subject in subjects:
            if subject in query_lower:
                entities['subject'].append(subject)
        
        # Location detection
        if 'international' in query_lower or 'overseas' in query_lower:
            entities['location'].append('international')
        
        return entities
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query processing pipeline
        
        Args:
            query: Raw user query
            
        Returns:
            Processed query information
        """
        # Normalize
        normalized = self.normalize_query(query)
        
        # Expand
        variations = self.expand_query(normalized)
        
        # Detect intent
        intent_info = self.detect_intent(normalized)
        
        # Extract entities
        entities = self.extract_entities(normalized)
        
        return {
            'original': query,
            'normalized': normalized,
            'variations': variations,
            'intent': intent_info,
            'entities': entities,
            'processed_query': normalized  # Use normalized for embedding
        }
    
    def get_metadata_filters(self, processed_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate metadata filters based on query processing
        
        Args:
            processed_query: Processed query information
            
        Returns:
            ChromaDB metadata filter or None
        """
        filters = {}
        entities = processed_query.get('entities', {})
        
        # Filter by course level if detected
        course_levels = entities.get('course_level', [])
        if course_levels:
            # ChromaDB uses $in for multiple values
            filters['category'] = {'$in': [f'Courses {level.title()}' for level in course_levels]}
        
        # Filter by subject if detected
        subjects = entities.get('subject', [])
        if subjects:
            # Could filter by subcategory if we have that metadata
            pass
        
        return filters if filters else None

