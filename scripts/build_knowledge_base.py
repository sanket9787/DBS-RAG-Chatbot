#!/usr/bin/env python3
"""
DBS Knowledge Base Construction Script
Phase 2: Data Collection - Knowledge Base Construction

This script builds the knowledge base for the RAG chatbot including:
- Content aggregation and organization
- Embedding generation
- Vector database population
- Metadata enrichment
- Search index creation

Author: DBS Chatbot Project
Date: October 2024
"""

import json
import logging
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Environment variables
from dotenv import load_dotenv

# Vector database and embeddings
import chromadb
import openai
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_base.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeBaseItem:
    """Data class for knowledge base items"""
    item_id: str
    content: str
    title: str
    category: str
    subcategory: str
    source_url: str
    metadata: Dict[str, Any]
    embedding: List[float]
    created_at: str

class KnowledgeBaseBuilder:
    """Main class for building the knowledge base"""
    
    def __init__(self, 
                 collection_name: str = "dbs_documents",
                 openai_api_key: str = None):
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not provided, embeddings will be simulated")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "DBS Chatbot Knowledge Base"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Initialize text processing
        self._setup_nltk()
        
        # Content processing settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_chunks_per_document = 10

    def _setup_nltk(self):
        """Setup NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API"""
        if not self.openai_client:
            # Generate random embedding for testing
            return np.random.rand(3072).tolist()
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to random embedding
            return np.random.rand(3072).tolist()

    def chunk_content(self, content: str, title: str = "") -> List[str]:
        """Split content into chunks for better retrieval"""
        if not content:
            return []
        
        # Split into sentences first
        sentences = sent_tokenize(content)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('.')[-self.chunk_overlap//50:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                current_length = len(current_chunk.split())
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Limit number of chunks per document
        return chunks[:self.max_chunks_per_document]

    def enrich_metadata(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata with additional information (flattened for ChromaDB)"""
        original_metadata = content_data.get('metadata', {}).copy()
        
        # Start with basic info (flatten nested structures)
        content = content_data.get('content', '')
        metadata = {
            'source_url': content_data.get('url', ''),
            'title': content_data.get('title', ''),
            'category': content_data.get('category', 'general'),
            'subcategory': content_data.get('subcategory', 'general'),
            'word_count': int(original_metadata.get('word_count', len(content.split()))),
            'char_count': int(original_metadata.get('char_count', len(content))),
            'has_images': str(original_metadata.get('has_images', False)),
            'has_links': str(original_metadata.get('has_links', False)),
            'has_tables': str(original_metadata.get('has_tables', False)),
        }
        
        # Flatten quality_assessment if it exists
        if 'quality_assessment' in original_metadata:
            quality_data = original_metadata['quality_assessment']
            metadata['quality_score'] = float(quality_data.get('quality_score', 0.0))
            
            # Flatten issues list to string
            issues = quality_data.get('issues', [])
            if isinstance(issues, list):
                metadata['quality_issues'] = ', '.join(str(i) for i in issues[:5])  # Limit to 5 issues
            else:
                metadata['quality_issues'] = str(issues)
            
            # Flatten metrics dict if it exists
            if 'metrics' in quality_data and isinstance(quality_data['metrics'], dict):
                metrics = quality_data['metrics']
                metadata['metrics_word_count'] = int(metrics.get('word_count', 0))
                metadata['metrics_char_count'] = int(metrics.get('char_count', 0))
                metadata['metrics_sentence_count'] = int(metrics.get('sentence_count', 0))
                metadata['metrics_relevance_score'] = float(metrics.get('relevance_score', 0.0))
        
        return metadata

    def process_content_item(self, content_data: Dict[str, Any]) -> List[KnowledgeBaseItem]:
        """Process a single content item into knowledge base items"""
        try:
            # Extract basic information
            content = content_data.get('content', '')
            title = content_data.get('title', '')
            category = content_data.get('category', 'general')
            subcategory = content_data.get('subcategory', 'general')
            source_url = content_data.get('url', '')
            content_hash = content_data.get('content_hash', '')
            
            if not content or len(content.strip()) < 50:
                logger.warning(f"Skipping content with insufficient text: {source_url}")
                return []
            
            # Enrich metadata
            enriched_metadata = self.enrich_metadata(content_data)
            
            # Chunk content
            chunks = self.chunk_content(content, title)
            
            if not chunks:
                logger.warning(f"No chunks created for content: {source_url}")
                return []
            
            # Create knowledge base items for each chunk
            knowledge_items = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID for this chunk
                chunk_id = f"{content_hash}_{i}"
                
                # Generate embedding
                embedding = self.generate_embedding(chunk)
                
                # Create metadata for this chunk
                chunk_metadata = {
                    **enriched_metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_word_count': len(chunk.split()),
                    'parent_title': title,
                    'parent_url': source_url
                }
                
                # Create knowledge base item
                kb_item = KnowledgeBaseItem(
                    item_id=chunk_id,
                    content=chunk,
                    title=f"{title} (Part {i+1})" if len(chunks) > 1 else title,
                    category=category,
                    subcategory=subcategory,
                    source_url=source_url,
                    metadata=chunk_metadata,
                    embedding=embedding,
                    created_at=datetime.now().isoformat()
                )
                
                knowledge_items.append(kb_item)
            
            logger.info(f"Processed {source_url}: {len(chunks)} chunks created")
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error processing content item: {str(e)}")
            return []

    def add_to_vector_database(self, knowledge_items: List[KnowledgeBaseItem]) -> bool:
        """Add knowledge items to vector database"""
        try:
            if not knowledge_items:
                return False
            
            # Prepare data for ChromaDB
            ids = [item.item_id for item in knowledge_items]
            documents = [item.content for item in knowledge_items]
            metadatas = [item.metadata for item in knowledge_items]
            embeddings = [item.embedding for item in knowledge_items]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(knowledge_items)} items to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding items to vector database: {str(e)}")
            return False

    def build_knowledge_base(self, 
                           web_content_file: str = "data/processed/cleaned_web_content.json",
                           pdf_content_file: str = "data/processed/cleaned_pdf_content.json",
                           output_file: str = "data/processed/knowledge_base.json") -> Dict[str, Any]:
        """Build the complete knowledge base"""
        logger.info("Starting knowledge base construction...")
        
        all_knowledge_items = []
        processing_stats = {
            'total_items_processed': 0,
            'total_chunks_created': 0,
            'web_content_items': 0,
            'pdf_content_items': 0,
            'failed_items': 0,
            'categories': {},
            'subcategories': {},
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Process web content
        if Path(web_content_file).exists():
            logger.info(f"Processing web content from {web_content_file}")
            
            with open(web_content_file, 'r', encoding='utf-8') as f:
                web_content = json.load(f)
            
            for item in web_content:
                knowledge_items = self.process_content_item(item)
                all_knowledge_items.extend(knowledge_items)
                processing_stats['web_content_items'] += 1
                processing_stats['total_chunks_created'] += len(knowledge_items)
                
                # Track categories
                category = item.get('category', 'general')
                processing_stats['categories'][category] = processing_stats['categories'].get(category, 0) + 1
                
                subcategory = item.get('subcategory', 'general')
                processing_stats['subcategories'][f"{category}_{subcategory}"] = processing_stats['subcategories'].get(f"{category}_{subcategory}", 0) + 1
        else:
            logger.warning(f"Web content file not found: {web_content_file}")
        
        # Process PDF content
        if Path(pdf_content_file).exists():
            logger.info(f"Processing PDF content from {pdf_content_file}")
            
            with open(pdf_content_file, 'r', encoding='utf-8') as f:
                pdf_content = json.load(f)
            
            for item in pdf_content:
                knowledge_items = self.process_content_item(item)
                all_knowledge_items.extend(knowledge_items)
                processing_stats['pdf_content_items'] += 1
                processing_stats['total_chunks_created'] += len(knowledge_items)
                
                # Track categories
                category = item.get('category', 'general')
                processing_stats['categories'][category] = processing_stats['categories'].get(category, 0) + 1
                
                subcategory = item.get('subcategory', 'general')
                processing_stats['subcategories'][f"{category}_{subcategory}"] = processing_stats['subcategories'].get(f"{category}_{subcategory}", 0) + 1
        else:
            logger.warning(f"PDF content file not found: {pdf_content_file}")
        
        # Add to vector database
        if all_knowledge_items:
            logger.info(f"Adding {len(all_knowledge_items)} knowledge items to vector database...")
            success = self.add_to_vector_database(all_knowledge_items)
            
            if success:
                logger.info("Successfully added all items to vector database")
            else:
                logger.error("Failed to add items to vector database")
        
        # Save knowledge base data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        knowledge_data = [asdict(item) for item in all_knowledge_items]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
        
        # Update processing stats
        processing_stats['total_items_processed'] = len(all_knowledge_items)
        
        # Save processing stats
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processing_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge base construction completed: {processing_stats['total_chunks_created']} chunks created")
        
        return processing_stats

    def test_knowledge_base(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Test the knowledge base with sample queries"""
        if not test_queries:
            test_queries = [
                "What courses are available at DBS?",
                "How do I apply for admission?",
                "What student support services are available?",
                "What are the entry requirements?",
                "How much does it cost to study at DBS?"
            ]
        
        logger.info("Testing knowledge base with sample queries...")
        
        test_results = {
            'queries_tested': len(test_queries),
            'successful_queries': 0,
            'failed_queries': 0,
            'query_results': [],
            'test_timestamp': datetime.now().isoformat()
        }
        
        for query in test_queries:
            try:
                # Generate query embedding
                query_embedding = self.generate_embedding(query)
                
                # Search the collection
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )
                
                if results['documents'] and results['documents'][0]:
                    test_results['successful_queries'] += 1
                    test_results['query_results'].append({
                        'query': query,
                        'status': 'success',
                        'results_count': len(results['documents'][0]),
                        'top_result': results['documents'][0][0][:100] + '...' if results['documents'][0] else None
                    })
                else:
                    test_results['failed_queries'] += 1
                    test_results['query_results'].append({
                        'query': query,
                        'status': 'failed',
                        'results_count': 0,
                        'top_result': None
                    })
                
            except Exception as e:
                logger.error(f"Error testing query '{query}': {str(e)}")
                test_results['failed_queries'] += 1
                test_results['query_results'].append({
                    'query': query,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save test results
        with open("data/processed/knowledge_base_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Knowledge base testing completed: {test_results['successful_queries']}/{test_results['queries_tested']} queries successful")
        
        return test_results

def main():
    """Main function to build knowledge base"""
    logger.info("Starting knowledge base construction...")
    
    # Load OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip('"').strip("'")
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        logger.warning("OpenAI API key not found in environment. Using simulated embeddings.")
        openai_api_key = None
    else:
        logger.info("OpenAI API key loaded from environment")
    
    # Initialize knowledge base builder
    kb_builder = KnowledgeBaseBuilder(
        openai_api_key=openai_api_key
    )
    
    # Build knowledge base
    stats = kb_builder.build_knowledge_base()
    
    # Test knowledge base
    test_results = kb_builder.test_knowledge_base()
    
    logger.info("Knowledge base construction completed!")
    logger.info(f"Construction stats: {json.dumps(stats, indent=2)}")
    logger.info(f"Test results: {json.dumps(test_results, indent=2)}")

if __name__ == "__main__":
    main()
