#!/usr/bin/env python3
"""
DBS Data Cleaning and Preprocessing Script
Phase 2: Data Collection - Data Cleaning and Preprocessing

This script cleans and preprocesses scraped content including:
- Text normalization and cleaning
- Duplicate detection and removal
- Content quality assessment
- Data validation and enrichment
- Chunking for RAG pipeline

Author: DBS Chatbot Project
Date: October 2024
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import unicodedata

# Text processing libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CleanedContent:
    """Data class for cleaned content"""
    original_id: str
    url: str
    title: str
    content: str
    category: str
    subcategory: str
    metadata: Dict[str, Any]
    quality_score: float
    cleaned_at: str
    content_hash: str

@dataclass
class ContentChunk:
    """Data class for content chunks"""
    chunk_id: str
    parent_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int
    created_at: str

class DataCleaner:
    """Main class for cleaning and preprocessing data"""
    
    def __init__(self):
        # Initialize NLTK
        self._setup_nltk()
        
        # Text processing tools
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Content quality thresholds
        self.quality_thresholds = {
            'min_length': 100,
            'max_length': 10000,
            'min_word_count': 20,
            'max_word_count': 2000,
            'min_sentence_count': 3,
            'max_sentence_count': 200
        }
        
        # Minimum quality score threshold (lowered for initial data collection)
        self.min_quality_score = 0.2
        
        # Duplicate detection
        self.content_hashes = set()
        
        # Content patterns for validation
        self.valid_patterns = {
            'dbs_related': [
                'dublin business school', 'dbs', 'course', 'programme',
                'admission', 'student', 'university', 'college', 'education'
            ],
            'contact_info': [
                'phone', 'email', 'address', 'contact', 'location',
                'telephone', 'mobile', 'fax'
            ],
            'academic_terms': [
                'degree', 'diploma', 'certificate', 'bachelor', 'master',
                'phd', 'undergraduate', 'postgraduate', 'module', 'assessment'
            ]
        }

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
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

    def normalize_text(self, text: str) -> str:
        """Normalize text content"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', '', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€¢', '•')
        text = text.replace('â€"', '–')
        text = text.replace('â€"', '—')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def clean_title(self, title: str) -> str:
        """Clean and normalize title"""
        if not title:
            return "Untitled"
        
        # Normalize text
        cleaned_title = self.normalize_text(title)
        
        # Remove common prefixes/suffixes
        cleaned_title = re.sub(r'^(DBS|Dublin Business School)\s*[-:]\s*', '', cleaned_title, flags=re.IGNORECASE)
        cleaned_title = re.sub(r'\s*[-:]\s*(DBS|Dublin Business School)$', '', cleaned_title, flags=re.IGNORECASE)
        
        # Capitalize properly
        cleaned_title = cleaned_title.title()
        
        return cleaned_title

    def assess_content_quality(self, content: str, title: str, category: str) -> Dict[str, Any]:
        """Assess content quality and return quality metrics"""
        if not content:
            return {
                'quality_score': 0.0,
                'issues': ['Empty content'],
                'metrics': {}
            }
        
        # Basic metrics
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = len(sent_tokenize(content))
        
        # Calculate quality score (0-1)
        quality_score = 0.0
        issues = []
        
        # Length checks
        if word_count < self.quality_thresholds['min_word_count']:
            issues.append(f"Too short: {word_count} words")
            quality_score -= 0.3
        elif word_count > self.quality_thresholds['max_word_count']:
            issues.append(f"Too long: {word_count} words")
            quality_score -= 0.1
        
        # Sentence structure
        if sentence_count < self.quality_thresholds['min_sentence_count']:
            issues.append(f"Too few sentences: {sentence_count}")
            quality_score -= 0.2
        elif sentence_count > self.quality_thresholds['max_sentence_count']:
            issues.append(f"Too many sentences: {sentence_count}")
            quality_score -= 0.1
        
        # Content relevance
        content_lower = content.lower()
        relevance_score = 0.0
        
        # Check for DBS-related terms
        dbs_terms = self.valid_patterns['dbs_related']
        dbs_matches = sum(1 for term in dbs_terms if term in content_lower)
        relevance_score += min(dbs_matches / len(dbs_terms), 1.0) * 0.3
        
        # Check for academic terms
        academic_terms = self.valid_patterns['academic_terms']
        academic_matches = sum(1 for term in academic_terms if term in content_lower)
        relevance_score += min(academic_matches / len(academic_terms), 1.0) * 0.2
        
        # Check for contact information
        contact_terms = self.valid_patterns['contact_info']
        contact_matches = sum(1 for term in contact_terms if term in content_lower)
        relevance_score += min(contact_matches / len(contact_terms), 1.0) * 0.1
        
        quality_score += relevance_score
        
        # Text structure analysis
        if re.search(r'[.!?]{2,}', content):  # Multiple punctuation
            issues.append("Poor punctuation")
            quality_score -= 0.1
        
        if re.search(r'[A-Z]{5,}', content):  # Excessive caps
            issues.append("Excessive capitalization")
            quality_score -= 0.1
        
        # Ensure quality score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'metrics': {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'relevance_score': relevance_score,
                'avg_words_per_sentence': word_count / max(sentence_count, 1)
            }
        }

    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash in self.content_hashes:
            return True
        
        self.content_hashes.add(content_hash)
        return False

    def clean_content(self, content_data: Dict[str, Any]) -> Optional[CleanedContent]:
        """Clean a single content item"""
        try:
            # Extract basic information
            url = content_data.get('url', '')
            title = content_data.get('title', '')
            content = content_data.get('content', '')
            category = content_data.get('category', 'general')
            subcategory = content_data.get('subcategory', 'general')
            metadata = content_data.get('metadata', {})
            original_id = content_data.get('content_hash', '')
            
            # Normalize content
            cleaned_content = self.normalize_text(content)
            cleaned_title = self.clean_title(title)
            
            # Check for duplicates
            if self.is_duplicate(cleaned_content):
                logger.warning(f"Duplicate content detected: {url}")
                return None
            
            # Assess quality
            quality_assessment = self.assess_content_quality(cleaned_content, cleaned_title, category)
            
            # Skip low-quality content
            if quality_assessment['quality_score'] < self.min_quality_score:
                logger.warning(f"Low quality content skipped: {url} (score: {quality_assessment['quality_score']:.2f})")
                return None
            
            # Update metadata
            updated_metadata = {
                **metadata,
                'original_word_count': len(content.split()),
                'cleaned_word_count': len(cleaned_content.split()),
                'quality_assessment': quality_assessment,
                'cleaning_timestamp': datetime.now().isoformat()
            }
            
            # Generate new content hash
            content_hash = hashlib.md5(cleaned_content.encode('utf-8')).hexdigest()
            
            # Create cleaned content object
            cleaned_content_obj = CleanedContent(
                original_id=original_id,
                url=url,
                title=cleaned_title,
                content=cleaned_content,
                category=category,
                subcategory=subcategory,
                metadata=updated_metadata,
                quality_score=quality_assessment['quality_score'],
                cleaned_at=datetime.now().isoformat(),
                content_hash=content_hash
            )
            
            logger.info(f"Successfully cleaned content: {url} (quality: {quality_assessment['quality_score']:.2f})")
            return cleaned_content_obj
            
        except Exception as e:
            logger.error(f"Error cleaning content: {str(e)}")
            return None

    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split content into chunks for RAG pipeline"""
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
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split('.')[-overlap//50:]  # Approximate overlap
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                current_length = len(current_chunk.split())
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def create_content_chunks(self, cleaned_content: CleanedContent) -> List[ContentChunk]:
        """Create chunks from cleaned content"""
        chunks = self.chunk_content(cleaned_content.content)
        
        content_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_metadata = {
                'parent_url': cleaned_content.url,
                'parent_title': cleaned_content.title,
                'parent_category': cleaned_content.category,
                'parent_subcategory': cleaned_content.subcategory,
                'chunk_word_count': len(chunk_content.split()),
                'chunk_char_count': len(chunk_content),
                'quality_score': cleaned_content.quality_score
            }
            
            chunk_id = f"{cleaned_content.content_hash}_{i}"
            
            chunk = ContentChunk(
                chunk_id=chunk_id,
                parent_id=cleaned_content.content_hash,
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_index=i,
                total_chunks=len(chunks),
                created_at=datetime.now().isoformat()
            )
            
            content_chunks.append(chunk)
        
        return content_chunks

    def process_data(self, input_file: str, output_file: str = "data/processed/cleaned_content.json") -> Dict[str, Any]:
        """Process all data from input file"""
        logger.info(f"Processing data from {input_file}")
        
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"Loaded {len(raw_data)} items for processing")
        
        # Process each item
        cleaned_items = []
        all_chunks = []
        processing_stats = {
            'total_items': len(raw_data),
            'cleaned_items': 0,
            'skipped_items': 0,
            'total_chunks': 0,
            'quality_scores': []
        }
        
        for item in raw_data:
            cleaned_item = self.clean_content(item)
            
            if cleaned_item:
                cleaned_items.append(cleaned_item)
                processing_stats['cleaned_items'] += 1
                processing_stats['quality_scores'].append(cleaned_item.quality_score)
                
                # Create chunks
                chunks = self.create_content_chunks(cleaned_item)
                all_chunks.extend(chunks)
                processing_stats['total_chunks'] += len(chunks)
            else:
                processing_stats['skipped_items'] += 1
        
        # Save cleaned data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cleaned_data = [asdict(item) for item in cleaned_items]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Save chunks
        chunks_file = output_file.replace('.json', '_chunks.json')
        chunks_data = [asdict(chunk) for chunk in all_chunks]
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Calculate final statistics
        if processing_stats['quality_scores']:
            processing_stats['avg_quality_score'] = sum(processing_stats['quality_scores']) / len(processing_stats['quality_scores'])
            processing_stats['min_quality_score'] = min(processing_stats['quality_scores'])
            processing_stats['max_quality_score'] = max(processing_stats['quality_scores'])
        
        processing_stats['cleaning_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Processing completed: {processing_stats['cleaned_items']} items cleaned, {processing_stats['total_chunks']} chunks created")
        
        return processing_stats

def main():
    """Main function to run data cleaning"""
    logger.info("Starting data cleaning and preprocessing...")
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Process scraped data
    scraped_file = "data/raw/scraped_content.json"
    if Path(scraped_file).exists():
        stats = cleaner.process_data(scraped_file, "data/processed/cleaned_web_content.json")
        
        # Save processing report
        with open("data/processed/cleaning_report.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Web content cleaning completed: {json.dumps(stats, indent=2)}")
    else:
        logger.warning(f"Scraped data file not found: {scraped_file}")
    
    # Process PDF data
    pdf_file = "data/processed/processed_pdfs.json"
    if Path(pdf_file).exists():
        stats = cleaner.process_data(pdf_file, "data/processed/cleaned_pdf_content.json")
        
        # Save processing report
        with open("data/processed/pdf_cleaning_report.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"PDF content cleaning completed: {json.dumps(stats, indent=2)}")
    else:
        logger.warning(f"PDF data file not found: {pdf_file}")
    
    logger.info("Data cleaning and preprocessing completed!")

if __name__ == "__main__":
    main()
