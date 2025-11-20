#!/usr/bin/env python3
"""
DBS PDF Processing Script
Phase 2: Data Collection - PDF Document Processing

This script processes PDF documents from DBS including:
- Course prospectuses and handbooks
- Student guides and manuals
- Policy documents and procedures
- Application forms and templates

Author: DBS Chatbot Project
Date: October 2024
"""

import os
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

# PDF processing libraries
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedPDF:
    """Data class for processed PDF content"""
    file_path: str
    file_name: str
    title: str
    content: str
    pages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    category: str
    subcategory: str
    processed_at: str
    content_hash: str

class PDFProcessor:
    """Main class for processing PDF documents"""
    
    def __init__(self, input_dir: str = "data/raw/pdfs", output_dir: str = "data/processed/pdfs"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLTK (download required data if not present)
        self._setup_nltk()
        
        # PDF processing methods
        self.extractors = {
            'pypdf2': self._extract_with_pypdf2,
            'pdfplumber': self._extract_with_pdfplumber,
            'pdfminer': self._extract_with_pdfminer
        }
        
        # Content categorization patterns
        self.category_patterns = {
            'courses': [
                'course', 'programme', 'degree', 'diploma', 'certificate',
                'undergraduate', 'postgraduate', 'bachelor', 'master', 'phd'
            ],
            'admissions': [
                'admission', 'application', 'entry', 'requirement', 'deadline',
                'apply', 'enroll', 'registration', 'fee', 'tuition'
            ],
            'student_support': [
                'student', 'support', 'service', 'library', 'career', 'accommodation',
                'counseling', 'advice', 'help', 'guidance', 'assistance'
            ],
            'policies': [
                'policy', 'procedure', 'regulation', 'rule', 'guideline',
                'terms', 'condition', 'agreement', 'contract'
            ],
            'general': [
                'handbook', 'guide', 'manual', 'brochure', 'prospectus',
                'information', 'about', 'overview', 'introduction'
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
        
        self.stop_words = set(stopwords.words('english'))

    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def extract_title_from_content(self, content: str, file_name: str) -> str:
        """Extract title from content or use filename"""
        # Try to find title in first few lines
        lines = content.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Check if it looks like a title
                if any(word in line.lower() for word in ['handbook', 'guide', 'prospectus', 'manual']):
                    return line
        
        # Fallback to filename without extension
        return Path(file_name).stem.replace('_', ' ').replace('-', ' ').title()

    def categorize_content(self, content: str, file_name: str) -> Tuple[str, str]:
        """Categorize content based on text analysis"""
        content_lower = content.lower()
        file_lower = file_name.lower()
        
        # Score each category
        category_scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                # Count pattern occurrences in content
                content_matches = len(re.findall(pattern, content_lower))
                # Count pattern occurrences in filename
                file_matches = len(re.findall(pattern, file_lower))
                # Weight filename matches more heavily
                score += content_matches + (file_matches * 2)
            
            category_scores[category] = score
        
        # Get category with highest score
        best_category = max(category_scores, key=category_scores.get)
        
        # Determine subcategory
        subcategory = self.get_subcategory(content_lower, file_lower, best_category)
        
        return best_category, subcategory

    def get_subcategory(self, content: str, file_name: str, category: str) -> str:
        """Get subcategory based on content analysis"""
        if category == 'courses':
            if any(word in content for word in ['undergraduate', 'bachelor', 'degree']):
                return 'undergraduate'
            elif any(word in content for word in ['postgraduate', 'master', 'phd', 'masters']):
                return 'postgraduate'
            elif any(word in content for word in ['part-time', 'part time', 'evening']):
                return 'part_time'
            elif any(word in content for word in ['online', 'distance', 'remote']):
                return 'online'
            else:
                return 'general'
        
        elif category == 'admissions':
            if any(word in content for word in ['international', 'overseas', 'visa']):
                return 'international'
            elif any(word in content for word in ['mature', 'adult', 'returning']):
                return 'mature_students'
            elif any(word in content for word in ['requirement', 'prerequisite', 'qualification']):
                return 'requirements'
            else:
                return 'general'
        
        elif category == 'student_support':
            if any(word in content for word in ['library', 'book', 'resource', 'database']):
                return 'library'
            elif any(word in content for word in ['career', 'job', 'employment', 'placement']):
                return 'careers'
            elif any(word in content for word in ['accommodation', 'housing', 'residence', 'dormitory']):
                return 'accommodation'
            else:
                return 'general'
        
        return 'general'

    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text using PyPDF2"""
        try:
            text = ""
            pages = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        pages.append({
                            'page_number': page_num + 1,
                            'content': page_text,
                            'word_count': len(page_text.split())
                        })
            
            return text, pages
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path}: {str(e)}")
            return "", []

    def _extract_with_pdfplumber(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text using pdfplumber"""
        try:
            text = ""
            pages = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        pages.append({
                            'page_number': page_num + 1,
                            'content': page_text,
                            'word_count': len(page_text.split())
                        })
            
            return text, pages
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path}: {str(e)}")
            return "", []

    def _extract_with_pdfminer(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text using pdfminer"""
        try:
            text = extract_text(str(file_path), laparams=LAParams())
            pages = [{
                'page_number': 1,
                'content': text,
                'word_count': len(text.split())
            }]
            
            return text, pages
        except Exception as e:
            logger.error(f"pdfminer extraction failed for {file_path}: {str(e)}")
            return "", []

    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text from PDF using multiple methods"""
        logger.info(f"Processing PDF: {file_path.name}")
        
        # Try different extraction methods
        for method_name, extractor in self.extractors.items():
            try:
                text, pages = extractor(file_path)
                if text and len(text.strip()) > 100:  # Minimum content threshold
                    logger.info(f"Successfully extracted text using {method_name}")
                    return text, pages
            except Exception as e:
                logger.warning(f"{method_name} failed for {file_path.name}: {str(e)}")
                continue
        
        logger.error(f"All extraction methods failed for {file_path.name}")
        return "", []

    def process_pdf(self, file_path: Path) -> Optional[ProcessedPDF]:
        """Process a single PDF file"""
        try:
            # Extract text and pages
            text, pages = self.extract_text_from_pdf(file_path)
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Insufficient content extracted from {file_path.name}")
                return None
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Extract title
            title = self.extract_title_from_content(cleaned_text, file_path.name)
            
            # Categorize content
            category, subcategory = self.categorize_content(cleaned_text, file_path.name)
            
            # Generate metadata
            metadata = {
                'file_size': file_path.stat().st_size,
                'page_count': len(pages),
                'word_count': len(cleaned_text.split()),
                'extraction_method': 'multiple',
                'has_images': False,  # Could be enhanced to detect images
                'has_tables': any('table' in page.get('content', '').lower() for page in pages),
                'language': 'en',  # Could be enhanced to detect language
                'processing_quality': 'high' if len(cleaned_text) > 1000 else 'medium'
            }
            
            # Generate content hash
            content_hash = self.generate_content_hash(cleaned_text)
            
            # Create processed PDF object
            processed_pdf = ProcessedPDF(
                file_path=str(file_path),
                file_name=file_path.name,
                title=title,
                content=cleaned_text,
                pages=pages,
                metadata=metadata,
                category=category,
                subcategory=subcategory,
                processed_at=datetime.now().isoformat(),
                content_hash=content_hash
            )
            
            logger.info(f"Successfully processed {file_path.name}: {len(cleaned_text)} characters")
            return processed_pdf
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            return None

    def find_pdf_files(self) -> List[Path]:
        """Find all PDF files in input directory"""
        pdf_files = []
        
        if not self.input_dir.exists():
            logger.warning(f"Input directory {self.input_dir} does not exist")
            return pdf_files
        
        # Find PDF files recursively
        for pattern in ['**/*.pdf', '**/*.PDF']:
            pdf_files.extend(self.input_dir.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return pdf_files

    def process_all_pdfs(self) -> List[ProcessedPDF]:
        """Process all PDF files in input directory"""
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return []
        
        processed_pdfs = []
        
        for pdf_file in pdf_files:
            processed_pdf = self.process_pdf(pdf_file)
            if processed_pdf:
                processed_pdfs.append(processed_pdf)
        
        logger.info(f"Successfully processed {len(processed_pdfs)} out of {len(pdf_files)} PDF files")
        return processed_pdfs

    def save_processed_data(self, processed_pdfs: List[ProcessedPDF], output_file: str = "data/processed/processed_pdfs.json"):
        """Save processed PDF data to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = [asdict(pdf) for pdf in processed_pdfs]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} processed PDFs to {output_file}")

    def generate_summary_report(self, processed_pdfs: List[ProcessedPDF]) -> Dict[str, Any]:
        """Generate summary report of processed PDFs"""
        if not processed_pdfs:
            return {"error": "No PDFs processed"}
        
        # Count by category
        category_counts = {}
        subcategory_counts = {}
        total_words = 0
        total_pages = 0
        
        for pdf in processed_pdfs:
            category = pdf.category
            subcategory = pdf.subcategory
            
            category_counts[category] = category_counts.get(category, 0) + 1
            subcategory_counts[f"{category}_{subcategory}"] = subcategory_counts.get(f"{category}_{subcategory}", 0) + 1
            
            total_words += pdf.metadata['word_count']
            total_pages += pdf.metadata['page_count']
        
        return {
            "total_pdfs": len(processed_pdfs),
            "total_pages": total_pages,
            "total_words": total_words,
            "category_distribution": category_counts,
            "subcategory_distribution": subcategory_counts,
            "average_words_per_pdf": total_words / len(processed_pdfs),
            "average_pages_per_pdf": total_pages / len(processed_pdfs),
            "processing_timestamp": datetime.now().isoformat()
        }

def main():
    """Main function to run PDF processing"""
    logger.info("Starting PDF processing...")
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Process all PDFs
    processed_pdfs = processor.process_all_pdfs()
    
    if processed_pdfs:
        # Save processed data
        processor.save_processed_data(processed_pdfs)
        
        # Generate report
        report = processor.generate_summary_report(processed_pdfs)
        
        # Save report
        with open("data/processed/pdf_processing_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("PDF processing completed successfully!")
        logger.info(f"Summary: {json.dumps(report, indent=2)}")
    else:
        logger.warning("No PDFs were processed successfully")

if __name__ == "__main__":
    main()
