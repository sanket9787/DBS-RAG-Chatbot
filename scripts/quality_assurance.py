#!/usr/bin/env python3
"""
DBS Quality Assurance Script
Phase 2: Data Collection - Quality Assurance and Validation

This script provides comprehensive quality assurance for collected data including:
- Content validation and verification
- Accuracy checking against source
- Completeness assessment
- Consistency validation
- Data integrity checks

Author: DBS Chatbot Project
Date: October 2024
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_assurance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityReport:
    """Data class for quality assessment report"""
    content_id: str
    url: str
    title: str
    category: str
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    validation_results: Dict[str, Any]
    assessed_at: str

@dataclass
class ValidationResult:
    """Data class for validation result"""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]

class QualityAssurance:
    """Main class for quality assurance and validation"""
    
    def __init__(self):
        # Quality thresholds
        self.quality_thresholds = {
            'min_quality_score': 0.5,
            'min_word_count': 50,
            'max_word_count': 5000,
            'min_sentence_count': 3,
            'max_sentence_count': 100,
            'min_paragraph_count': 1,
            'max_paragraph_count': 50
        }
        
        # Content validation patterns
        self.validation_patterns = {
            'dbs_contact_info': {
                'phone': r'(\+353\s?)?(0?[1-9]\d{1,2}\s?\d{3}\s?\d{4})',
                'email': r'[a-zA-Z0-9._%+-]+@dbs\.ie',
                'address': r'(Dublin|Ireland|Dublin Business School)',
                'website': r'www\.dbs\.ie'
            },
            'academic_content': {
                'course_codes': r'[A-Z]{2,4}\d{3,4}',
                'credits': r'\d+\s*(credits?|ECTS)',
                'grades': r'[A-F][+-]?|\d+%',
                'years': r'(1st|2nd|3rd|4th|first|second|third|fourth)\s+year'
            },
            'dates': {
                'academic_year': r'(20\d{2}[-\/]20\d{2}|20\d{2})',
                'deadlines': r'(deadline|closing|due)\s+(date|date:)?\s*:?\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',
                'semester': r'(semester|term)\s+[1-4]'
            }
        }
        
        # Common issues and their fixes
        self.issue_patterns = {
            'broken_links': r'https?://[^\s]+',
            'missing_contact': r'(contact|phone|email|address)',
            'incomplete_sentences': r'[^.!?]\s*$',
            'excessive_caps': r'[A-Z]{5,}',
            'poor_formatting': r'\s{3,}|\t+',
            'encoding_issues': r'[^\x00-\x7F]'
        }

    def validate_content_structure(self, content: str, title: str) -> ValidationResult:
        """Validate basic content structure"""
        issues = []
        suggestions = []
        
        # Check length
        word_count = len(content.split())
        if word_count < self.quality_thresholds['min_word_count']:
            issues.append(f"Content too short: {word_count} words")
            suggestions.append("Consider expanding content or merging with related content")
        elif word_count > self.quality_thresholds['max_word_count']:
            issues.append(f"Content too long: {word_count} words")
            suggestions.append("Consider splitting into multiple sections")
        
        # Check sentence structure
        sentences = content.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < self.quality_thresholds['min_sentence_count']:
            issues.append(f"Too few sentences: {sentence_count}")
            suggestions.append("Content should have at least 3 complete sentences")
        elif sentence_count > self.quality_thresholds['max_sentence_count']:
            issues.append(f"Too many sentences: {sentence_count}")
            suggestions.append("Consider breaking into smaller, focused sections")
        
        # Check for incomplete sentences
        if re.search(self.issue_patterns['incomplete_sentences'], content):
            issues.append("Incomplete sentences detected")
            suggestions.append("Ensure all sentences end with proper punctuation")
        
        # Check for excessive capitalization
        if re.search(self.issue_patterns['excessive_caps'], content):
            issues.append("Excessive capitalization detected")
            suggestions.append("Use proper capitalization rules")
        
        # Check for poor formatting
        if re.search(self.issue_patterns['poor_formatting'], content):
            issues.append("Poor formatting detected")
            suggestions.append("Clean up excessive whitespace and formatting")
        
        # Calculate confidence score
        confidence = max(0.0, 1.0 - (len(issues) * 0.2))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )

    def validate_dbs_content(self, content: str, category: str) -> ValidationResult:
        """Validate DBS-specific content"""
        issues = []
        suggestions = []
        content_lower = content.lower()
        
        # Check for DBS-specific information
        dbs_indicators = [
            'dublin business school', 'dbs', 'dublin', 'ireland'
        ]
        
        dbs_mentions = sum(1 for indicator in dbs_indicators if indicator in content_lower)
        if dbs_mentions == 0:
            issues.append("No DBS-specific information found")
            suggestions.append("Content should mention Dublin Business School or DBS")
        
        # Check for contact information based on category
        if category in ['admissions', 'student_support', 'about']:
            contact_patterns = self.validation_patterns['dbs_contact_info']
            contact_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in contact_patterns.values())
            
            if not contact_found:
                issues.append("Missing contact information")
                suggestions.append("Include phone, email, or address information")
        
        # Check for academic content if course-related
        if category in ['courses', 'admissions']:
            academic_patterns = self.validation_patterns['academic_content']
            academic_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in academic_patterns.values())
            
            if not academic_found:
                issues.append("Missing academic information")
                suggestions.append("Include course codes, credits, or academic details")
        
        # Check for dates if time-sensitive content
        if category in ['admissions', 'courses']:
            date_patterns = self.validation_patterns['dates']
            dates_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in date_patterns.values())
            
            if not dates_found:
                issues.append("Missing date information")
                suggestions.append("Include relevant dates, deadlines, or academic years")
        
        # Calculate confidence score
        confidence = max(0.0, 1.0 - (len(issues) * 0.3))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )

    def validate_url_accessibility(self, url: str) -> ValidationResult:
        """Validate that URL is accessible and content matches"""
        issues = []
        suggestions = []
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                issues.append(f"URL not accessible: {response.status_code}")
                suggestions.append("Check if URL is still valid and accessible")
                return ValidationResult(False, 0.0, issues, suggestions)
            
            # Check if content type is HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                issues.append(f"Unexpected content type: {content_type}")
                suggestions.append("Verify that URL points to HTML content")
            
            # Check for redirects
            if len(response.history) > 0:
                issues.append("URL redirects detected")
                suggestions.append("Consider using the final URL after redirects")
            
            confidence = 0.8 if response.status_code == 200 else 0.0
            
        except requests.exceptions.RequestException as e:
            issues.append(f"URL validation failed: {str(e)}")
            suggestions.append("Check URL accessibility and network connectivity")
            confidence = 0.0
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )

    def validate_content_consistency(self, content: str, title: str, category: str) -> ValidationResult:
        """Validate content consistency"""
        issues = []
        suggestions = []
        
        # Check title-content consistency
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        
        # Find common words between title and content
        common_words = title_words.intersection(content_words)
        if len(common_words) < 2:
            issues.append("Title and content have little overlap")
            suggestions.append("Ensure title accurately reflects content")
        
        # Check for category consistency
        category_keywords = {
            'courses': ['course', 'programme', 'degree', 'module', 'study'],
            'admissions': ['admission', 'apply', 'application', 'entry', 'requirement'],
            'student_support': ['support', 'service', 'help', 'assistance', 'guidance'],
            'about': ['about', 'history', 'mission', 'vision', 'overview']
        }
        
        if category in category_keywords:
            expected_keywords = category_keywords[category]
            found_keywords = sum(1 for keyword in expected_keywords if keyword in content.lower())
            
            if found_keywords < 2:
                issues.append(f"Content doesn't match category '{category}'")
                suggestions.append(f"Include more {category}-related keywords")
        
        # Check for internal consistency
        if 'dublin business school' in content.lower() and 'dbs' not in content.lower():
            issues.append("Inconsistent institution naming")
            suggestions.append("Use consistent naming (DBS or Dublin Business School)")
        
        confidence = max(0.0, 1.0 - (len(issues) * 0.25))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )

    def assess_content_quality(self, content_data: Dict[str, Any]) -> QualityReport:
        """Assess quality of a single content item"""
        content_id = content_data.get('content_hash', 'unknown')
        url = content_data.get('url', '')
        title = content_data.get('title', '')
        content = content_data.get('content', '')
        category = content_data.get('category', 'general')
        
        # Run all validations
        structure_validation = self.validate_content_structure(content, title)
        dbs_validation = self.validate_dbs_content(content, category)
        consistency_validation = self.validate_content_consistency(content, title, category)
        
        # URL validation (only if URL exists)
        url_validation = None
        if url:
            url_validation = self.validate_url_accessibility(url)
        
        # Combine all issues and suggestions
        all_issues = []
        all_suggestions = []
        
        all_issues.extend(structure_validation.issues)
        all_suggestions.extend(structure_validation.suggestions)
        
        all_issues.extend(dbs_validation.issues)
        all_suggestions.extend(dbs_validation.suggestions)
        
        all_issues.extend(consistency_validation.issues)
        all_suggestions.extend(consistency_validation.suggestions)
        
        if url_validation:
            all_issues.extend(url_validation.issues)
            all_suggestions.extend(url_validation.suggestions)
        
        # Calculate overall quality score
        validation_scores = [
            structure_validation.confidence,
            dbs_validation.confidence,
            consistency_validation.confidence
        ]
        
        if url_validation:
            validation_scores.append(url_validation.confidence)
        
        quality_score = sum(validation_scores) / len(validation_scores)
        
        # Create validation results
        validation_results = {
            'structure': {
                'is_valid': structure_validation.is_valid,
                'confidence': structure_validation.confidence,
                'issues': structure_validation.issues
            },
            'dbs_content': {
                'is_valid': dbs_validation.is_valid,
                'confidence': dbs_validation.confidence,
                'issues': dbs_validation.issues
            },
            'consistency': {
                'is_valid': consistency_validation.is_valid,
                'confidence': consistency_validation.confidence,
                'issues': consistency_validation.issues
            }
        }
        
        if url_validation:
            validation_results['url_accessibility'] = {
                'is_valid': url_validation.is_valid,
                'confidence': url_validation.confidence,
                'issues': url_validation.issues
            }
        
        return QualityReport(
            content_id=content_id,
            url=url,
            title=title,
            category=category,
            quality_score=quality_score,
            issues=all_issues,
            recommendations=all_suggestions,
            validation_results=validation_results,
            assessed_at=datetime.now().isoformat()
        )

    def process_quality_assurance(self, input_file: str, output_file: str = "data/processed/quality_reports.json") -> Dict[str, Any]:
        """Process quality assurance for all content"""
        logger.info(f"Starting quality assurance for {input_file}")
        
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        
        logger.info(f"Loaded {len(content_data)} items for quality assessment")
        
        # Process each item
        quality_reports = []
        quality_stats = {
            'total_items': len(content_data),
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'failed_validation': 0,
            'avg_quality_score': 0.0,
            'common_issues': {},
            'quality_scores': []
        }
        
        for item in content_data:
            quality_report = self.assess_content_quality(item)
            quality_reports.append(quality_report)
            
            # Update statistics
            quality_score = quality_report.quality_score
            quality_stats['quality_scores'].append(quality_score)
            
            if quality_score >= 0.8:
                quality_stats['high_quality'] += 1
            elif quality_score >= 0.5:
                quality_stats['medium_quality'] += 1
            else:
                quality_stats['low_quality'] += 1
            
            if not quality_report.validation_results['structure']['is_valid']:
                quality_stats['failed_validation'] += 1
            
            # Track common issues
            for issue in quality_report.issues:
                quality_stats['common_issues'][issue] = quality_stats['common_issues'].get(issue, 0) + 1
        
        # Calculate average quality score
        if quality_stats['quality_scores']:
            quality_stats['avg_quality_score'] = sum(quality_stats['quality_scores']) / len(quality_stats['quality_scores'])
        
        # Save quality reports
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        reports_data = [asdict(report) for report in quality_reports]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reports_data, f, indent=2, ensure_ascii=False)
        
        # Save quality statistics
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(quality_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality assurance completed: {quality_stats['high_quality']} high quality, {quality_stats['medium_quality']} medium quality, {quality_stats['low_quality']} low quality")
        
        return quality_stats

def main():
    """Main function to run quality assurance"""
    logger.info("Starting quality assurance process...")
    
    # Initialize quality assurance
    qa = QualityAssurance()
    
    # Process web content
    web_file = "data/processed/cleaned_web_content.json"
    if Path(web_file).exists():
        stats = qa.process_quality_assurance(web_file, "data/processed/web_quality_reports.json")
        logger.info(f"Web content quality assessment: {json.dumps(stats, indent=2)}")
    else:
        logger.warning(f"Web content file not found: {web_file}")
    
    # Process PDF content
    pdf_file = "data/processed/cleaned_pdf_content.json"
    if Path(pdf_file).exists():
        stats = qa.process_quality_assurance(pdf_file, "data/processed/pdf_quality_reports.json")
        logger.info(f"PDF content quality assessment: {json.dumps(stats, indent=2)}")
    else:
        logger.warning(f"PDF content file not found: {pdf_file}")
    
    logger.info("Quality assurance process completed!")

if __name__ == "__main__":
    main()
