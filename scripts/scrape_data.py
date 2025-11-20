#!/usr/bin/env python3
"""
DBS Data Collection Script
Phase 2: Data Collection - Web Scraping Infrastructure

This script implements comprehensive web scraping for DBS content including:
- Course information and descriptions
- Admissions requirements and processes
- Student support services
- General DBS information
- FAQ content

Author: DBS Chatbot Project
Date: October 2024
"""

import asyncio
import aiohttp
import logging
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import re
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Data class for scraped content"""
    url: str
    title: str
    content: str
    category: str
    subcategory: str
    metadata: Dict[str, Any]
    scraped_at: str
    content_hash: str

class DBSWebsiteScraper:
    """Main scraper class for DBS website content"""
    
    def __init__(self, base_url: str = "https://www.dbs.ie", delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay
        self.session = None
        self.scraped_urls = set()
        self.scraped_content = []
        
        # URL patterns for different content types
        self.url_patterns = {
            'courses': [
                '/courses/',
                '/undergraduate/',
                '/postgraduate/',
                '/part-time/',
                '/online/'
            ],
            'admissions': [
                '/admissions/',
                '/how-to-apply/',
                '/entry-requirements/',
                '/international/',
                '/mature-students/'
            ],
            'student_support': [
                '/student-support/',
                '/student-services/',
                '/library/',
                '/careers/',
                '/accommodation/'
            ],
            'about': [
                '/about/',
                '/our-story/',
                '/leadership/',
                '/contact/'
            ],
            'faq': [
                '/faq/',
                '/help/',
                '/support/'
            ]
        }
        
        # Content selectors for different page types
        self.content_selectors = {
            'main_content': [
                'main', 'article', '.content', '.main-content',
                '.page-content', '#content', '.entry-content'
            ],
            'title': [
                'h1', '.page-title', '.entry-title', 'title'
            ],
            'navigation': [
                'nav', '.navigation', '.menu', '.breadcrumb'
            ]
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'DBS-Chatbot/1.0 (Educational Purpose)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

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

    def extract_content_from_soup(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract structured content from BeautifulSoup object"""
        content_data = {
            'title': '',
            'content': '',
            'metadata': {}
        }
        
        # Extract title
        title_selectors = self.content_selectors['title']
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text(strip=True):
                content_data['title'] = self.clean_text(title_elem.get_text())
                break
        
        # Fallback to page title if no other title found
        if not content_data['title']:
            title_elem = soup.find('title')
            if title_elem:
                content_data['title'] = self.clean_text(title_elem.get_text())
        
        # Extract main content
        content_selectors = self.content_selectors['main_content']
        main_content = ""
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove navigation and footer elements
                for elem in content_elem.find_all(['nav', 'footer', 'aside', '.navigation', '.menu']):
                    elem.decompose()
                
                main_content = content_elem.get_text(separator=' ', strip=True)
                if main_content:
                    break
        
        # If no main content found, extract from body
        if not main_content:
            body = soup.find('body')
            if body:
                # Remove script and style elements
                for elem in body.find_all(['script', 'style', 'nav', 'footer', 'header']):
                    elem.decompose()
                main_content = body.get_text(separator=' ', strip=True)
        
        content_data['content'] = self.clean_text(main_content)
        
        # Extract metadata
        content_data['metadata'] = {
            'word_count': len(content_data['content'].split()),
            'has_images': len(soup.find_all('img')) > 0,
            'has_links': len(soup.find_all('a')) > 0,
            'has_tables': len(soup.find_all('table')) > 0,
            'last_modified': soup.find('meta', attrs={'name': 'last-modified'}) or 
                           soup.find('meta', attrs={'property': 'article:modified_time'})
        }
        
        return content_data

    def categorize_url(self, url: str) -> tuple[str, str]:
        """Categorize URL based on path patterns"""
        url_lower = url.lower()
        
        for category, patterns in self.url_patterns.items():
            for pattern in patterns:
                if pattern in url_lower:
                    # Determine subcategory based on URL structure
                    subcategory = self.get_subcategory(url, category)
                    return category, subcategory
        
        return 'general', 'other'

    def get_subcategory(self, url: str, category: str) -> str:
        """Get subcategory based on URL structure"""
        path_parts = urlparse(url).path.strip('/').split('/')
        
        if category == 'courses':
            if 'undergraduate' in url.lower():
                return 'undergraduate'
            elif 'postgraduate' in url.lower():
                return 'postgraduate'
            elif 'part-time' in url.lower():
                return 'part_time'
            elif 'online' in url.lower():
                return 'online'
            else:
                return 'general'
        
        elif category == 'admissions':
            if 'international' in url.lower():
                return 'international'
            elif 'mature' in url.lower():
                return 'mature_students'
            elif 'requirements' in url.lower():
                return 'requirements'
            else:
                return 'general'
        
        elif category == 'student_support':
            if 'library' in url.lower():
                return 'library'
            elif 'careers' in url.lower():
                return 'careers'
            elif 'accommodation' in url.lower():
                return 'accommodation'
            else:
                return 'general'
        
        return 'general'

    async def scrape_page(self, url: str) -> Optional[ScrapedContent]:
        """Scrape a single page and return structured content"""
        try:
            logger.info(f"Scraping: {url}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract content
                content_data = self.extract_content_from_soup(soup, url)
                
                if not content_data['content'] or len(content_data['content']) < 100:
                    logger.warning(f"Insufficient content from {url}")
                    return None
                
                # Categorize content
                category, subcategory = self.categorize_url(url)
                
                # Generate content hash for deduplication
                content_hash = self.generate_content_hash(content_data['content'])
                
                # Create scraped content object
                scraped_content = ScrapedContent(
                    url=url,
                    title=content_data['title'],
                    content=content_data['content'],
                    category=category,
                    subcategory=subcategory,
                    metadata=content_data['metadata'],
                    scraped_at=datetime.now().isoformat(),
                    content_hash=content_hash
                )
                
                logger.info(f"Successfully scraped {url}: {len(content_data['content'])} characters")
                return scraped_content
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def discover_urls(self, start_urls: List[str]) -> List[str]:
        """Discover URLs to scrape from starting URLs"""
        discovered_urls = set()
        
        # Add starting URLs
        for url in start_urls:
            discovered_urls.add(url)
        
        # Add comprehensive DBS pages
        common_pages = [
            # Main pages
            f"{self.base_url}/",
            f"{self.base_url}/about/",
            f"{self.base_url}/contact/",
            
            # Courses
            f"{self.base_url}/courses/",
            f"{self.base_url}/undergraduate/",
            f"{self.base_url}/postgraduate/",
            f"{self.base_url}/part-time/",
            f"{self.base_url}/online/",
            f"{self.base_url}/professional-diplomas/",
            f"{self.base_url}/springboard/",
            f"{self.base_url}/professional-accountancy/",
            
            # Admissions
            f"{self.base_url}/admissions/",
            f"{self.base_url}/entry-requirements/",
            f"{self.base_url}/how-to-apply/",
            f"{self.base_url}/international/",
            f"{self.base_url}/mature-students/",
            f"{self.base_url}/fees/",
            f"{self.base_url}/scholarships/",
            
            # Student Support
            f"{self.base_url}/student-support/",
            f"{self.base_url}/library/",
            f"{self.base_url}/careers/",
            f"{self.base_url}/accommodation/",
            f"{self.base_url}/student-life/",
            f"{self.base_url}/sports/",
            f"{self.base_url}/societies/",
            f"{self.base_url}/disability-support/",
            
            # Academic
            f"{self.base_url}/academic-school/",
            f"{self.base_url}/staff/",
            f"{self.base_url}/research/",
            f"{self.base_url}/alumni/",
            
            # Specific course categories
            f"{self.base_url}/courses/accounting-finance/",
            f"{self.base_url}/courses/arts/",
            f"{self.base_url}/courses/business-management/",
            f"{self.base_url}/courses/counselling-psychotherapy/",
            f"{self.base_url}/courses/information-technology/",
            f"{self.base_url}/courses/law/",
            f"{self.base_url}/courses/marketing-event-management/",
            f"{self.base_url}/courses/media-journalism/",
            f"{self.base_url}/courses/psychology-social-science/",
            
            # Additional important pages
            f"{self.base_url}/news/",
            f"{self.base_url}/events/",
            f"{self.base_url}/open-days/",
            f"{self.base_url}/prospectus/",
            f"{self.base_url}/virtual-tour/",
            f"{self.base_url}/campus/",
            f"{self.base_url}/facilities/",
            f"{self.base_url}/location/",
        ]
        
        for url in common_pages:
            discovered_urls.add(url)
        
        return list(discovered_urls)

    async def discover_urls_from_page(self, url: str, max_depth: int = 2) -> List[str]:
        """Discover URLs by crawling from a starting page"""
        discovered_urls = set()
        visited_urls = set()
        urls_to_visit = [url]
        current_depth = 0
        
        while urls_to_visit and current_depth < max_depth:
            current_batch = urls_to_visit.copy()
            urls_to_visit.clear()
            
            for current_url in current_batch:
                if current_url in visited_urls:
                    continue
                    
                visited_urls.add(current_url)
                
                try:
                    async with self.session.get(current_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract links
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                
                                # Convert relative URLs to absolute
                                if href.startswith('/'):
                                    full_url = f"{self.base_url}{href}"
                                elif href.startswith('http') and self.base_url in href:
                                    full_url = href
                                else:
                                    continue
                                
                                # Filter for DBS pages only
                                if (self.base_url in full_url and 
                                    not any(skip in full_url.lower() for skip in [
                                        'mailto:', 'tel:', '#', 'javascript:', 
                                        '.pdf', '.doc', '.docx', '.jpg', '.png', '.gif',
                                        'facebook.com', 'twitter.com', 'linkedin.com',
                                        'instagram.com', 'youtube.com'
                                    ])):
                                    discovered_urls.add(full_url)
                                    
                                    # Add to next depth if not too deep
                                    if current_depth < max_depth - 1:
                                        urls_to_visit.append(full_url)
                                        
                except Exception as e:
                    logger.warning(f"Error discovering URLs from {current_url}: {str(e)}")
                    continue
            
            current_depth += 1
            
        return list(discovered_urls)

    async def scrape_all(self, start_urls: List[str]) -> List[ScrapedContent]:
        """Scrape all discovered URLs"""
        # Get predefined URLs
        predefined_urls = self.discover_urls(start_urls)
        logger.info(f"Found {len(predefined_urls)} predefined URLs")
        
        # Discover additional URLs by crawling
        crawled_urls = set()
        for start_url in start_urls[:3]:  # Crawl from first 3 starting URLs
            try:
                new_urls = await self.discover_urls_from_page(start_url, max_depth=2)
                crawled_urls.update(new_urls)
                logger.info(f"Discovered {len(new_urls)} URLs from {start_url}")
            except Exception as e:
                logger.warning(f"Error crawling from {start_url}: {str(e)}")
        
        # Combine all URLs
        all_urls = set(predefined_urls) | crawled_urls
        discovered_urls = list(all_urls)
        logger.info(f"Total discovered URLs: {len(discovered_urls)}")
        
        scraped_content = []
        
        for i, url in enumerate(discovered_urls):
            if url in self.scraped_urls:
                continue
                
            self.scraped_urls.add(url)
            
            # Rate limiting
            if i > 0:
                await asyncio.sleep(self.delay)
            
            content = await self.scrape_page(url)
            if content:
                scraped_content.append(content)
                self.scraped_content.append(content)
        
        logger.info(f"Scraped {len(scraped_content)} pages successfully")
        return scraped_content

    def save_scraped_data(self, output_file: str = "data/raw/scraped_content.json"):
        """Save scraped data to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = [asdict(content) for content in self.scraped_content]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} scraped pages to {output_file}")

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of scraped data"""
        if not self.scraped_content:
            return {"error": "No content scraped"}
        
        # Count by category
        category_counts = {}
        subcategory_counts = {}
        total_words = 0
        
        for content in self.scraped_content:
            category = content.category
            subcategory = content.subcategory
            
            category_counts[category] = category_counts.get(category, 0) + 1
            subcategory_counts[f"{category}_{subcategory}"] = subcategory_counts.get(f"{category}_{subcategory}", 0) + 1
            
            total_words += len(content.content.split())
        
        return {
            "total_pages": len(self.scraped_content),
            "total_words": total_words,
            "category_distribution": category_counts,
            "subcategory_distribution": subcategory_counts,
            "average_words_per_page": total_words / len(self.scraped_content),
            "scraping_timestamp": datetime.now().isoformat()
        }

async def main():
    """Main function to run the scraper"""
    logger.info("Starting DBS website scraping...")
    
    # Starting URLs for DBS content
    start_urls = [
        "https://www.dbs.ie/",
        "https://www.dbs.ie/courses/",
        "https://www.dbs.ie/admissions/",
        "https://www.dbs.ie/student-support/",
        "https://www.dbs.ie/about/",
    ]
    
    async with DBSWebsiteScraper() as scraper:
        # Scrape all content
        scraped_content = await scraper.scrape_all(start_urls)
        
        # Save data
        scraper.save_scraped_data()
        
        # Generate report
        report = scraper.generate_summary_report()
        
        # Save report
        with open("data/raw/scraping_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Scraping completed successfully!")
        logger.info(f"Summary: {json.dumps(report, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
