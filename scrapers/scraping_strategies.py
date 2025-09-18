"""
Multi-strategy scraping system for educational PDF collection.
Provides different scraping approaches for various website types and content delivery methods.
"""

import asyncio
import logging
import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import aiohttp
from bs4 import BeautifulSoup

from data.models import SourceConfig, PDFMetadata, ScrapingStrategy


class ScrapingStrategyBase(ABC):
    """
    Abstract base class for all scraping strategies.
    Defines the interface that all concrete strategies must implement.
    """
    
    def __init__(self, source_config: SourceConfig):
        """
        Initialize the scraping strategy.
        
        Args:
            source_config: Configuration for the source being scraped
        """
        self.source_config = source_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize the scraping strategy (setup session, etc.)."""
        connector = aiohttp.TCPConnector(limit=self.source_config.concurrent_downloads)
        timeout = aiohttp.ClientTimeout(total=self.source_config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.source_config.headers
        )
    
    async def cleanup(self) -> None:
        """Clean up resources (close session, etc.)."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def discover_pdf_urls(self, start_url: str, max_depth: int = None) -> List[str]:
        """
        Discover PDF URLs from the given starting URL.
        
        Args:
            start_url: URL to start discovery from
            max_depth: Maximum crawling depth (None uses source config default)
            
        Returns:
            List of discovered PDF URLs
        """
        pass
    
    @abstractmethod
    async def extract_page_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from a web page.
        
        Args:
            url: URL of the page to analyze
            
        Returns:
            Dictionary containing extracted metadata
        """
        pass
    
    def _matches_pdf_patterns(self, url: str) -> bool:
        """
        Check if URL matches any of the configured PDF patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL matches PDF patterns, False otherwise
        """
        for pattern in self.source_config.pdf_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def _matches_exclude_patterns(self, url: str) -> bool:
        """
        Check if URL matches any exclude patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be excluded, False otherwise
        """
        for pattern in self.source_config.exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """
        Validate if URL is a valid PDF URL for this source.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid PDF URL, False otherwise
        """
        if self._matches_exclude_patterns(url):
            return False
        
        return self._matches_pdf_patterns(url)
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting based on source configuration."""
        if self.source_config.rate_limit > 0:
            await asyncio.sleep(self.source_config.rate_limit)


class StaticHTMLStrategy(ScrapingStrategyBase):
    """
    Scraping strategy for static HTML websites using BeautifulSoup.
    Best for traditional websites with server-side rendered content.
    """
    
    async def discover_pdf_urls(self, start_url: str, max_depth: int = None) -> List[str]:
        """
        Discover PDF URLs by crawling static HTML pages.
        
        Args:
            start_url: URL to start crawling from
            max_depth: Maximum crawling depth
            
        Returns:
            List of discovered PDF URLs
        """
        if max_depth is None:
            max_depth = self.source_config.max_depth
        
        discovered_urls = set()
        visited_pages = set()
        pages_to_visit = [(start_url, 0)]  # (url, depth)
        
        while pages_to_visit:
            current_url, depth = pages_to_visit.pop(0)
            
            if depth > max_depth or current_url in visited_pages:
                continue
            
            visited_pages.add(current_url)
            
            try:
                await self._rate_limit()
                
                async with self.session.get(current_url) as response:
                    if response.status != 200:
                        self.logger.warning(f"Failed to fetch {current_url}: {response.status}")
                        continue
                    
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Find all links
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        href = link['href']
                        absolute_url = urljoin(current_url, href)
                        
                        # Check if it's a PDF URL
                        if self._is_valid_pdf_url(absolute_url):
                            discovered_urls.add(absolute_url)
                        
                        # Add to crawl queue if it's a page link and within depth
                        elif depth < max_depth and self._is_crawlable_page(absolute_url):
                            pages_to_visit.append((absolute_url, depth + 1))
            
            except Exception as e:
                self.logger.error(f"Error crawling {current_url}: {e}")
        
        self.logger.info(f"Discovered {len(discovered_urls)} PDF URLs from {start_url}")
        return list(discovered_urls)
    
    async def extract_page_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from an HTML page.
        
        Args:
            url: URL of the page to analyze
            
        Returns:
            Dictionary containing page metadata
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {}
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                metadata = {
                    'url': url,
                    'title': self._extract_title(soup),
                    'description': self._extract_description(soup),
                    'keywords': self._extract_keywords(soup),
                    'author': self._extract_author(soup),
                    'institution': self._extract_institution(soup),
                    'course_info': self._extract_course_info(soup),
                    'pdf_links': self._extract_pdf_links(soup, url),
                    'content_type': 'text/html'
                }
                
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {url}: {e}")
            return {}
    
    def _is_crawlable_page(self, url: str) -> bool:
        """
        Determine if a URL represents a crawlable page.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be crawled, False otherwise
        """
        parsed = urlparse(url)
        
        # Stay within the same domain
        source_domain = urlparse(self.source_config.base_url).netloc
        if parsed.netloc != source_domain:
            return False
        
        # Skip non-HTML content
        skip_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.tar', '.gz']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip common non-content pages
        skip_patterns = ['/admin', '/login', '/logout', '/search', '/api/', '?download=']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
        
        return True
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description from meta tags."""
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            return desc_tag['content'].strip()
        
        # Fallback to first paragraph
        p_tag = soup.find('p')
        if p_tag:
            return p_tag.get_text().strip()[:200]
        
        return ""
    
    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract keywords from meta tags and content."""
        keywords = []
        
        # Meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            meta_keywords = [k.strip() for k in keywords_tag['content'].split(',')]
            keywords.extend(meta_keywords)
        
        # Extract from headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = heading.get_text().strip()
            if heading_text:
                keywords.append(heading_text)
        
        return keywords[:10]  # Limit to top 10
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author information."""
        # Try meta author tag
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and author_tag.get('content'):
            return author_tag['content'].strip()
        
        # Look for common author patterns
        author_patterns = [
            r'(?:by|author|instructor):\s*([^<>\n]+)',
            r'(?:prof|professor|dr)\.?\s+([^<>\n]+)',
        ]
        
        page_text = soup.get_text()
        for pattern in author_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_institution(self, soup: BeautifulSoup) -> str:
        """Extract institution information."""
        # Look for university/institution names
        institution_patterns = [
            r'([^<>\n]+ university)',
            r'([^<>\n]+ college)',
            r'([^<>\n]+ institute)',
            r'university of ([^<>\n]+)'
        ]
        
        page_text = soup.get_text()
        for pattern in institution_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return self.source_config.institution
    
    def _extract_course_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract course-related information."""
        course_info = {}
        
        # Look for course codes
        course_code_pattern = r'\b([A-Z]{2,4}\s*\d{3,4}[A-Z]?)\b'
        page_text = soup.get_text()
        
        course_match = re.search(course_code_pattern, page_text)
        if course_match:
            course_info['course_code'] = course_match.group(1)
        
        # Look for semester/term information
        term_pattern = r'\b(fall|spring|summer|winter)\s+(\d{4})\b'
        term_match = re.search(term_pattern, page_text, re.IGNORECASE)
        if term_match:
            course_info['term'] = f"{term_match.group(1).title()} {term_match.group(2)}"
        
        return course_info
    
    def _extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract PDF links from the page."""
        pdf_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            
            if self._is_valid_pdf_url(absolute_url):
                pdf_links.append(absolute_url)
        
        return pdf_links


class DynamicJavaScriptStrategy(ScrapingStrategyBase):
    """
    Scraping strategy for JavaScript-heavy websites that require browser automation.
    Uses headless browser simulation for dynamic content rendering.
    """
    
    def __init__(self, source_config: SourceConfig):
        super().__init__(source_config)
        self.browser = None
        self.page = None
    
    async def initialize(self) -> None:
        """Initialize browser automation."""
        await super().initialize()
        
        try:
            # Try to import playwright for browser automation
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.page = await self.browser.new_page()
            
            # Set user agent
            if 'User-Agent' in self.source_config.headers:
                await self.page.set_extra_http_headers({
                    'User-Agent': self.source_config.headers['User-Agent']
                })
            
        except ImportError:
            self.logger.warning("Playwright not available, falling back to static HTML strategy")
            # Fallback to static strategy
            self._fallback_strategy = StaticHTMLStrategy(self.source_config)
            await self._fallback_strategy.initialize()
    
    async def cleanup(self) -> None:
        """Clean up browser resources."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        
        if hasattr(self, '_fallback_strategy'):
            await self._fallback_strategy.cleanup()
        
        await super().cleanup()
    
    async def discover_pdf_urls(self, start_url: str, max_depth: int = None) -> List[str]:
        """
        Discover PDF URLs using browser automation for JavaScript rendering.
        
        Args:
            start_url: URL to start crawling from
            max_depth: Maximum crawling depth
            
        Returns:
            List of discovered PDF URLs
        """
        if not self.browser:
            # Use fallback strategy
            if hasattr(self, '_fallback_strategy'):
                return await self._fallback_strategy.discover_pdf_urls(start_url, max_depth)
            return []
        
        if max_depth is None:
            max_depth = self.source_config.max_depth
        
        discovered_urls = set()
        visited_pages = set()
        pages_to_visit = [(start_url, 0)]
        
        while pages_to_visit:
            current_url, depth = pages_to_visit.pop(0)
            
            if depth > max_depth or current_url in visited_pages:
                continue
            
            visited_pages.add(current_url)
            
            try:
                await self._rate_limit()
                
                # Navigate to page and wait for content to load
                await self.page.goto(current_url, wait_until='networkidle')
                
                # Wait for dynamic content to load
                await asyncio.sleep(2)
                
                # Get all links after JavaScript execution
                links = await self.page.evaluate('''
                    () => {
                        const links = [];
                        document.querySelectorAll('a[href]').forEach(link => {
                            links.push(link.href);
                        });
                        return links;
                    }
                ''')
                
                for link_url in links:
                    # Check if it's a PDF URL
                    if self._is_valid_pdf_url(link_url):
                        discovered_urls.add(link_url)
                    
                    # Add to crawl queue if it's a page link and within depth
                    elif depth < max_depth and self._is_crawlable_page(link_url):
                        pages_to_visit.append((link_url, depth + 1))
            
            except Exception as e:
                self.logger.error(f"Error crawling {current_url} with browser: {e}")
        
        self.logger.info(f"Discovered {len(discovered_urls)} PDF URLs from {start_url} using browser")
        return list(discovered_urls)
    
    async def extract_page_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from a JavaScript-rendered page.
        
        Args:
            url: URL of the page to analyze
            
        Returns:
            Dictionary containing page metadata
        """
        if not self.browser:
            # Use fallback strategy
            if hasattr(self, '_fallback_strategy'):
                return await self._fallback_strategy.extract_page_metadata(url)
            return {}
        
        try:
            await self.page.goto(url, wait_until='networkidle')
            await asyncio.sleep(2)  # Wait for dynamic content
            
            # Extract metadata using JavaScript
            metadata = await self.page.evaluate('''
                () => {
                    const getMetaContent = (name) => {
                        const meta = document.querySelector(`meta[name="${name}"]`);
                        return meta ? meta.content : '';
                    };
                    
                    const title = document.title || document.querySelector('h1')?.textContent || '';
                    const description = getMetaContent('description') || 
                                     document.querySelector('p')?.textContent?.substring(0, 200) || '';
                    
                    const pdfLinks = [];
                    document.querySelectorAll('a[href*=".pdf"]').forEach(link => {
                        pdfLinks.push(link.href);
                    });
                    
                    return {
                        title: title.trim(),
                        description: description.trim(),
                        author: getMetaContent('author'),
                        keywords: getMetaContent('keywords').split(',').map(k => k.trim()),
                        pdf_links: pdfLinks,
                        content_type: 'text/html'
                    };
                }
            ''')
            
            metadata['url'] = url
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {url} with browser: {e}")
            return {}
    
    def _is_crawlable_page(self, url: str) -> bool:
        """Same logic as StaticHTMLStrategy."""
        parsed = urlparse(url)
        
        # Stay within the same domain
        source_domain = urlparse(self.source_config.base_url).netloc
        if parsed.netloc != source_domain:
            return False
        
        # Skip non-HTML content
        skip_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.tar', '.gz']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip common non-content pages
        skip_patterns = ['/admin', '/login', '/logout', '/search', '/api/', '?download=']
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
        
        return True


class APIEndpointStrategy(ScrapingStrategyBase):
    """
    Scraping strategy for academic repositories that provide API access.
    Optimized for structured data retrieval from REST APIs.
    """
    
    async def discover_pdf_urls(self, start_url: str, max_depth: int = None) -> List[str]:
        """
        Discover PDF URLs through API endpoints.
        
        Args:
            start_url: Base API URL or starting point
            max_depth: Not used for API strategy
            
        Returns:
            List of discovered PDF URLs
        """
        discovered_urls = set()
        
        # Handle different API types
        if 'arxiv.org' in self.source_config.base_url:
            urls = await self._discover_arxiv_pdfs(start_url)
            discovered_urls.update(urls)
        
        elif 'researchgate.net' in self.source_config.base_url:
            urls = await self._discover_researchgate_pdfs(start_url)
            discovered_urls.update(urls)
        
        else:
            # Generic API discovery
            urls = await self._discover_generic_api_pdfs(start_url)
            discovered_urls.update(urls)
        
        self.logger.info(f"Discovered {len(discovered_urls)} PDF URLs from API {start_url}")
        return list(discovered_urls)
    
    async def extract_page_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from API responses.
        
        Args:
            url: API endpoint URL
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {}
                
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    data = await response.json()
                    return self._parse_json_metadata(data, url)
                
                elif 'application/xml' in content_type or 'text/xml' in content_type:
                    text = await response.text()
                    return self._parse_xml_metadata(text, url)
                
                else:
                    # Fallback to text parsing
                    text = await response.text()
                    return {'url': url, 'content': text[:500], 'content_type': content_type}
                    
        except Exception as e:
            self.logger.error(f"Error extracting API metadata from {url}: {e}")
            return {}
    
    async def _discover_arxiv_pdfs(self, query_url: str) -> List[str]:
        """Discover PDFs from arXiv API."""
        pdf_urls = []
        
        try:
            # Build arXiv API query
            if 'api.arxiv.org' not in query_url:
                # Convert to API URL
                api_url = "http://export.arxiv.org/api/query"
                params = {
                    'search_query': 'cat:cs.* OR cat:math.*',  # Computer Science or Math
                    'start': 0,
                    'max_results': 100,
                    'sortBy': 'lastUpdatedDate',
                    'sortOrder': 'descending'
                }
                
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                query_url = f"{api_url}?{query_string}"
            
            async with self.session.get(query_url) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    
                    # Parse XML to extract PDF URLs
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(xml_content)
                    
                    # arXiv uses Atom namespace
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    for entry in root.findall('atom:entry', ns):
                        # Find PDF link
                        for link in entry.findall('atom:link', ns):
                            if link.get('type') == 'application/pdf':
                                pdf_url = link.get('href')
                                if pdf_url:
                                    pdf_urls.append(pdf_url)
        
        except Exception as e:
            self.logger.error(f"Error discovering arXiv PDFs: {e}")
        
        return pdf_urls
    
    async def _discover_researchgate_pdfs(self, query_url: str) -> List[str]:
        """Discover PDFs from ResearchGate (limited API access)."""
        # ResearchGate has limited public API, so this would need to be
        # implemented based on their specific API documentation
        self.logger.warning("ResearchGate API integration not fully implemented")
        return []
    
    async def _discover_generic_api_pdfs(self, api_url: str) -> List[str]:
        """Generic API PDF discovery for institutional repositories."""
        pdf_urls = []
        
        try:
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        data = await response.json()
                        pdf_urls = self._extract_pdfs_from_json(data)
                    
                    elif 'application/xml' in content_type:
                        xml_content = await response.text()
                        pdf_urls = self._extract_pdfs_from_xml(xml_content)
        
        except Exception as e:
            self.logger.error(f"Error with generic API discovery: {e}")
        
        return pdf_urls
    
    def _extract_pdfs_from_json(self, data: Dict) -> List[str]:
        """Extract PDF URLs from JSON API response."""
        pdf_urls = []
        
        def extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in ['pdf_url', 'download_url', 'file_url'] and isinstance(value, str):
                        if value.endswith('.pdf'):
                            pdf_urls.append(value)
                    else:
                        extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        extract_recursive(data)
        return pdf_urls
    
    def _extract_pdfs_from_xml(self, xml_content: str) -> List[str]:
        """Extract PDF URLs from XML API response."""
        pdf_urls = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            # Look for common XML patterns
            for elem in root.iter():
                if elem.text and elem.text.endswith('.pdf'):
                    pdf_urls.append(elem.text)
                
                # Check attributes
                for attr_value in elem.attrib.values():
                    if isinstance(attr_value, str) and attr_value.endswith('.pdf'):
                        pdf_urls.append(attr_value)
        
        except Exception as e:
            self.logger.error(f"Error parsing XML for PDFs: {e}")
        
        return pdf_urls
    
    def _parse_json_metadata(self, data: Dict, url: str) -> Dict[str, Any]:
        """Parse metadata from JSON API response."""
        metadata = {'url': url, 'content_type': 'application/json'}
        
        # Common JSON metadata fields
        field_mappings = {
            'title': ['title', 'name', 'subject'],
            'author': ['author', 'creator', 'authors'],
            'description': ['description', 'abstract', 'summary'],
            'keywords': ['keywords', 'tags', 'subjects'],
            'institution': ['institution', 'publisher', 'source']
        }
        
        for meta_key, json_keys in field_mappings.items():
            for json_key in json_keys:
                if json_key in data:
                    metadata[meta_key] = data[json_key]
                    break
        
        return metadata
    
    def _parse_xml_metadata(self, xml_content: str, url: str) -> Dict[str, Any]:
        """Parse metadata from XML API response."""
        metadata = {'url': url, 'content_type': 'application/xml'}
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            # Extract common metadata elements
            for elem in root.iter():
                tag_lower = elem.tag.lower()
                if 'title' in tag_lower and elem.text:
                    metadata['title'] = elem.text.strip()
                elif 'author' in tag_lower and elem.text:
                    metadata['author'] = elem.text.strip()
                elif 'description' in tag_lower or 'abstract' in tag_lower:
                    if elem.text:
                        metadata['description'] = elem.text.strip()
        
        except Exception as e:
            self.logger.error(f"Error parsing XML metadata: {e}")
        
        return metadata


class StrategyFactory:
    """
    Factory class for creating appropriate scraping strategies based on source configuration.
    """
    
    @staticmethod
    def create_strategy(source_config: SourceConfig) -> ScrapingStrategyBase:
        """
        Create the appropriate scraping strategy for a source.
        
        Args:
            source_config: Configuration for the source
            
        Returns:
            Appropriate scraping strategy instance
        """
        strategy_map = {
            ScrapingStrategy.STATIC_HTML: StaticHTMLStrategy,
            ScrapingStrategy.DYNAMIC_JAVASCRIPT: DynamicJavaScriptStrategy,
            ScrapingStrategy.API_ENDPOINT: APIEndpointStrategy,
        }
        
        strategy_class = strategy_map.get(source_config.scraping_strategy, StaticHTMLStrategy)
        return strategy_class(source_config)
    
    @staticmethod
    def get_available_strategies() -> List[ScrapingStrategy]:
        """
        Get list of available scraping strategies.
        
        Returns:
            List of available strategy types
        """
        return list(ScrapingStrategy)