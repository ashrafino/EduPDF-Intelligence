"""
Intelligent crawling system for educational PDF discovery.
Implements recursive link discovery, academic site structure recognition, 
URL pattern matching, and sitemap parsing for efficient crawling.
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
import aiohttp
from bs4 import BeautifulSoup

from data.models import SourceConfig, PDFMetadata
from scrapers.scraping_strategies import StrategyFactory


class IntelligentCrawler:
    """
    Intelligent crawler that adapts to academic site structures and efficiently discovers PDFs.
    Implements depth-limited crawling, sitemap parsing, and academic content recognition.
    """
    
    def __init__(self, source_config: SourceConfig):
        """
        Initialize the intelligent crawler.
        
        Args:
            source_config: Configuration for the source being crawled
        """
        self.source_config = source_config
        self.logger = logging.getLogger(__name__)
        
        # Crawling state
        self.visited_urls: Set[str] = set()
        self.discovered_pdfs: Set[str] = set()
        self.crawl_queue: List[Tuple[str, int, str]] = []  # (url, depth, parent_url)
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        # Academic site patterns
        self.academic_patterns = self._initialize_academic_patterns()
        
        # Create appropriate scraping strategy
        self.strategy = StrategyFactory.create_strategy(source_config)
    
    def _initialize_academic_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for recognizing academic site structures."""
        return {
            'course_pages': [
                r'/courses?/',
                r'/classes?/',
                r'/subjects?/',
                r'/curriculum/',
                r'/academics?/',
                r'/teaching/',
                r'/syllabi?/',
                r'/course[-_]?materials?/'
            ],
            'faculty_directories': [
                r'/faculty/',
                r'/staff/',
                r'/people/',
                r'/directory/',
                r'/professors?/',
                r'/instructors?/',
                r'/researchers?/'
            ],
            'department_pages': [
                r'/departments?/',
                r'/schools?/',
                r'/colleges?/',
                r'/divisions?/',
                r'/programs?/',
                r'/majors?/'
            ],
            'resource_pages': [
                r'/resources?/',
                r'/materials?/',
                r'/downloads?/',
                r'/publications?/',
                r'/papers?/',
                r'/documents?/',
                r'/library/',
                r'/archives?/'
            ],
            'lecture_materials': [
                r'/lectures?/',
                r'/notes?/',
                r'/slides?/',
                r'/presentations?/',
                r'/handouts?/',
                r'/assignments?/',
                r'/homework/',
                r'/exercises?/'
            ]
        }
    
    async def crawl_source(self, max_depth: int = None, max_pages: int = 1000) -> List[str]:
        """
        Perform intelligent crawling of the source to discover PDFs.
        
        Args:
            max_depth: Maximum crawling depth (uses source config if None)
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of discovered PDF URLs
        """
        if max_depth is None:
            max_depth = self.source_config.max_depth
        
        self.logger.info(f"Starting intelligent crawl of {self.source_config.name}")
        
        async with self.strategy:
            # Step 1: Check robots.txt
            await self._load_robots_txt()
            
            # Step 2: Try sitemap discovery first
            sitemap_pdfs = await self._discover_from_sitemaps()
            self.discovered_pdfs.update(sitemap_pdfs)
            
            # Step 3: Initialize crawl queue with strategic entry points
            entry_points = await self._discover_entry_points()
            for url in entry_points:
                self.crawl_queue.append((url, 0, self.source_config.base_url))
            
            # Step 4: Perform intelligent crawling
            pages_crawled = 0
            while self.crawl_queue and pages_crawled < max_pages:
                current_url, depth, parent_url = self.crawl_queue.pop(0)
                
                if depth > max_depth or current_url in self.visited_urls:
                    continue
                
                if not await self._is_crawlable_url(current_url):
                    continue
                
                self.visited_urls.add(current_url)
                pages_crawled += 1
                
                try:
                    # Crawl the page
                    page_pdfs, new_links = await self._crawl_page(current_url, depth)
                    self.discovered_pdfs.update(page_pdfs)
                    
                    # Add new links to queue with priority
                    prioritized_links = self._prioritize_links(new_links, current_url)
                    for link_url, priority in prioritized_links:
                        if link_url not in self.visited_urls:
                            # Insert based on priority (higher priority first)
                            insert_pos = 0
                            for i, (_, _, _) in enumerate(self.crawl_queue):
                                if priority <= self._calculate_url_priority(self.crawl_queue[i][0]):
                                    insert_pos = i + 1
                                else:
                                    break
                            self.crawl_queue.insert(insert_pos, (link_url, depth + 1, current_url))
                    
                    # Rate limiting
                    await asyncio.sleep(self.source_config.rate_limit)
                    
                except Exception as e:
                    self.logger.error(f"Error crawling {current_url}: {e}")
        
        self.logger.info(f"Crawl completed. Found {len(self.discovered_pdfs)} PDFs from {pages_crawled} pages")
        return list(self.discovered_pdfs)
    
    async def _load_robots_txt(self) -> None:
        """Load and parse robots.txt for the source domain."""
        try:
            parsed_url = urlparse(self.source_config.base_url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_url, timeout=10) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        
                        # Parse robots.txt
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.read()
                        
                        # Store for later use
                        self.robots_cache[parsed_url.netloc] = rp
                        
                        self.logger.info(f"Loaded robots.txt from {robots_url}")
                    else:
                        self.logger.info(f"No robots.txt found at {robots_url}")
        
        except Exception as e:
            self.logger.warning(f"Error loading robots.txt: {e}")
    
    async def _discover_from_sitemaps(self) -> Set[str]:
        """Discover PDFs from XML sitemaps."""
        pdf_urls = set()
        
        try:
            # Common sitemap locations
            sitemap_urls = [
                urljoin(self.source_config.base_url, '/sitemap.xml'),
                urljoin(self.source_config.base_url, '/sitemap_index.xml'),
                urljoin(self.source_config.base_url, '/sitemaps/sitemap.xml')
            ]
            
            async with aiohttp.ClientSession() as session:
                for sitemap_url in sitemap_urls:
                    try:
                        async with session.get(sitemap_url, timeout=15) as response:
                            if response.status == 200:
                                xml_content = await response.text()
                                sitemap_pdfs = await self._parse_sitemap(xml_content, session)
                                pdf_urls.update(sitemap_pdfs)
                                self.logger.info(f"Found {len(sitemap_pdfs)} PDFs in sitemap {sitemap_url}")
                    
                    except Exception as e:
                        self.logger.debug(f"Could not access sitemap {sitemap_url}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in sitemap discovery: {e}")
        
        return pdf_urls
    
    async def _parse_sitemap(self, xml_content: str, session: aiohttp.ClientSession) -> Set[str]:
        """Parse XML sitemap and extract PDF URLs."""
        pdf_urls = set()
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle sitemap index files
            if 'sitemapindex' in root.tag:
                for sitemap in root:
                    loc_elem = sitemap.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None:
                        # Recursively parse sub-sitemaps
                        try:
                            async with session.get(loc_elem.text, timeout=10) as response:
                                if response.status == 200:
                                    sub_xml = await response.text()
                                    sub_pdfs = await self._parse_sitemap(sub_xml, session)
                                    pdf_urls.update(sub_pdfs)
                        except Exception as e:
                            self.logger.debug(f"Error parsing sub-sitemap {loc_elem.text}: {e}")
            
            # Handle regular sitemap files
            else:
                for url in root:
                    loc_elem = url.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None:
                        url_text = loc_elem.text
                        if self._is_valid_pdf_url(url_text):
                            pdf_urls.add(url_text)
        
        except ET.ParseError as e:
            self.logger.error(f"Error parsing sitemap XML: {e}")
        
        return pdf_urls
    
    async def _discover_entry_points(self) -> List[str]:
        """Discover strategic entry points for crawling based on academic site structure."""
        entry_points = [self.source_config.base_url]
        
        try:
            async with aiohttp.ClientSession() as session:
                # Analyze the main page to find academic structure
                async with session.get(self.source_config.base_url, timeout=15) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Find links that match academic patterns
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            absolute_url = urljoin(self.source_config.base_url, href)
                            
                            if self._is_academic_entry_point(absolute_url):
                                entry_points.append(absolute_url)
                        
                        # Look for common academic navigation patterns
                        nav_links = self._extract_navigation_links(soup)
                        entry_points.extend(nav_links)
        
        except Exception as e:
            self.logger.error(f"Error discovering entry points: {e}")
        
        # Remove duplicates and limit to reasonable number
        unique_entry_points = list(set(entry_points))[:20]
        self.logger.info(f"Discovered {len(unique_entry_points)} entry points")
        
        return unique_entry_points
    
    def _is_academic_entry_point(self, url: str) -> bool:
        """Check if URL is a good academic entry point."""
        url_lower = url.lower()
        
        # Check against academic patterns
        for category, patterns in self.academic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return True
        
        return False
    
    def _extract_navigation_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract navigation links that likely lead to academic content."""
        nav_links = []
        
        # Look for navigation elements
        nav_selectors = ['nav', '.navigation', '.menu', '.navbar', '#menu', '#nav']
        
        for selector in nav_selectors:
            nav_elements = soup.select(selector)
            for nav in nav_elements:
                for link in nav.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(self.source_config.base_url, href)
                    
                    # Check if link text suggests academic content
                    link_text = link.get_text().lower()
                    academic_keywords = [
                        'courses', 'classes', 'academics', 'departments', 'faculty',
                        'research', 'publications', 'resources', 'materials', 'library'
                    ]
                    
                    if any(keyword in link_text for keyword in academic_keywords):
                        nav_links.append(absolute_url)
        
        return nav_links
    
    async def _crawl_page(self, url: str, depth: int) -> Tuple[Set[str], List[str]]:
        """
        Crawl a single page and extract PDFs and links.
        
        Args:
            url: URL to crawl
            depth: Current crawling depth
            
        Returns:
            Tuple of (PDF URLs found, new links to crawl)
        """
        page_pdfs = set()
        new_links = []
        
        try:
            # Use the appropriate strategy to discover PDFs
            strategy_pdfs = await self.strategy.discover_pdf_urls(url, max_depth=1)
            page_pdfs.update(strategy_pdfs)
            
            # Extract page metadata for link analysis
            metadata = await self.strategy.extract_page_metadata(url)
            
            # Get additional links from metadata
            if 'pdf_links' in metadata:
                page_pdfs.update(metadata['pdf_links'])
            
            # Discover new crawlable links
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.source_config.timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            absolute_url = urljoin(url, href)
                            
                            if self._should_crawl_link(absolute_url, depth):
                                new_links.append(absolute_url)
        
        except Exception as e:
            self.logger.error(f"Error crawling page {url}: {e}")
        
        return page_pdfs, new_links
    
    def _should_crawl_link(self, url: str, current_depth: int) -> bool:
        """Determine if a link should be added to the crawl queue."""
        # Basic validation
        if url in self.visited_urls:
            return False
        
        if not self._is_same_domain(url):
            return False
        
        if not self._is_crawlable_content_type(url):
            return False
        
        # Check robots.txt
        if not self._is_allowed_by_robots(url):
            return False
        
        # Prioritize academic content
        if self._is_academic_content(url):
            return True
        
        # General crawlability check
        return self._is_general_crawlable(url)
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is from the same domain as the source."""
        source_domain = urlparse(self.source_config.base_url).netloc
        url_domain = urlparse(url).netloc
        return source_domain == url_domain
    
    def _is_crawlable_content_type(self, url: str) -> bool:
        """Check if URL points to crawlable content."""
        # Skip binary files
        skip_extensions = [
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv',
            '.exe', '.dmg', '.pkg'
        ]
        
        url_lower = url.lower()
        return not any(url_lower.endswith(ext) for ext in skip_extensions)
    
    def _is_allowed_by_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        domain = urlparse(url).netloc
        
        if domain in self.robots_cache:
            rp = self.robots_cache[domain]
            user_agent = self.source_config.headers.get('User-Agent', '*')
            return rp.can_fetch(user_agent, url)
        
        return True  # Allow if no robots.txt
    
    def _is_academic_content(self, url: str) -> bool:
        """Check if URL likely contains academic content."""
        url_lower = url.lower()
        
        # Check against academic patterns
        for category, patterns in self.academic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return True
        
        # Check for academic keywords in URL
        academic_keywords = [
            'course', 'class', 'lecture', 'syllabus', 'assignment',
            'homework', 'exam', 'quiz', 'project', 'lab',
            'research', 'publication', 'paper', 'thesis',
            'faculty', 'professor', 'instructor', 'student'
        ]
        
        return any(keyword in url_lower for keyword in academic_keywords)
    
    def _is_general_crawlable(self, url: str) -> bool:
        """General crawlability check for non-academic content."""
        url_lower = url.lower()
        
        # Skip admin and system pages
        skip_patterns = [
            '/admin', '/login', '/logout', '/register', '/signup',
            '/search', '/api/', '/ajax/', '/json/', '/xml/',
            '/feed/', '/rss/', '/atom/',
            '?download=', '?action=', '?cmd=',
            '/cgi-bin/', '/scripts/', '/includes/'
        ]
        
        return not any(pattern in url_lower for pattern in skip_patterns)
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """Check if URL is a valid PDF based on source patterns."""
        # Check exclude patterns first
        for pattern in self.source_config.exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        # Check PDF patterns
        for pattern in self.source_config.pdf_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    async def _is_crawlable_url(self, url: str) -> bool:
        """Async check if URL should be crawled."""
        return (self._is_same_domain(url) and 
                self._is_crawlable_content_type(url) and
                self._is_allowed_by_robots(url))
    
    def _prioritize_links(self, links: List[str], parent_url: str) -> List[Tuple[str, int]]:
        """
        Prioritize links for crawling based on academic relevance.
        
        Args:
            links: List of URLs to prioritize
            parent_url: URL of the parent page
            
        Returns:
            List of (URL, priority) tuples, sorted by priority (higher first)
        """
        prioritized = []
        
        for url in links:
            priority = self._calculate_url_priority(url)
            prioritized.append((url, priority))
        
        # Sort by priority (higher first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized
    
    def _calculate_url_priority(self, url: str) -> int:
        """Calculate priority score for a URL."""
        priority = 0
        url_lower = url.lower()
        
        # High priority for direct academic content
        high_priority_patterns = [
            r'/courses?/', r'/classes?/', r'/lectures?/',
            r'/materials?/', r'/resources?/', r'/downloads?/'
        ]
        
        for pattern in high_priority_patterns:
            if re.search(pattern, url_lower):
                priority += 10
        
        # Medium priority for department/faculty pages
        medium_priority_patterns = [
            r'/departments?/', r'/faculty/', r'/research/',
            r'/publications?/', r'/papers?/'
        ]
        
        for pattern in medium_priority_patterns:
            if re.search(pattern, url_lower):
                priority += 5
        
        # Bonus for subject area relevance
        for subject in self.source_config.subject_areas:
            if subject.replace('_', ' ') in url_lower or subject.replace('_', '-') in url_lower:
                priority += 3
        
        # Penalty for deep nesting
        path_depth = len([p for p in urlparse(url).path.split('/') if p])
        priority -= max(0, path_depth - 3)
        
        return priority
    
    def get_crawl_statistics(self) -> Dict[str, Any]:
        """Get statistics about the crawling process."""
        return {
            'total_pages_visited': len(self.visited_urls),
            'total_pdfs_discovered': len(self.discovered_pdfs),
            'pages_in_queue': len(self.crawl_queue),
            'discovery_rate': len(self.discovered_pdfs) / max(1, len(self.visited_urls)),
            'source_name': self.source_config.name,
            'base_url': self.source_config.base_url
        }
    
    def reset_crawl_state(self) -> None:
        """Reset the crawler state for a new crawl."""
        self.visited_urls.clear()
        self.discovered_pdfs.clear()
        self.crawl_queue.clear()
        self.robots_cache.clear()