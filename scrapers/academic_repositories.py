"""
Academic repository integrations for major research and educational platforms.
Provides specialized scrapers for arXiv, ResearchGate, institutional repositories, and Google Scholar.
"""

import asyncio
import logging
import re
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, quote_plus
import aiohttp
from bs4 import BeautifulSoup

from data.models import SourceConfig, PDFMetadata, AcademicLevel, ScrapingStrategy


class AcademicRepositoryBase(ABC):
    """
    Abstract base class for academic repository integrations.
    Defines common interface for all repository scrapers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the repository scraper.
        
        Args:
            config: Configuration dictionary for the repository
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = self.config.get('rate_limit', 2.0)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize HTTP session and any required setup."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for papers in the repository.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        pass
    
    @abstractmethod
    async def get_pdf_urls(self, paper_metadata: Dict[str, Any]) -> List[str]:
        """
        Extract PDF URLs from paper metadata.
        
        Args:
            paper_metadata: Paper metadata dictionary
            
        Returns:
            List of PDF URLs
        """
        pass
    
    async def _rate_limit_delay(self) -> None:
        """Apply rate limiting delay."""
        if self.rate_limit > 0:
            await asyncio.sleep(self.rate_limit)


class ArxivRepository(AcademicRepositoryBase):
    """
    arXiv.org repository integration using their API.
    Provides access to research papers in computer science, mathematics, physics, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = "http://export.arxiv.org/api/query"
        self.pdf_base_url = "https://arxiv.org/pdf/"
        
    async def search_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search arXiv papers using their API.
        
        Args:
            query: Search query (can include categories like 'cat:cs.*')
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        papers = []
        start = 0
        batch_size = min(100, max_results)  # arXiv API limit
        
        while len(papers) < max_results:
            await self._rate_limit_delay()
            
            params = {
                'search_query': query,
                'start': start,
                'max_results': batch_size,
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            url = f"{self.base_url}?{query_string}"
            
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"arXiv API error: {response.status}")
                        break
                    
                    xml_content = await response.text()
                    batch_papers = self._parse_arxiv_xml(xml_content)
                    
                    if not batch_papers:
                        break
                    
                    papers.extend(batch_papers)
                    start += batch_size
                    
                    if len(batch_papers) < batch_size:
                        break  # No more results
                        
            except Exception as e:
                self.logger.error(f"Error searching arXiv: {e}")
                break
        
        return papers[:max_results]
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {}
                
                # Basic metadata
                title_elem = entry.find('atom:title', ns)
                paper['title'] = title_elem.text.strip() if title_elem is not None else ""
                
                summary_elem = entry.find('atom:summary', ns)
                paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else ""
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper['authors'] = authors
                
                # arXiv ID and categories
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None:
                    arxiv_id = id_elem.text.split('/')[-1]
                    paper['arxiv_id'] = arxiv_id
                    paper['pdf_url'] = f"{self.pdf_base_url}{arxiv_id}.pdf"
                
                # Categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                paper['categories'] = categories
                paper['subject_areas'] = self._map_arxiv_categories(categories)
                
                # Publication date
                published_elem = entry.find('atom:published', ns)
                if published_elem is not None:
                    paper['published_date'] = published_elem.text
                
                # Links
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        paper['pdf_url'] = link.get('href')
                
                paper['source'] = 'arxiv'
                paper['repository'] = 'arXiv.org'
                papers.append(paper)
                
        except ET.ParseError as e:
            self.logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers
    
    def _map_arxiv_categories(self, categories: List[str]) -> List[str]:
        """Map arXiv categories to subject areas."""
        category_mapping = {
            'cs.': 'computer_science',
            'math.': 'mathematics',
            'physics.': 'physics',
            'stat.': 'statistics',
            'econ.': 'economics',
            'q-bio.': 'biology',
            'cond-mat.': 'physics',
            'astro-ph.': 'astronomy',
            'hep-': 'physics',
            'nucl-': 'physics',
            'quant-ph': 'physics'
        }
        
        subject_areas = set()
        for category in categories:
            for prefix, subject in category_mapping.items():
                if category.startswith(prefix):
                    subject_areas.add(subject)
                    break
        
        return list(subject_areas) if subject_areas else ['general']
    
    async def get_pdf_urls(self, paper_metadata: Dict[str, Any]) -> List[str]:
        """Extract PDF URLs from arXiv paper metadata."""
        urls = []
        
        if 'pdf_url' in paper_metadata:
            urls.append(paper_metadata['pdf_url'])
        elif 'arxiv_id' in paper_metadata:
            urls.append(f"{self.pdf_base_url}{paper_metadata['arxiv_id']}.pdf")
        
        return urls
    
    async def search_by_category(self, category: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search papers by arXiv category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'math.CO')
            max_results: Maximum results to return
            
        Returns:
            List of paper metadata
        """
        query = f"cat:{category}"
        return await self.search_papers(query, max_results)
    
    async def get_recent_papers(self, days: int = 7, categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent papers from specified categories.
        
        Args:
            days: Number of days to look back
            categories: List of categories to search (default: cs.*, math.*)
            
        Returns:
            List of recent paper metadata
        """
        if categories is None:
            categories = ['cs.*', 'math.*']
        
        all_papers = []
        
        for category in categories:
            query = f"cat:{category}"
            papers = await self.search_papers(query, max_results=200)
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_papers = []
            
            for paper in papers:
                if 'published_date' in paper:
                    try:
                        pub_date = datetime.fromisoformat(paper['published_date'].replace('Z', '+00:00'))
                        if pub_date >= cutoff_date:
                            recent_papers.append(paper)
                    except ValueError:
                        continue
            
            all_papers.extend(recent_papers)
        
        return all_papers


class ResearchGateRepository(AcademicRepositoryBase):
    """
    ResearchGate integration (limited due to API restrictions).
    Uses web scraping for public content discovery.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = "https://www.researchgate.net"
        self.search_url = f"{self.base_url}/search"
        
    async def initialize(self) -> None:
        """Initialize with headers to mimic browser requests."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
    
    async def search_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search ResearchGate papers (limited functionality).
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        papers = []
        
        try:
            await self._rate_limit_delay()
            
            params = {
                'q': query,
                'type': 'publication'
            }
            
            async with self.session.get(self.search_url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"ResearchGate search failed: {response.status}")
                    return papers
                
                html_content = await response.text()
                papers = self._parse_researchgate_search(html_content)
                
        except Exception as e:
            self.logger.error(f"Error searching ResearchGate: {e}")
        
        return papers[:max_results]
    
    def _parse_researchgate_search(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse ResearchGate search results HTML."""
        papers = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # ResearchGate's structure may change, this is a basic implementation
            publication_items = soup.find_all('div', class_=re.compile(r'publication-item|search-result'))
            
            for item in publication_items:
                paper = {}
                
                # Title
                title_elem = item.find(['h3', 'h4', 'a'], class_=re.compile(r'title|publication-title'))
                if title_elem:
                    paper['title'] = title_elem.get_text().strip()
                
                # Authors
                authors_elem = item.find('div', class_=re.compile(r'authors|publication-authors'))
                if authors_elem:
                    authors_text = authors_elem.get_text().strip()
                    paper['authors'] = [author.strip() for author in authors_text.split(',')]
                
                # Abstract/Description
                abstract_elem = item.find('div', class_=re.compile(r'abstract|description'))
                if abstract_elem:
                    paper['abstract'] = abstract_elem.get_text().strip()
                
                # PDF link (if available)
                pdf_link = item.find('a', href=re.compile(r'\.pdf'))
                if pdf_link:
                    paper['pdf_url'] = urljoin(self.base_url, pdf_link['href'])
                
                paper['source'] = 'researchgate'
                paper['repository'] = 'ResearchGate'
                
                if paper.get('title'):  # Only add if we have at least a title
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"Error parsing ResearchGate HTML: {e}")
        
        return papers
    
    async def get_pdf_urls(self, paper_metadata: Dict[str, Any]) -> List[str]:
        """Extract PDF URLs from ResearchGate paper metadata."""
        urls = []
        
        if 'pdf_url' in paper_metadata:
            urls.append(paper_metadata['pdf_url'])
        
        return urls


class InstitutionalRepository(AcademicRepositoryBase):
    """
    Generic institutional repository integration.
    Supports common repository software like DSpace, EPrints, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = config.get('base_url', '') if config else ''
        self.repository_type = config.get('type', 'generic') if config else 'generic'
        
    async def search_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search institutional repository.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        if self.repository_type == 'dspace':
            return await self._search_dspace(query, max_results)
        elif self.repository_type == 'eprints':
            return await self._search_eprints(query, max_results)
        else:
            return await self._search_generic(query, max_results)
    
    async def _search_dspace(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search DSpace repository."""
        papers = []
        
        try:
            # DSpace REST API endpoint
            api_url = f"{self.base_url}/rest/items"
            params = {
                'query': query,
                'limit': min(max_results, 100)
            }
            
            await self._rate_limit_delay()
            
            async with self.session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_dspace_response(data)
                    
        except Exception as e:
            self.logger.error(f"Error searching DSpace repository: {e}")
        
        return papers
    
    async def _search_eprints(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search EPrints repository."""
        papers = []
        
        try:
            # EPrints search endpoint
            search_url = f"{self.base_url}/cgi/search/simple"
            params = {
                'q': query,
                'output': 'JSON',
                'n': min(max_results, 100)
            }
            
            await self._rate_limit_delay()
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_eprints_response(data)
                    
        except Exception as e:
            self.logger.error(f"Error searching EPrints repository: {e}")
        
        return papers
    
    async def _search_generic(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generic repository search using web scraping."""
        papers = []
        
        try:
            # Try common search endpoints
            search_endpoints = ['/search', '/browse', '/discover']
            
            for endpoint in search_endpoints:
                search_url = f"{self.base_url}{endpoint}"
                params = {'q': query, 'query': query}
                
                await self._rate_limit_delay()
                
                async with self.session.get(search_url, params=params) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        endpoint_papers = self._parse_generic_repository(html_content)
                        papers.extend(endpoint_papers)
                        
                        if len(papers) >= max_results:
                            break
                            
        except Exception as e:
            self.logger.error(f"Error searching generic repository: {e}")
        
        return papers[:max_results]
    
    def _parse_dspace_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse DSpace API response."""
        papers = []
        
        items = data.get('items', []) if isinstance(data, dict) else data
        
        for item in items:
            paper = {
                'source': 'institutional_repository',
                'repository': self.base_url,
                'repository_type': 'dspace'
            }
            
            # Extract metadata
            metadata = item.get('metadata', {})
            
            if 'dc.title' in metadata:
                paper['title'] = metadata['dc.title'][0]['value']
            
            if 'dc.creator' in metadata:
                paper['authors'] = [author['value'] for author in metadata['dc.creator']]
            
            if 'dc.description.abstract' in metadata:
                paper['abstract'] = metadata['dc.description.abstract'][0]['value']
            
            if 'dc.subject' in metadata:
                paper['keywords'] = [subject['value'] for subject in metadata['dc.subject']]
            
            # Look for PDF bitstreams
            bitstreams = item.get('bitstreams', [])
            for bitstream in bitstreams:
                if bitstream.get('mimeType') == 'application/pdf':
                    paper['pdf_url'] = f"{self.base_url}/bitstream/{bitstream['uuid']}/{bitstream['name']}"
                    break
            
            if paper.get('title'):
                papers.append(paper)
        
        return papers
    
    def _parse_eprints_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse EPrints API response."""
        papers = []
        
        items = data.get('items', []) if isinstance(data, dict) else []
        
        for item in items:
            paper = {
                'source': 'institutional_repository',
                'repository': self.base_url,
                'repository_type': 'eprints'
            }
            
            paper['title'] = item.get('title', '')
            paper['authors'] = [creator.get('name', '') for creator in item.get('creators', [])]
            paper['abstract'] = item.get('abstract', '')
            paper['keywords'] = item.get('keywords', '').split(',') if item.get('keywords') else []
            
            # Look for PDF documents
            documents = item.get('documents', [])
            for doc in documents:
                if doc.get('mime_type') == 'application/pdf':
                    paper['pdf_url'] = doc.get('uri', '')
                    break
            
            if paper.get('title'):
                papers.append(paper)
        
        return papers
    
    def _parse_generic_repository(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse generic repository HTML."""
        papers = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for common patterns in repository HTML
            result_items = soup.find_all(['div', 'li'], class_=re.compile(r'result|item|record|publication'))
            
            for item in result_items:
                paper = {
                    'source': 'institutional_repository',
                    'repository': self.base_url,
                    'repository_type': 'generic'
                }
                
                # Title
                title_elem = item.find(['h1', 'h2', 'h3', 'h4'], class_=re.compile(r'title'))
                if not title_elem:
                    title_elem = item.find('a', href=True)
                
                if title_elem:
                    paper['title'] = title_elem.get_text().strip()
                
                # PDF link
                pdf_link = item.find('a', href=re.compile(r'\.pdf'))
                if pdf_link:
                    paper['pdf_url'] = urljoin(self.base_url, pdf_link['href'])
                
                # Authors
                author_elem = item.find(['div', 'span'], class_=re.compile(r'author|creator'))
                if author_elem:
                    authors_text = author_elem.get_text().strip()
                    paper['authors'] = [author.strip() for author in authors_text.split(',')]
                
                if paper.get('title'):
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"Error parsing generic repository HTML: {e}")
        
        return papers
    
    async def get_pdf_urls(self, paper_metadata: Dict[str, Any]) -> List[str]:
        """Extract PDF URLs from institutional repository metadata."""
        urls = []
        
        if 'pdf_url' in paper_metadata:
            urls.append(paper_metadata['pdf_url'])
        
        return urls


class GoogleScholarRepository(AcademicRepositoryBase):
    """
    Google Scholar integration for citation tracking and paper discovery.
    Note: Google Scholar has strict rate limiting and anti-bot measures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = "https://scholar.google.com"
        self.search_url = f"{self.base_url}/scholar"
        self.rate_limit = 5.0  # More conservative rate limiting
        
    async def initialize(self) -> None:
        """Initialize with browser-like headers."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
    
    async def search_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search Google Scholar (use with caution due to rate limits).
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        papers = []
        start = 0
        
        while len(papers) < max_results:
            await self._rate_limit_delay()
            
            params = {
                'q': query,
                'start': start,
                'num': min(10, max_results - len(papers))  # Google Scholar shows 10 per page
            }
            
            try:
                async with self.session.get(self.search_url, params=params) as response:
                    if response.status != 200:
                        self.logger.warning(f"Google Scholar returned status {response.status}")
                        break
                    
                    html_content = await response.text()
                    
                    # Check for CAPTCHA or blocking
                    if 'captcha' in html_content.lower() or 'unusual traffic' in html_content.lower():
                        self.logger.warning("Google Scholar detected unusual traffic - stopping search")
                        break
                    
                    batch_papers = self._parse_scholar_results(html_content)
                    
                    if not batch_papers:
                        break
                    
                    papers.extend(batch_papers)
                    start += len(batch_papers)
                    
            except Exception as e:
                self.logger.error(f"Error searching Google Scholar: {e}")
                break
        
        return papers[:max_results]
    
    def _parse_scholar_results(self, html_content: str) -> List[Dict[str, Any]]:
        """Parse Google Scholar search results."""
        papers = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Google Scholar result structure
            result_divs = soup.find_all('div', class_='gs_r gs_or gs_scl')
            
            for result in result_divs:
                paper = {
                    'source': 'google_scholar',
                    'repository': 'Google Scholar'
                }
                
                # Title and main link
                title_elem = result.find('h3', class_='gs_rt')
                if title_elem:
                    title_link = title_elem.find('a')
                    if title_link:
                        paper['title'] = title_link.get_text().strip()
                        paper['url'] = title_link.get('href', '')
                
                # Authors and publication info
                authors_elem = result.find('div', class_='gs_a')
                if authors_elem:
                    authors_text = authors_elem.get_text()
                    # Parse authors (usually before first dash)
                    if ' - ' in authors_text:
                        authors_part = authors_text.split(' - ')[0]
                        paper['authors'] = [author.strip() for author in authors_part.split(',')]
                
                # Abstract/snippet
                snippet_elem = result.find('div', class_='gs_rs')
                if snippet_elem:
                    paper['abstract'] = snippet_elem.get_text().strip()
                
                # Citation count
                cite_elem = result.find('a', string=re.compile(r'Cited by \d+'))
                if cite_elem:
                    cite_text = cite_elem.get_text()
                    cite_match = re.search(r'Cited by (\d+)', cite_text)
                    if cite_match:
                        paper['citation_count'] = int(cite_match.group(1))
                
                # PDF link
                pdf_links = result.find_all('a', href=re.compile(r'\.pdf'))
                for pdf_link in pdf_links:
                    if 'pdf' in pdf_link.get_text().lower():
                        paper['pdf_url'] = pdf_link.get('href')
                        break
                
                if paper.get('title'):
                    papers.append(paper)
                    
        except Exception as e:
            self.logger.error(f"Error parsing Google Scholar results: {e}")
        
        return papers
    
    async def get_pdf_urls(self, paper_metadata: Dict[str, Any]) -> List[str]:
        """Extract PDF URLs from Google Scholar metadata."""
        urls = []
        
        if 'pdf_url' in paper_metadata:
            urls.append(paper_metadata['pdf_url'])
        
        return urls
    
    async def get_citation_info(self, paper_title: str) -> Dict[str, Any]:
        """
        Get citation information for a specific paper.
        
        Args:
            paper_title: Title of the paper to look up
            
        Returns:
            Dictionary with citation information
        """
        citation_info = {}
        
        try:
            await self._rate_limit_delay()
            
            params = {'q': f'"{paper_title}"'}
            
            async with self.session.get(self.search_url, params=params) as response:
                if response.status == 200:
                    html_content = await response.text()
                    results = self._parse_scholar_results(html_content)
                    
                    if results:
                        # Return citation info for the first (most relevant) result
                        result = results[0]
                        citation_info = {
                            'title': result.get('title', ''),
                            'citation_count': result.get('citation_count', 0),
                            'authors': result.get('authors', []),
                            'url': result.get('url', '')
                        }
                        
        except Exception as e:
            self.logger.error(f"Error getting citation info: {e}")
        
        return citation_info


class AcademicRepositoryManager:
    """
    Manager class for coordinating multiple academic repository integrations.
    Provides unified interface for searching across multiple repositories.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the repository manager.
        
        Args:
            config: Configuration for repositories
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.repositories = {}
        
        # Initialize repositories based on config
        self._initialize_repositories()
    
    def _initialize_repositories(self) -> None:
        """Initialize available repositories."""
        repo_configs = self.config.get('repositories', {})
        
        # arXiv
        if repo_configs.get('arxiv', {}).get('enabled', True):
            self.repositories['arxiv'] = ArxivRepository(repo_configs.get('arxiv', {}))
        
        # ResearchGate
        if repo_configs.get('researchgate', {}).get('enabled', False):
            self.repositories['researchgate'] = ResearchGateRepository(repo_configs.get('researchgate', {}))
        
        # Google Scholar
        if repo_configs.get('google_scholar', {}).get('enabled', False):
            self.repositories['google_scholar'] = GoogleScholarRepository(repo_configs.get('google_scholar', {}))
        
        # Institutional repositories
        institutional_repos = repo_configs.get('institutional', [])
        for i, repo_config in enumerate(institutional_repos):
            if repo_config.get('enabled', True):
                repo_name = f"institutional_{i}"
                self.repositories[repo_name] = InstitutionalRepository(repo_config)
    
    async def search_all_repositories(self, query: str, max_results_per_repo: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all enabled repositories.
        
        Args:
            query: Search query string
            max_results_per_repo: Maximum results per repository
            
        Returns:
            Dictionary mapping repository names to search results
        """
        results = {}
        
        # Create tasks for concurrent searching
        tasks = []
        for repo_name, repository in self.repositories.items():
            task = asyncio.create_task(
                self._search_repository_safe(repo_name, repository, query, max_results_per_repo),
                name=f"search_{repo_name}"
            )
            tasks.append(task)
        
        # Wait for all searches to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(completed_results):
            repo_name = list(self.repositories.keys())[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Error searching {repo_name}: {result}")
                results[repo_name] = []
            else:
                results[repo_name] = result
        
        return results
    
    async def _search_repository_safe(self, repo_name: str, repository: AcademicRepositoryBase, 
                                    query: str, max_results: int) -> List[Dict[str, Any]]:
        """Safely search a repository with error handling."""
        try:
            async with repository:
                return await repository.search_papers(query, max_results)
        except Exception as e:
            self.logger.error(f"Error searching repository {repo_name}: {e}")
            return []
    
    async def get_all_pdf_urls(self, search_results: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Extract all PDF URLs from search results across repositories.
        
        Args:
            search_results: Results from search_all_repositories
            
        Returns:
            List of unique PDF URLs
        """
        all_urls = set()
        
        for repo_name, papers in search_results.items():
            if repo_name not in self.repositories:
                continue
            
            repository = self.repositories[repo_name]
            
            try:
                async with repository:
                    for paper in papers:
                        urls = await repository.get_pdf_urls(paper)
                        all_urls.update(urls)
                        
            except Exception as e:
                self.logger.error(f"Error extracting URLs from {repo_name}: {e}")
        
        return list(all_urls)
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get statistics about available repositories."""
        return {
            'total_repositories': len(self.repositories),
            'enabled_repositories': list(self.repositories.keys()),
            'repository_types': {
                name: repo.__class__.__name__ 
                for name, repo in self.repositories.items()
            }
        }