"""
Adaptive source manager for intelligent PDF source discovery and management.
Handles source configuration, validation, health checking, and discovery capabilities.
Enhanced with robust error handling and recovery mechanisms.
"""

import asyncio
import logging
import yaml
import json
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
import re

from data.models import SourceConfig, ScrapingStrategy, AcademicLevel
from utils.error_handling import (
    with_error_handling, CircuitBreaker, CircuitBreakerConfig,
    StructuredErrorLogger, RetryManager
)
from utils.error_recovery import ErrorRecoveryManager
from scrapers.academic_repositories import AcademicRepositoryManager
from scrapers.international_sources import InternationalSourceManager


class SourceManager:
    """
    Manages PDF sources with discovery capabilities, validation, and health checking.
    Supports loading from configuration files and adaptive source discovery.
    Enhanced with comprehensive error handling and recovery mechanisms.
    """
    
    def __init__(self, config_path: str = "config/sources.yaml", enable_error_recovery: bool = True):
        """
        Initialize the source manager.
        
        Args:
            config_path: Path to the source configuration file
            enable_error_recovery: Whether to enable error recovery mechanisms
        """
        self.config_path = Path(config_path)
        self.sources: Dict[str, SourceConfig] = {}
        self.discovered_sources: Dict[str, SourceConfig] = {}
        self.health_status: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
        
        # Error handling and recovery
        self.enable_error_recovery = enable_error_recovery
        if enable_error_recovery:
            self.error_logger = StructuredErrorLogger("logs/source_errors.jsonl")
            self.recovery_manager = ErrorRecoveryManager("checkpoints/sources")
            self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Academic repository integration
        self.academic_repo_manager = None
        self.international_source_manager = None
        
        # Load initial sources from configuration
        self.load_sources_from_config()
    
    def load_sources_from_config(self) -> None:
        """
        Load source configurations from YAML or JSON files.
        Supports both formats for flexibility.
        """
        try:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                self._load_yaml_config()
            elif self.config_path.suffix.lower() == '.json':
                self._load_json_config()
            else:
                self.logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}")
            self._create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading source config: {e}")
            self._create_default_config()
    
    def _load_yaml_config(self) -> None:
        """Load sources from YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        for source_id, source_data in config_data.items():
            try:
                # Convert string strategy to enum
                if 'scraping_strategy' in source_data:
                    strategy_str = source_data['scraping_strategy']
                    source_data['scraping_strategy'] = ScrapingStrategy(strategy_str)
                
                # Convert academic levels if present
                if 'academic_levels' in source_data:
                    levels = []
                    for level_str in source_data['academic_levels']:
                        levels.append(AcademicLevel(level_str))
                    source_data['academic_levels'] = levels
                
                # Handle datetime fields
                if 'last_crawled' in source_data and source_data['last_crawled']:
                    if isinstance(source_data['last_crawled'], str):
                        source_data['last_crawled'] = datetime.fromisoformat(source_data['last_crawled'])
                
                source_config = SourceConfig(**source_data)
                self.sources[source_id] = source_config
                
            except Exception as e:
                self.logger.error(f"Error parsing source config for {source_id}: {e}")
    
    def _load_json_config(self) -> None:
        """Load sources from JSON configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        for source_id, source_data in config_data.items():
            try:
                # Similar processing as YAML but for JSON format
                if 'scraping_strategy' in source_data:
                    strategy_str = source_data['scraping_strategy']
                    source_data['scraping_strategy'] = ScrapingStrategy(strategy_str)
                
                if 'academic_levels' in source_data:
                    levels = []
                    for level_str in source_data['academic_levels']:
                        levels.append(AcademicLevel(level_str))
                    source_data['academic_levels'] = levels
                
                if 'last_crawled' in source_data and source_data['last_crawled']:
                    source_data['last_crawled'] = datetime.fromisoformat(source_data['last_crawled'])
                
                source_config = SourceConfig(**source_data)
                self.sources[source_id] = source_config
                
            except Exception as e:
                self.logger.error(f"Error parsing source config for {source_id}: {e}")
    
    def _create_default_config(self) -> None:
        """Create a default configuration with basic academic sources."""
        default_sources = {
            'mit_ocw': SourceConfig(
                name="MIT OpenCourseWare",
                base_url="https://ocw.mit.edu/",
                description="MIT's free online course materials",
                scraping_strategy=ScrapingStrategy.STATIC_HTML,
                rate_limit=2.0,
                max_depth=3,
                pdf_patterns=[r"\.pdf$", r"lecture.*\.pdf", r"notes.*\.pdf"],
                subject_areas=["computer_science", "mathematics", "engineering"],
                institution="MIT",
                country="US"
            ),
            'arxiv': SourceConfig(
                name="arXiv.org",
                base_url="https://arxiv.org/",
                description="Open access research papers",
                scraping_strategy=ScrapingStrategy.API_ENDPOINT,
                rate_limit=3.0,
                max_depth=1,
                pdf_patterns=[r"\.pdf$"],
                subject_areas=["computer_science", "mathematics", "physics"],
                institution="Cornell University",
                country="US"
            )
        }
        
        self.sources = default_sources
        self.logger.info("Created default source configuration")
    
    async def discover_sources(self, seed_urls: List[str], max_sources: int = 50) -> List[SourceConfig]:
        """
        Discover new PDF sources from seed URLs using intelligent crawling.
        
        Args:
            seed_urls: Starting URLs for source discovery
            max_sources: Maximum number of sources to discover
            
        Returns:
            List of discovered source configurations
        """
        discovered = []
        processed_domains = set()
        
        async with aiohttp.ClientSession() as session:
            for seed_url in seed_urls:
                if len(discovered) >= max_sources:
                    break
                
                domain = urlparse(seed_url).netloc
                if domain in processed_domains:
                    continue
                
                processed_domains.add(domain)
                
                try:
                    # Analyze the seed URL to determine if it's an academic source
                    source_config = await self._analyze_potential_source(session, seed_url)
                    if source_config and self._is_academic_source(source_config):
                        discovered.append(source_config)
                        self.discovered_sources[domain] = source_config
                        
                        # Discover related sources from this domain
                        related_sources = await self._discover_related_sources(session, seed_url, max_depth=2)
                        for related in related_sources:
                            if len(discovered) >= max_sources:
                                break
                            if related not in discovered:
                                discovered.append(related)
                                
                except Exception as e:
                    self.logger.error(f"Error discovering sources from {seed_url}: {e}")
        
        self.logger.info(f"Discovered {len(discovered)} new sources")
        return discovered
    
    async def _analyze_potential_source(self, session: aiohttp.ClientSession, url: str) -> Optional[SourceConfig]:
        """
        Analyze a URL to determine if it's a valid academic PDF source.
        
        Args:
            session: HTTP session for requests
            url: URL to analyze
            
        Returns:
            SourceConfig if valid academic source, None otherwise
        """
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None
                
                content = await response.text()
                parsed_url = urlparse(url)
                
                # Extract basic information
                name = self._extract_site_name(content, parsed_url.netloc)
                description = self._extract_site_description(content)
                
                # Determine scraping strategy based on content
                strategy = self._determine_scraping_strategy(content)
                
                # Find PDF patterns
                pdf_patterns = self._extract_pdf_patterns(content, url)
                
                # Classify subject areas
                subject_areas = self._classify_subject_areas(content, url)
                
                # Detect institution information
                institution = self._extract_institution_info(content, parsed_url.netloc)
                
                return SourceConfig(
                    name=name,
                    base_url=f"{parsed_url.scheme}://{parsed_url.netloc}",
                    description=description,
                    scraping_strategy=strategy,
                    pdf_patterns=pdf_patterns,
                    subject_areas=subject_areas,
                    institution=institution,
                    rate_limit=2.0,  # Conservative default
                    max_depth=2
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing source {url}: {e}")
            return None
    
    def _extract_site_name(self, content: str, domain: str) -> str:
        """Extract site name from HTML content or domain."""
        # Try to find title tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up common title patterns
            title = re.sub(r'\s*-\s*.*$', '', title)  # Remove " - subtitle"
            title = re.sub(r'\s*\|\s*.*$', '', title)  # Remove " | subtitle"
            if len(title) > 5:
                return title
        
        # Fallback to domain name
        return domain.replace('www.', '').replace('.edu', '').replace('.org', '').title()
    
    def _extract_site_description(self, content: str) -> str:
        """Extract site description from meta tags."""
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if desc_match:
            return desc_match.group(1).strip()
        return ""
    
    def _determine_scraping_strategy(self, content: str) -> ScrapingStrategy:
        """Determine the best scraping strategy based on page content."""
        # Check for JavaScript frameworks
        js_indicators = ['react', 'angular', 'vue', 'spa-', 'single-page']
        if any(indicator in content.lower() for indicator in js_indicators):
            return ScrapingStrategy.DYNAMIC_JAVASCRIPT
        
        # Check for API endpoints
        api_indicators = ['api/', '/api', 'rest', 'json', 'xml']
        if any(indicator in content.lower() for indicator in api_indicators):
            return ScrapingStrategy.API_ENDPOINT
        
        # Default to static HTML
        return ScrapingStrategy.STATIC_HTML
    
    def _extract_pdf_patterns(self, content: str, base_url: str) -> List[str]:
        """Extract PDF URL patterns from page content."""
        patterns = set()
        
        # Find existing PDF links
        pdf_links = re.findall(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', content, re.IGNORECASE)
        
        for link in pdf_links:
            # Extract patterns from PDF URLs
            if 'lecture' in link.lower():
                patterns.add(r'lecture.*\.pdf')
            if 'notes' in link.lower():
                patterns.add(r'notes.*\.pdf')
            if 'slides' in link.lower():
                patterns.add(r'slides.*\.pdf')
            if 'homework' in link.lower() or 'assignment' in link.lower():
                patterns.add(r'(homework|assignment).*\.pdf')
        
        # Always include basic PDF pattern
        patterns.add(r'\.pdf$')
        
        return list(patterns)
    
    def _classify_subject_areas(self, content: str, url: str) -> List[str]:
        """Classify subject areas based on content and URL."""
        subjects = set()
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Subject area keywords
        subject_keywords = {
            'computer_science': ['computer science', 'cs', 'programming', 'software', 'algorithm'],
            'mathematics': ['mathematics', 'math', 'calculus', 'algebra', 'statistics'],
            'physics': ['physics', 'quantum', 'mechanics', 'thermodynamics'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil'],
            'biology': ['biology', 'genetics', 'molecular', 'biochemistry'],
            'chemistry': ['chemistry', 'organic', 'inorganic', 'chemical'],
            'economics': ['economics', 'finance', 'business', 'management']
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in content_lower or keyword in url_lower for keyword in keywords):
                subjects.add(subject)
        
        return list(subjects) if subjects else ['general']
    
    def _extract_institution_info(self, content: str, domain: str) -> str:
        """Extract institution name from content or domain."""
        # Try to find institution in content
        institution_patterns = [
            r'university of ([^<>\n]+)',
            r'([^<>\n]+ university)',
            r'([^<>\n]+ college)',
            r'([^<>\n]+ institute)'
        ]
        
        for pattern in institution_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        # Extract from domain
        if '.edu' in domain:
            parts = domain.replace('.edu', '').split('.')
            return parts[-1].title() + ' University'
        
        return domain.replace('www.', '').title()
    
    def _is_academic_source(self, source_config: SourceConfig) -> bool:
        """
        Determine if a source configuration represents a valid academic source.
        
        Args:
            source_config: Source configuration to validate
            
        Returns:
            True if academic source, False otherwise
        """
        # Check domain indicators
        domain = urlparse(source_config.base_url).netloc.lower()
        academic_domains = ['.edu', '.ac.', 'university', 'college', 'institute', 'arxiv', 'researchgate']
        
        if any(indicator in domain for indicator in academic_domains):
            return True
        
        # Check subject areas
        if source_config.subject_areas and any(area != 'general' for area in source_config.subject_areas):
            return True
        
        # Check institution information
        if source_config.institution and len(source_config.institution) > 3:
            return True
        
        return False
    
    async def _discover_related_sources(self, session: aiohttp.ClientSession, base_url: str, max_depth: int = 2) -> List[SourceConfig]:
        """
        Discover related academic sources from a base URL.
        
        Args:
            session: HTTP session for requests
            base_url: Base URL to search from
            max_depth: Maximum crawling depth
            
        Returns:
            List of related source configurations
        """
        related_sources = []
        
        try:
            # Look for common academic link patterns
            async with session.get(base_url, timeout=10) as response:
                if response.status != 200:
                    return related_sources
                
                content = await response.text()
                
                # Find links to other academic institutions
                academic_links = re.findall(
                    r'href=["\']([^"\']*(?:\.edu|university|college|institute)[^"\']*)["\']',
                    content, re.IGNORECASE
                )
                
                for link in academic_links[:10]:  # Limit to prevent excessive crawling
                    if link.startswith('http'):
                        full_url = link
                    else:
                        full_url = urljoin(base_url, link)
                    
                    source_config = await self._analyze_potential_source(session, full_url)
                    if source_config and self._is_academic_source(source_config):
                        related_sources.append(source_config)
                        
        except Exception as e:
            self.logger.error(f"Error discovering related sources from {base_url}: {e}")
        
        return related_sources
    
    async def validate_source(self, source: SourceConfig) -> bool:
        """
        Validate that a source is accessible and contains educational content.
        Enhanced with error handling and recovery mechanisms.
        
        Args:
            source: Source configuration to validate
            
        Returns:
            True if source is valid and accessible, False otherwise
        """
        if self.enable_error_recovery:
            return await self._validate_source_with_recovery(source)
        else:
            return await self._validate_source_basic(source)
    
    @with_error_handling(max_retries=3)
    async def _validate_source_with_recovery(self, source: SourceConfig) -> bool:
        """Validate source with error recovery mechanisms."""
        return await self._validate_source_basic(source)
    
    async def _validate_source_basic(self, source: SourceConfig) -> bool:
        """Basic source validation without error recovery."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test basic connectivity
                async with session.get(source.base_url, timeout=source.timeout) as response:
                    if response.status != 200:
                        self.logger.warning(f"Source {source.name} returned status {response.status}")
                        return False
                
                # Check for PDF content
                content = await response.text()
                pdf_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in source.pdf_patterns)
                
                if not pdf_found:
                    self.logger.warning(f"No PDF content found at source {source.name}")
                    return False
                
                # Update health status
                self.health_status[source.name] = {
                    'status': 'healthy',
                    'last_checked': datetime.now(),
                    'response_time': response.headers.get('X-Response-Time', 'unknown'),
                    'pdf_count_estimate': len(re.findall(r'\.pdf', content, re.IGNORECASE))
                }
                
                return True
                
        except Exception as e:
            if self.enable_error_recovery:
                self.error_logger.log_error(
                    e,
                    context={'validation': True},
                    source_url=source.base_url,
                    source_name=source.name
                )
            
            self.logger.error(f"Error validating source {source.name}: {e}")
            self.health_status[source.name] = {
                'status': 'unhealthy',
                'last_checked': datetime.now(),
                'error': str(e)
            }
            return False
    
    async def check_source_health(self, source_name: str) -> Dict:
        """
        Perform comprehensive health check on a specific source.
        
        Args:
            source_name: Name of the source to check
            
        Returns:
            Dictionary containing health status information
        """
        if source_name not in self.sources:
            return {'status': 'not_found', 'error': 'Source not configured'}
        
        source = self.sources[source_name]
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source.base_url, timeout=source.timeout) as response:
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    health_info = {
                        'status': 'healthy' if response.status == 200 else 'unhealthy',
                        'response_code': response.status,
                        'response_time': response_time,
                        'last_checked': datetime.now(),
                        'content_type': response.headers.get('content-type', 'unknown'),
                        'server': response.headers.get('server', 'unknown')
                    }
                    
                    if response.status == 200:
                        content = await response.text()
                        health_info['pdf_links_found'] = len(re.findall(r'\.pdf', content, re.IGNORECASE))
                        health_info['content_size'] = len(content)
                    
                    self.health_status[source_name] = health_info
                    return health_info
                    
        except Exception as e:
            error_info = {
                'status': 'error',
                'error': str(e),
                'last_checked': datetime.now(),
                'response_time': (datetime.now() - start_time).total_seconds()
            }
            self.health_status[source_name] = error_info
            return error_info
    
    def get_active_sources(self) -> List[SourceConfig]:
        """
        Get all active source configurations.
        
        Returns:
            List of active source configurations
        """
        return [source for source in self.sources.values() if source.is_active]
    
    def get_sources_by_subject(self, subject_area: str) -> List[SourceConfig]:
        """
        Get sources that cover a specific subject area.
        
        Args:
            subject_area: Subject area to filter by
            
        Returns:
            List of source configurations for the subject area
        """
        return [
            source for source in self.sources.values()
            if source.is_active and subject_area in source.subject_areas
        ]
    
    def get_source_health_summary(self) -> Dict:
        """
        Get summary of health status for all sources.
        
        Returns:
            Dictionary containing health summary statistics
        """
        total_sources = len(self.sources)
        healthy_sources = sum(1 for status in self.health_status.values() if status.get('status') == 'healthy')
        
        return {
            'total_sources': total_sources,
            'healthy_sources': healthy_sources,
            'unhealthy_sources': total_sources - healthy_sources,
            'health_percentage': (healthy_sources / total_sources * 100) if total_sources > 0 else 0,
            'last_updated': datetime.now()
        }
    
    def save_discovered_sources(self, output_path: str = "config/discovered_sources.yaml") -> None:
        """
        Save discovered sources to a configuration file.
        
        Args:
            output_path: Path to save the discovered sources
        """
        if not self.discovered_sources:
            self.logger.info("No discovered sources to save")
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert discovered sources to serializable format
        serializable_sources = {}
        for source_id, source_config in self.discovered_sources.items():
            source_dict = {
                'name': source_config.name,
                'base_url': source_config.base_url,
                'description': source_config.description,
                'scraping_strategy': source_config.scraping_strategy.value,
                'rate_limit': source_config.rate_limit,
                'max_depth': source_config.max_depth,
                'pdf_patterns': source_config.pdf_patterns,
                'subject_areas': source_config.subject_areas,
                'languages': source_config.languages,
                'institution': source_config.institution,
                'is_active': source_config.is_active
            }
            serializable_sources[source_id] = source_dict
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(serializable_sources, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Saved {len(serializable_sources)} discovered sources to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving discovered sources: {e}")
    
    def _get_circuit_breaker(self, source_name: str) -> Optional[CircuitBreaker]:
        """
        Get or create a circuit breaker for a source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            CircuitBreaker instance if error recovery is enabled, None otherwise
        """
        if not self.enable_error_recovery:
            return None
        
        if source_name not in self.circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=3,  # More sensitive for source validation
                recovery_timeout=600.0,  # 10 minutes
                expected_exception=Exception
            )
            self.circuit_breakers[source_name] = CircuitBreaker(config, source_name)
        
        return self.circuit_breakers[source_name]
    
    async def validate_sources_batch(self, sources: List[SourceConfig]) -> Dict[str, bool]:
        """
        Validate multiple sources concurrently with error handling.
        
        Args:
            sources: List of sources to validate
            
        Returns:
            Dictionary mapping source names to validation results
        """
        if self.enable_error_recovery:
            # Create checkpoint for batch validation
            source_names = [source.name for source in sources]
            checkpoint_id = self.recovery_manager.create_operation_checkpoint(
                "source_validation",
                source_names,
                metadata={'total_sources': len(sources)}
            )
        
        results = {}
        
        # Create validation tasks
        tasks = []
        for source in sources:
            task = asyncio.create_task(
                self.validate_source(source),
                name=f"validate_{source.name}"
            )
            tasks.append((source.name, task))
        
        # Wait for all validations to complete
        for source_name, task in tasks:
            try:
                result = await task
                results[source_name] = result
                
                if self.enable_error_recovery:
                    # Update checkpoint
                    status = "completed" if result else "failed"
                    self.recovery_manager.checkpoint_manager.update_checkpoint(
                        checkpoint_id,
                        completed_items=[source_name] if result else [],
                        failed_items=[source_name] if not result else []
                    )
                    
            except Exception as e:
                self.logger.error(f"Error validating source {source_name}: {e}")
                results[source_name] = False
                
                if self.enable_error_recovery:
                    self.error_logger.log_error(
                        e,
                        context={'batch_validation': True, 'checkpoint_id': checkpoint_id},
                        source_name=source_name
                    )
        
        return results
    
    async def initialize_academic_repositories(self, repo_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize academic repository integrations.
        
        Args:
            repo_config: Configuration for academic repositories
        """
        try:
            self.academic_repo_manager = AcademicRepositoryManager(repo_config)
            self.logger.info("Academic repository manager initialized")
        except Exception as e:
            self.logger.error(f"Error initializing academic repositories: {e}")
    
    async def initialize_international_sources(self, intl_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize international source support.
        
        Args:
            intl_config: Configuration for international sources
        """
        try:
            self.international_source_manager = InternationalSourceManager(intl_config)
            self.logger.info("International source manager initialized")
        except Exception as e:
            self.logger.error(f"Error initializing international sources: {e}")
    
    async def search_academic_repositories(self, query: str, max_results_per_repo: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across academic repositories (arXiv, ResearchGate, etc.).
        
        Args:
            query: Search query string
            max_results_per_repo: Maximum results per repository
            
        Returns:
            Dictionary mapping repository names to search results
        """
        if not self.academic_repo_manager:
            await self.initialize_academic_repositories()
        
        if self.academic_repo_manager:
            return await self.academic_repo_manager.search_all_repositories(query, max_results_per_repo)
        else:
            self.logger.warning("Academic repository manager not available")
            return {}
    
    async def get_arxiv_papers(self, categories: List[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get papers from arXiv by categories.
        
        Args:
            categories: List of arXiv categories (e.g., ['cs.*', 'math.*'])
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        if not self.academic_repo_manager:
            await self.initialize_academic_repositories()
        
        if self.academic_repo_manager and 'arxiv' in self.academic_repo_manager.repositories:
            arxiv_repo = self.academic_repo_manager.repositories['arxiv']
            
            async with arxiv_repo:
                if categories:
                    all_papers = []
                    for category in categories:
                        papers = await arxiv_repo.search_by_category(category, max_results // len(categories))
                        all_papers.extend(papers)
                    return all_papers[:max_results]
                else:
                    return await arxiv_repo.get_recent_papers(max_results=max_results)
        
        return []
    
    async def discover_institutional_repositories(self, seed_urls: List[str]) -> List[SourceConfig]:
        """
        Discover institutional repositories from seed URLs.
        
        Args:
            seed_urls: List of seed URLs to analyze
            
        Returns:
            List of discovered institutional repository configurations
        """
        discovered = []
        
        if not self.international_source_manager:
            await self.initialize_international_sources()
        
        if self.international_source_manager:
            async with self.international_source_manager:
                for url in seed_urls:
                    try:
                        # Analyze URL to determine if it's an institutional repository
                        content, encoding = await self.international_source_manager.fetch_with_encoding_detection(url)
                        
                        if content:
                            language, confidence = self.international_source_manager.detect_language(content)
                            
                            # Check if it looks like an institutional repository
                            if self._is_institutional_repository(content, url):
                                source_config = await self._create_institutional_source_config(url, content, language)
                                if source_config:
                                    discovered.append(source_config)
                                    
                    except Exception as e:
                        self.logger.error(f"Error analyzing institutional repository {url}: {e}")
        
        return discovered
    
    def _is_institutional_repository(self, content: str, url: str) -> bool:
        """Check if URL/content represents an institutional repository."""
        # Look for common institutional repository indicators
        repo_indicators = [
            'dspace', 'eprints', 'digital repository', 'institutional repository',
            'thesis', 'dissertation', 'research papers', 'academic publications'
        ]
        
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Check for .edu domains or repository software
        if '.edu' in url_lower or any(indicator in content_lower for indicator in repo_indicators):
            return True
        
        # Check for common repository URL patterns
        repo_patterns = ['/dspace/', '/eprints/', '/repository/', '/digital/', '/ir/']
        if any(pattern in url_lower for pattern in repo_patterns):
            return True
        
        return False
    
    async def _create_institutional_source_config(self, url: str, content: str, language: str) -> Optional[SourceConfig]:
        """Create source configuration for institutional repository."""
        try:
            parsed_url = urlparse(url)
            
            # Extract basic information
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            title_elem = soup.find('title')
            name = title_elem.get_text().strip() if title_elem else parsed_url.netloc
            
            desc_elem = soup.find('meta', attrs={'name': 'description'})
            description = desc_elem.get('content', '').strip() if desc_elem else ""
            
            # Determine repository type
            repo_type = 'generic'
            if 'dspace' in content.lower():
                repo_type = 'dspace'
            elif 'eprints' in content.lower():
                repo_type = 'eprints'
            
            # Get language-specific patterns
            if self.international_source_manager:
                lang_patterns = self.international_source_manager.get_language_specific_patterns(language)
                pdf_patterns = lang_patterns.get('pdf_patterns', [r'\.pdf$'])
            else:
                pdf_patterns = [r'\.pdf$']
            
            return SourceConfig(
                name=name,
                base_url=f"{parsed_url.scheme}://{parsed_url.netloc}",
                description=description,
                scraping_strategy=ScrapingStrategy.STATIC_HTML,
                pdf_patterns=pdf_patterns,
                languages=[language],
                rate_limit=2.0,
                max_depth=3,
                source_type='institutional_repository',
                institution=self._extract_institution_name(name, parsed_url.netloc),
                is_active=True
            )
            
        except Exception as e:
            self.logger.error(f"Error creating institutional source config: {e}")
            return None
    
    def _extract_institution_name(self, name: str, domain: str) -> str:
        """Extract institution name from repository name or domain."""
        # Clean up common repository suffixes
        clean_name = re.sub(r'\s*(digital\s+)?(repository|archive|library)\s*', '', name, flags=re.IGNORECASE)
        
        if len(clean_name) > 5:
            return clean_name.strip()
        
        # Extract from domain
        if '.edu' in domain:
            parts = domain.replace('.edu', '').split('.')
            return parts[-1].title() + ' University'
        
        return domain.replace('www.', '').title()
    
    async def scrape_international_source(self, source_config: SourceConfig) -> List[str]:
        """
        Scrape an international source with language-aware processing.
        
        Args:
            source_config: Source configuration
            
        Returns:
            List of discovered PDF URLs
        """
        if not self.international_source_manager:
            await self.initialize_international_sources()
        
        if self.international_source_manager:
            async with self.international_source_manager:
                return await self.international_source_manager.scrape_international_source(source_config)
        
        return []
    
    async def get_regional_academic_databases(self, region: str) -> List[Dict[str, Any]]:
        """
        Get academic databases for a specific region.
        
        Args:
            region: Region identifier (e.g., 'china', 'japan', 'europe')
            
        Returns:
            List of database configurations
        """
        if not self.international_source_manager:
            await self.initialize_international_sources()
        
        if self.international_source_manager:
            return await self.international_source_manager.get_regional_databases(region)
        
        return []
    
    def add_academic_repository_sources(self) -> None:
        """Add major academic repositories as sources."""
        # arXiv
        arxiv_source = SourceConfig(
            name="arXiv.org",
            base_url="https://arxiv.org/",
            description="Open access research papers in physics, mathematics, computer science, and more",
            scraping_strategy=ScrapingStrategy.API_ENDPOINT,
            rate_limit=3.0,
            max_depth=1,
            pdf_patterns=[r'\.pdf$'],
            subject_areas=["computer_science", "mathematics", "physics"],
            institution="Cornell University",
            source_type="academic_repository",
            is_active=True
        )
        self.sources['arxiv'] = arxiv_source
        
        # ResearchGate
        researchgate_source = SourceConfig(
            name="ResearchGate",
            base_url="https://www.researchgate.net/",
            description="Social network for researchers and academics",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            rate_limit=5.0,
            max_depth=2,
            pdf_patterns=[r'\.pdf$'],
            subject_areas=["all"],
            source_type="academic_repository",
            is_active=True
        )
        self.sources['researchgate'] = researchgate_source
        
        # Google Scholar (use with caution)
        scholar_source = SourceConfig(
            name="Google Scholar",
            base_url="https://scholar.google.com/",
            description="Academic search engine for scholarly literature",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            rate_limit=10.0,  # Very conservative
            max_depth=1,
            pdf_patterns=[r'\.pdf$'],
            subject_areas=["all"],
            source_type="academic_repository",
            is_active=False  # Disabled by default due to anti-bot measures
        )
        self.sources['google_scholar'] = scholar_source
        
        self.logger.info("Added major academic repository sources")
    
    def add_international_repository_sources(self) -> None:
        """Add major international academic repositories."""
        international_repos = [
            {
                'id': 'hal_france',
                'name': 'HAL (Hyper Articles en Ligne)',
                'url': 'https://hal.archives-ouvertes.fr/',
                'language': 'fr',
                'country': 'France',
                'subjects': ['all']
            },
            {
                'id': 'cinii_japan',
                'name': 'CiNii (NII Academic Content Portal)',
                'url': 'https://ci.nii.ac.jp/',
                'language': 'ja',
                'country': 'Japan',
                'subjects': ['all']
            },
            {
                'id': 'cnki_china',
                'name': 'CNKI (China National Knowledge Infrastructure)',
                'url': 'https://www.cnki.net/',
                'language': 'zh',
                'country': 'China',
                'subjects': ['all']
            },
            {
                'id': 'scielo',
                'name': 'SciELO',
                'url': 'https://scielo.org/',
                'language': 'es',
                'country': 'Latin America',
                'subjects': ['science', 'medicine', 'humanities']
            }
        ]
        
        for repo in international_repos:
            source_config = SourceConfig(
                name=repo['name'],
                base_url=repo['url'],
                description=f"Academic repository from {repo['country']}",
                scraping_strategy=ScrapingStrategy.STATIC_HTML,
                rate_limit=3.0,
                max_depth=2,
                pdf_patterns=[r'\.pdf$'],
                subject_areas=repo['subjects'],
                languages=[repo['language']],
                country=repo['country'],
                source_type="international_repository",
                is_active=True
            )
            self.sources[repo['id']] = source_config
        
        self.logger.info("Added international repository sources")
    
    async def expand_source_coverage(self) -> Dict[str, int]:
        """
        Expand source coverage by adding academic repositories and international sources.
        
        Returns:
            Dictionary with counts of added sources
        """
        initial_count = len(self.sources)
        
        # Add major academic repositories
        self.add_academic_repository_sources()
        
        # Add international repositories
        self.add_international_repository_sources()
        
        # Initialize managers if not already done
        if not self.academic_repo_manager:
            await self.initialize_academic_repositories()
        
        if not self.international_source_manager:
            await self.initialize_international_sources()
        
        final_count = len(self.sources)
        
        return {
            'initial_sources': initial_count,
            'final_sources': final_count,
            'added_sources': final_count - initial_count,
            'academic_repositories': 3,  # arXiv, ResearchGate, Google Scholar
            'international_repositories': 4  # HAL, CiNii, CNKI, SciELO
        }
    
    def get_source_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for source operations.
        
        Returns:
            Dictionary containing error statistics
        """
        if not self.enable_error_recovery:
            return {'error_recovery_disabled': True}
        
        stats = {
            'circuit_breakers': {},
            'recovery_stats': self.recovery_manager.get_recovery_statistics(),
            'health_summary': self.get_source_health_summary()
        }
        
        # Add circuit breaker states
        for source_name, breaker in self.circuit_breakers.items():
            stats['circuit_breakers'][source_name] = breaker.get_state()
        
        return stats
    
    def reset_source_circuit_breaker(self, source_name: str) -> bool:
        """
        Manually reset a circuit breaker for a source.
        
        Args:
            source_name: Name of the source
            
        Returns:
            True if reset successful, False if breaker not found
        """
        if source_name in self.circuit_breakers:
            breaker = self.circuit_breakers[source_name]
            breaker.state = breaker.CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            self.logger.info(f"Manually reset circuit breaker for source {source_name}")
            return True
        
        return False
    
    async def recover_from_source_failures(self) -> Dict[str, Any]:
        """
        Attempt to recover from source failures by re-validating unhealthy sources.
        
        Returns:
            Dictionary containing recovery results
        """
        if not self.enable_error_recovery:
            return {'error_recovery_disabled': True}
        
        # Find unhealthy sources
        unhealthy_sources = []
        for source_name, health_info in self.health_status.items():
            if health_info.get('status') == 'unhealthy':
                if source_name in self.sources:
                    unhealthy_sources.append(self.sources[source_name])
        
        if not unhealthy_sources:
            return {'message': 'No unhealthy sources found', 'recovered_sources': []}
        
        self.logger.info(f"Attempting to recover {len(unhealthy_sources)} unhealthy sources")
        
        # Validate unhealthy sources
        recovery_results = await self.validate_sources_batch(unhealthy_sources)
        
        # Count recoveries
        recovered_sources = [
            source_name for source_name, result in recovery_results.items()
            if result
        ]
        
        recovery_stats = {
            'attempted_recovery': len(unhealthy_sources),
            'recovered_sources': recovered_sources,
            'recovery_count': len(recovered_sources),
            'recovery_rate': len(recovered_sources) / len(unhealthy_sources) * 100 if unhealthy_sources else 0
        }
        
        self.logger.info(
            f"Recovery completed: {len(recovered_sources)}/{len(unhealthy_sources)} sources recovered"
        )
        
        return recovery_stats