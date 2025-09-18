"""
Configuration management system for the educational PDF scraper.
Handles application settings, source configurations, and runtime parameters.
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class ScrapingStrategy(Enum):
    """Enumeration of available scraping strategies."""
    STATIC_HTML = "static_html"
    DYNAMIC_SELENIUM = "dynamic_selenium"
    API_INTEGRATION = "api_integration"


class AcademicLevel(Enum):
    """Enumeration of academic levels for content classification."""
    HIGH_SCHOOL = "high_school"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    RESEARCH = "research"


@dataclass
class SourceConfig:
    """Configuration for a PDF source."""
    name: str
    base_url: str
    scraping_strategy: ScrapingStrategy
    rate_limit: float = 1.0  # seconds between requests
    max_depth: int = 3
    pdf_patterns: List[str] = None
    subject_areas: List[str] = None
    languages: List[str] = None
    is_active: bool = True
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.pdf_patterns is None:
            self.pdf_patterns = [r'\.pdf$']
        if self.subject_areas is None:
            self.subject_areas = []
        if self.languages is None:
            self.languages = ['en']
        if self.headers is None:
            self.headers = {}


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    max_workers: int = 4
    max_concurrent_downloads: int = 10
    download_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_file_size_mb: int = 100
    min_pages: int = 5
    max_pages: int = 1000


@dataclass
class FilterConfig:
    """Configuration for content filtering."""
    min_quality_score: float = 0.6
    educational_keywords: List[str] = None
    excluded_keywords: List[str] = None
    min_text_ratio: float = 0.1
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.educational_keywords is None:
            self.educational_keywords = [
                'course', 'lecture', 'tutorial', 'textbook', 'syllabus',
                'assignment', 'homework', 'exercise', 'solution', 'notes'
            ]
        if self.excluded_keywords is None:
            self.excluded_keywords = [
                'advertisement', 'commercial', 'marketing', 'spam'
            ]
        if self.supported_languages is None:
            self.supported_languages = ['en', 'fr', 'ar', 'es', 'de']


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    db_path: str = "data/metadata.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backup_files: int = 7


@dataclass
class AppConfig:
    """Main application configuration."""
    base_output_dir: str = "EducationalPDFs"
    log_level: str = "INFO"
    log_file: str = "scraper.log"
    processing: ProcessingConfig = None
    filtering: FilterConfig = None
    database: DatabaseConfig = None
    
    def __post_init__(self):
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.filtering is None:
            self.filtering = FilterConfig()
        if self.database is None:
            self.database = DatabaseConfig()


class ConfigManager:
    """Manages application configuration and source definitions."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.app_config_path = self.config_dir / "app_config.json"
        self.sources_config_path = self.config_dir / "sources.yaml"
        
        self._app_config: Optional[AppConfig] = None
        self._sources: Dict[str, SourceConfig] = {}
    
    def load_app_config(self) -> AppConfig:
        """Load application configuration from file or create default."""
        if self.app_config_path.exists():
            with open(self.app_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                # Convert nested dicts back to dataclasses
                if 'processing' in config_data:
                    config_data['processing'] = ProcessingConfig(**config_data['processing'])
                if 'filtering' in config_data:
                    config_data['filtering'] = FilterConfig(**config_data['filtering'])
                if 'database' in config_data:
                    config_data['database'] = DatabaseConfig(**config_data['database'])
                self._app_config = AppConfig(**config_data)
        else:
            self._app_config = AppConfig()
            self.save_app_config()
        
        return self._app_config
    
    def save_app_config(self):
        """Save application configuration to file."""
        if self._app_config:
            with open(self.app_config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self._app_config), f, indent=2, default=str)
    
    def load_sources(self) -> Dict[str, SourceConfig]:
        """Load source configurations from file or create defaults."""
        if self.sources_config_path.exists():
            with open(self.sources_config_path, 'r', encoding='utf-8') as f:
                sources_data = yaml.safe_load(f)
                for name, config in sources_data.items():
                    # Convert strategy string to enum
                    if 'scraping_strategy' in config:
                        config['scraping_strategy'] = ScrapingStrategy(config['scraping_strategy'])
                    self._sources[name] = SourceConfig(**config)
        else:
            self._create_default_sources()
            self.save_sources()
        
        return self._sources
    
    def save_sources(self):
        """Save source configurations to file."""
        sources_data = {}
        for name, config in self._sources.items():
            config_dict = asdict(config)
            # Convert enum to string for serialization
            config_dict['scraping_strategy'] = config.scraping_strategy.value
            sources_data[name] = config_dict
        
        with open(self.sources_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sources_data, f, default_flow_style=False, indent=2)
    
    def _create_default_sources(self):
        """Create default source configurations."""
        self._sources = {
            'mit_ocw_cs': SourceConfig(
                name='MIT OCW Computer Science',
                base_url='https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/',
                scraping_strategy=ScrapingStrategy.STATIC_HTML,
                subject_areas=['computer_science'],
                languages=['en']
            ),
            'mit_ocw_math': SourceConfig(
                name='MIT OCW Mathematics',
                base_url='https://ocw.mit.edu/courses/mathematics/',
                scraping_strategy=ScrapingStrategy.STATIC_HTML,
                subject_areas=['mathematics'],
                languages=['en']
            ),
            'arxiv': SourceConfig(
                name='arXiv.org',
                base_url='https://arxiv.org/',
                scraping_strategy=ScrapingStrategy.API_INTEGRATION,
                subject_areas=['computer_science', 'mathematics', 'physics'],
                languages=['en']
            ),
            'stanford_cs': SourceConfig(
                name='Stanford Computer Science',
                base_url='https://cs.stanford.edu/',
                scraping_strategy=ScrapingStrategy.STATIC_HTML,
                subject_areas=['computer_science'],
                languages=['en']
            )
        }
    
    def add_source(self, name: str, config: SourceConfig):
        """Add a new source configuration."""
        self._sources[name] = config
        self.save_sources()
    
    def remove_source(self, name: str):
        """Remove a source configuration."""
        if name in self._sources:
            del self._sources[name]
            self.save_sources()
    
    def get_active_sources(self) -> Dict[str, SourceConfig]:
        """Get only active source configurations."""
        return {name: config for name, config in self._sources.items() if config.is_active}
    
    def update_source_status(self, name: str, is_active: bool):
        """Update the active status of a source."""
        if name in self._sources:
            self._sources[name].is_active = is_active
            self.save_sources()
    
    @property
    def app_config(self) -> AppConfig:
        """Get the current application configuration."""
        if self._app_config is None:
            self.load_app_config()
        return self._app_config
    
    @property
    def sources(self) -> Dict[str, SourceConfig]:
        """Get the current source configurations."""
        if not self._sources:
            self.load_sources()
        return self._sources


# Global configuration manager instance
config_manager = ConfigManager()