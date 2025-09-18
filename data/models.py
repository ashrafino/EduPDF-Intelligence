"""
Enhanced data models for the educational PDF scraper.
Provides comprehensive metadata structures and configuration classes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path


class AcademicLevel(Enum):
    """Academic level classification for educational content."""
    HIGH_SCHOOL = "high_school"
    UNDERGRADUATE = "undergraduate" 
    GRADUATE = "graduate"
    POSTGRADUATE = "postgraduate"
    UNKNOWN = "unknown"


class TaskType(Enum):
    """Types of processing tasks in the queue."""
    DOWNLOAD = "download"
    EXTRACT_METADATA = "extract_metadata"
    CLASSIFY_CONTENT = "classify_content"
    DEDUPLICATE = "deduplicate"


class ScrapingStrategy(Enum):
    """Different strategies for scraping content from sources."""
    STATIC_HTML = "static_html"
    DYNAMIC_JAVASCRIPT = "dynamic_javascript"
    API_ENDPOINT = "api_endpoint"
    RSS_FEED = "rss_feed"


@dataclass
class PDFMetadata:
    """
    Comprehensive metadata for educational PDF documents.
    Supports all requirements for search, classification, and organization.
    """
    # Basic file information
    filename: str
    file_path: str
    file_size: int
    
    # Content metadata
    title: str
    authors: List[str] = field(default_factory=list)
    institution: str = ""
    subject_area: str = ""
    academic_level: AcademicLevel = AcademicLevel.UNKNOWN
    
    # Language and content analysis
    language: str = "en"
    language_confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    
    # Quality metrics
    page_count: int = 0
    text_ratio: float = 0.0  # Ratio of text to images
    quality_score: float = 0.0
    readability_score: float = 0.0
    
    # Deduplication and similarity
    content_hash: str = ""
    similarity_hash: str = ""
    
    # Source tracking
    download_source: str = ""
    source_url: str = ""
    
    # Timestamps
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    download_date: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    course_code: str = ""
    isbn: str = ""
    doi: str = ""
    
    # Processing status
    is_processed: bool = False
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class SourceConfig:
    """
    Configuration for PDF sources with flexible scraping strategies.
    Supports diverse academic repositories and content types.
    """
    # Basic source information
    name: str
    base_url: str
    description: str = ""
    
    # Scraping configuration
    scraping_strategy: ScrapingStrategy = ScrapingStrategy.STATIC_HTML
    rate_limit: float = 1.0  # Seconds between requests
    max_depth: int = 3  # Maximum crawling depth
    
    # Content filtering
    pdf_patterns: List[str] = field(default_factory=list)  # URL patterns for PDFs
    exclude_patterns: List[str] = field(default_factory=list)  # Patterns to exclude
    
    # Classification
    subject_areas: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    academic_levels: List[AcademicLevel] = field(default_factory=list)
    
    # Source management
    is_active: bool = True
    priority: int = 1  # Higher numbers = higher priority
    last_crawled: Optional[datetime] = None
    
    # Authentication and headers
    headers: Dict[str, str] = field(default_factory=dict)
    auth_required: bool = False
    api_key: str = ""
    
    # Performance settings
    concurrent_downloads: int = 5
    timeout: int = 30
    retry_attempts: int = 3
    
    # Quality filters
    min_file_size: int = 1024  # Minimum file size in bytes
    max_file_size: int = 100 * 1024 * 1024  # Maximum file size (100MB)
    min_pages: int = 5
    
    # Metadata
    institution: str = ""
    country: str = ""
    source_type: str = "university"  # university, repository, journal, etc.


@dataclass
class ProcessingTask:
    """
    Task queue item for processing operations.
    Supports priority-based processing and retry logic.
    """
    task_id: str
    task_type: TaskType
    url: str
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    
    # Task-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_config: Optional[SourceConfig] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status tracking
    status: str = "pending"  # pending, processing, completed, failed
    error_message: str = ""
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if task has expired based on creation time."""
        age = datetime.now() - self.created_at
        return age.total_seconds() > (max_age_hours * 3600)
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == "failed"


@dataclass
class CollectionStats:
    """
    Statistics for tracking collection progress and quality.
    """
    total_sources: int = 0
    active_sources: int = 0
    total_pdfs_found: int = 0
    total_pdfs_downloaded: int = 0
    total_pdfs_processed: int = 0
    
    # Quality metrics
    high_quality_pdfs: int = 0
    duplicates_found: int = 0
    duplicates_removed: int = 0
    
    # Subject area breakdown
    subject_counts: Dict[str, int] = field(default_factory=dict)
    academic_level_counts: Dict[str, int] = field(default_factory=dict)
    language_counts: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    avg_download_speed: float = 0.0  # MB/s
    avg_processing_time: float = 0.0  # seconds per PDF
    
    # Error tracking
    download_errors: int = 0
    processing_errors: int = 0
    
    # Timestamps
    collection_started: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)