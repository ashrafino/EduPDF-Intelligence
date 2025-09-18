"""
Data module for the educational PDF scraper.
Provides data models, database management, and storage utilities.
"""

from .models import (
    PDFMetadata,
    SourceConfig, 
    ProcessingTask,
    CollectionStats,
    AcademicLevel,
    TaskType,
    ScrapingStrategy
)

from .database import DatabaseManager, DatabaseMigration
from .init_db import initialize_database

__all__ = [
    'PDFMetadata',
    'SourceConfig',
    'ProcessingTask', 
    'CollectionStats',
    'AcademicLevel',
    'TaskType',
    'ScrapingStrategy',
    'DatabaseManager',
    'DatabaseMigration',
    'initialize_database'
]