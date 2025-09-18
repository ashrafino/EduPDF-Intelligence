#!/usr/bin/env python3
"""
Database initialization script for the educational PDF scraper.
Creates and initializes the database with proper schema and sample data.
"""

import logging
from pathlib import Path
from datetime import datetime

from .database import DatabaseManager, DatabaseMigration
from .models import SourceConfig, ScrapingStrategy, AcademicLevel


def initialize_database(db_path: str = "data/scraper.db", add_sample_sources: bool = True) -> DatabaseManager:
    """
    Initialize the database with schema and optionally add sample sources.
    
    Args:
        db_path: Path to the database file
        add_sample_sources: Whether to add sample source configurations
    
    Returns:
        DatabaseManager instance
    """
    logger = logging.getLogger(__name__)
    
    # Create database manager
    db_manager = DatabaseManager(db_path)
    logger.info(f"Database initialized at {db_path}")
    
    # Validate schema
    migration = DatabaseMigration(db_manager)
    if not migration.validate_schema():
        logger.error("Database schema validation failed")
        raise RuntimeError("Database schema is invalid")
    
    # Add sample sources if requested
    if add_sample_sources:
        add_sample_source_configs(db_manager)
        logger.info("Sample source configurations added")
    
    return db_manager


def add_sample_source_configs(db_manager: DatabaseManager):
    """Add sample source configurations for testing and initial setup."""
    
    sample_sources = [
        SourceConfig(
            name="MIT OpenCourseWare",
            base_url="https://ocw.mit.edu",
            description="MIT's free online course materials",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            rate_limit=2.0,
            max_depth=4,
            pdf_patterns=["*.pdf", "/courses/*/download/*.pdf"],
            subject_areas=["Computer Science", "Mathematics", "Engineering", "Physics"],
            languages=["en"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE],
            priority=5,
            institution="Massachusetts Institute of Technology",
            country="USA",
            source_type="university"
        ),
        
        SourceConfig(
            name="Stanford CS Course Materials",
            base_url="https://cs.stanford.edu",
            description="Stanford Computer Science course materials",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            rate_limit=1.5,
            max_depth=3,
            pdf_patterns=["/courses/*/handouts/*.pdf", "/courses/*/lectures/*.pdf"],
            subject_areas=["Computer Science", "Artificial Intelligence"],
            languages=["en"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE],
            priority=4,
            institution="Stanford University",
            country="USA",
            source_type="university"
        ),
        
        SourceConfig(
            name="arXiv Computer Science",
            base_url="https://arxiv.org",
            description="arXiv preprint repository - Computer Science section",
            scraping_strategy=ScrapingStrategy.API_ENDPOINT,
            rate_limit=3.0,
            max_depth=1,
            pdf_patterns=["/pdf/*.pdf"],
            subject_areas=["Computer Science", "Machine Learning", "Artificial Intelligence"],
            languages=["en"],
            academic_levels=[AcademicLevel.GRADUATE, AcademicLevel.POSTGRADUATE],
            priority=3,
            institution="Cornell University",
            country="USA",
            source_type="repository"
        ),
        
        SourceConfig(
            name="UC Berkeley EECS",
            base_url="https://eecs.berkeley.edu",
            description="UC Berkeley Electrical Engineering and Computer Sciences",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            rate_limit=2.0,
            max_depth=3,
            pdf_patterns=["/courses/*/notes/*.pdf", "/research/*/papers/*.pdf"],
            subject_areas=["Computer Science", "Electrical Engineering"],
            languages=["en"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE],
            priority=4,
            institution="University of California, Berkeley",
            country="USA",
            source_type="university"
        ),
        
        SourceConfig(
            name="Carnegie Mellon SCS",
            base_url="https://www.cs.cmu.edu",
            description="Carnegie Mellon School of Computer Science",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            rate_limit=1.5,
            max_depth=3,
            pdf_patterns=["/courses/*/lectures/*.pdf", "/courses/*/assignments/*.pdf"],
            subject_areas=["Computer Science", "Robotics", "Machine Learning"],
            languages=["en"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE],
            priority=4,
            institution="Carnegie Mellon University",
            country="USA",
            source_type="university"
        )
    ]
    
    # Insert sample sources
    for source in sample_sources:
        try:
            source_id = db_manager.insert_source_config(source)
            logging.getLogger(__name__).info(f"Added source: {source.name} (ID: {source_id})")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to add source {source.name}: {e}")


def main():
    """Command-line interface for database initialization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize the PDF scraper database")
    parser.add_argument("--db-path", default="data/scraper.db", 
                       help="Path to database file (default: data/scraper.db)")
    parser.add_argument("--no-samples", action="store_true",
                       help="Don't add sample source configurations")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize database
        db_manager = initialize_database(
            db_path=args.db_path,
            add_sample_sources=not args.no_samples
        )
        
        print(f"✅ Database successfully initialized at {args.db_path}")
        
        # Show statistics
        active_sources = db_manager.get_active_sources()
        print(f"📊 Active sources configured: {len(active_sources)}")
        
        for source in active_sources:
            print(f"   • {source.name} ({source.institution})")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())