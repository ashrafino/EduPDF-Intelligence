"""
Database schema and management for the educational PDF scraper.
Provides SQLite storage with indexing for efficient metadata queries.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

from .models import PDFMetadata, SourceConfig, ProcessingTask, CollectionStats, AcademicLevel, TaskType, ScrapingStrategy


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for PDF metadata and configuration.
    Provides CRUD operations with proper indexing and migration support.
    """
    
    def __init__(self, db_path: str = "data/scraper.db"):
        """Initialize database manager with specified database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.version = 1
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Initialize database with schema and indexes."""
        with self.get_connection() as conn:
            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check current version
            current_version = self._get_schema_version(conn)
            
            if current_version < self.version:
                self._create_schema(conn)
                self._create_indexes(conn)
                self._update_schema_version(conn)
                conn.commit()
                logger.info(f"Database initialized with schema version {self.version}")
    
    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version."""
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        return result[0] if result[0] is not None else 0
    
    def _update_schema_version(self, conn: sqlite3.Connection):
        """Update schema version."""
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (self.version,))
    
    def _create_schema(self, conn: sqlite3.Connection):
        """Create database schema with all required tables."""
        
        # PDF metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pdf_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT UNIQUE NOT NULL,
                file_size INTEGER,
                
                -- Content metadata
                title TEXT,
                authors TEXT,  -- JSON array
                institution TEXT,
                subject_area TEXT,
                academic_level TEXT,
                
                -- Language and content
                language TEXT,
                language_confidence REAL,
                keywords TEXT,  -- JSON array
                
                -- Quality metrics
                page_count INTEGER,
                text_ratio REAL,
                quality_score REAL,
                readability_score REAL,
                
                -- Deduplication
                content_hash TEXT UNIQUE,
                similarity_hash TEXT,
                
                -- Source tracking
                download_source TEXT,
                source_url TEXT,
                
                -- Timestamps
                created_date TIMESTAMP,
                last_modified TIMESTAMP,
                download_date TIMESTAMP,
                
                -- Additional metadata
                description TEXT,
                tags TEXT,  -- JSON array
                course_code TEXT,
                isbn TEXT,
                doi TEXT,
                
                -- Processing status
                is_processed BOOLEAN DEFAULT FALSE,
                processing_errors TEXT  -- JSON array
            )
        """)
        
        # Source configuration table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                base_url TEXT NOT NULL,
                description TEXT,
                
                -- Scraping configuration
                scraping_strategy TEXT,
                rate_limit REAL,
                max_depth INTEGER,
                
                -- Content filtering
                pdf_patterns TEXT,  -- JSON array
                exclude_patterns TEXT,  -- JSON array
                
                -- Classification
                subject_areas TEXT,  -- JSON array
                languages TEXT,  -- JSON array
                academic_levels TEXT,  -- JSON array
                
                -- Source management
                is_active BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 1,
                last_crawled TIMESTAMP,
                
                -- Authentication
                headers TEXT,  -- JSON object
                auth_required BOOLEAN DEFAULT FALSE,
                api_key TEXT,
                
                -- Performance settings
                concurrent_downloads INTEGER DEFAULT 5,
                timeout INTEGER DEFAULT 30,
                retry_attempts INTEGER DEFAULT 3,
                
                -- Quality filters
                min_file_size INTEGER,
                max_file_size INTEGER,
                min_pages INTEGER,
                
                -- Metadata
                institution TEXT,
                country TEXT,
                source_type TEXT
            )
        """)
        
        # Processing tasks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                url TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                
                -- Task data
                metadata TEXT,  -- JSON object
                source_config_id INTEGER,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                
                -- Status
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                
                FOREIGN KEY (source_config_id) REFERENCES source_configs (id)
            )
        """)
        
        # Collection statistics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- Counts
                total_sources INTEGER DEFAULT 0,
                active_sources INTEGER DEFAULT 0,
                total_pdfs_found INTEGER DEFAULT 0,
                total_pdfs_downloaded INTEGER DEFAULT 0,
                total_pdfs_processed INTEGER DEFAULT 0,
                
                -- Quality metrics
                high_quality_pdfs INTEGER DEFAULT 0,
                duplicates_found INTEGER DEFAULT 0,
                duplicates_removed INTEGER DEFAULT 0,
                
                -- Breakdowns (JSON objects)
                subject_counts TEXT,
                academic_level_counts TEXT,
                language_counts TEXT,
                
                -- Performance metrics
                avg_download_speed REAL DEFAULT 0.0,
                avg_processing_time REAL DEFAULT 0.0,
                
                -- Error tracking
                download_errors INTEGER DEFAULT 0,
                processing_errors INTEGER DEFAULT 0,
                
                -- Timestamps
                collection_started TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for efficient queries."""
        
        # PDF metadata indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_pdf_content_hash ON pdf_metadata(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_similarity_hash ON pdf_metadata(similarity_hash)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_subject_area ON pdf_metadata(subject_area)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_academic_level ON pdf_metadata(academic_level)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_language ON pdf_metadata(language)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_quality_score ON pdf_metadata(quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_download_source ON pdf_metadata(download_source)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_is_processed ON pdf_metadata(is_processed)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_download_date ON pdf_metadata(download_date)",
            
            # Source config indexes
            "CREATE INDEX IF NOT EXISTS idx_source_is_active ON source_configs(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_source_priority ON source_configs(priority)",
            "CREATE INDEX IF NOT EXISTS idx_source_last_crawled ON source_configs(last_crawled)",
            
            # Processing tasks indexes
            "CREATE INDEX IF NOT EXISTS idx_task_status ON processing_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_task_priority ON processing_tasks(priority)",
            "CREATE INDEX IF NOT EXISTS idx_task_created_at ON processing_tasks(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_task_type ON processing_tasks(task_type)",
            
            # Composite indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_pdf_subject_level ON pdf_metadata(subject_area, academic_level)",
            "CREATE INDEX IF NOT EXISTS idx_pdf_quality_processed ON pdf_metadata(quality_score, is_processed)",
            "CREATE INDEX IF NOT EXISTS idx_task_status_priority ON processing_tasks(status, priority)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
        
        logger.info("Database indexes created successfully")  
  
    # PDF Metadata CRUD Operations
    
    def insert_pdf_metadata(self, metadata: PDFMetadata) -> int:
        """Insert PDF metadata into database. Returns the inserted ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO pdf_metadata (
                    filename, file_path, file_size, title, authors, institution,
                    subject_area, academic_level, language, language_confidence,
                    keywords, page_count, text_ratio, quality_score, readability_score,
                    content_hash, similarity_hash, download_source, source_url,
                    created_date, last_modified, download_date, description,
                    tags, course_code, isbn, doi, is_processed, processing_errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.filename, metadata.file_path, metadata.file_size,
                metadata.title, json.dumps(metadata.authors), metadata.institution,
                metadata.subject_area, metadata.academic_level.value, metadata.language,
                metadata.language_confidence, json.dumps(metadata.keywords),
                metadata.page_count, metadata.text_ratio, metadata.quality_score,
                metadata.readability_score, metadata.content_hash, metadata.similarity_hash,
                metadata.download_source, metadata.source_url, metadata.created_date,
                metadata.last_modified, metadata.download_date, metadata.description,
                json.dumps(metadata.tags), metadata.course_code, metadata.isbn,
                metadata.doi, metadata.is_processed, json.dumps(metadata.processing_errors)
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_pdf_metadata(self, pdf_id: int) -> Optional[PDFMetadata]:
        """Get PDF metadata by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM pdf_metadata WHERE id = ?", (pdf_id,))
            row = cursor.fetchone()
            return self._row_to_pdf_metadata(row) if row else None
    
    def get_pdf_by_hash(self, content_hash: str) -> Optional[PDFMetadata]:
        """Get PDF metadata by content hash."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM pdf_metadata WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()
            return self._row_to_pdf_metadata(row) if row else None
    
    def find_similar_pdfs(self, similarity_hash: str, threshold: float = 0.8) -> List[PDFMetadata]:
        """Find PDFs with similar content hashes."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM pdf_metadata WHERE similarity_hash = ? AND quality_score >= ?",
                (similarity_hash, threshold)
            )
            return [self._row_to_pdf_metadata(row) for row in cursor.fetchall()]
    
    def update_pdf_metadata(self, pdf_id: int, metadata: PDFMetadata) -> bool:
        """Update PDF metadata. Returns True if successful."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                UPDATE pdf_metadata SET
                    title = ?, authors = ?, institution = ?, subject_area = ?,
                    academic_level = ?, language = ?, language_confidence = ?,
                    keywords = ?, page_count = ?, text_ratio = ?, quality_score = ?,
                    readability_score = ?, similarity_hash = ?, description = ?,
                    tags = ?, course_code = ?, isbn = ?, doi = ?, is_processed = ?,
                    processing_errors = ?, last_modified = ?
                WHERE id = ?
            """, (
                metadata.title, json.dumps(metadata.authors), metadata.institution,
                metadata.subject_area, metadata.academic_level.value, metadata.language,
                metadata.language_confidence, json.dumps(metadata.keywords),
                metadata.page_count, metadata.text_ratio, metadata.quality_score,
                metadata.readability_score, metadata.similarity_hash, metadata.description,
                json.dumps(metadata.tags), metadata.course_code, metadata.isbn,
                metadata.doi, metadata.is_processed, json.dumps(metadata.processing_errors),
                datetime.now(), pdf_id
            ))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_pdf_metadata(self, pdf_id: int) -> bool:
        """Delete PDF metadata. Returns True if successful."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM pdf_metadata WHERE id = ?", (pdf_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def search_pdfs(self, 
                   subject_area: Optional[str] = None,
                   academic_level: Optional[AcademicLevel] = None,
                   language: Optional[str] = None,
                   min_quality: float = 0.0,
                   limit: int = 100) -> List[PDFMetadata]:
        """Search PDFs with filters."""
        query = "SELECT * FROM pdf_metadata WHERE quality_score >= ?"
        params = [min_quality]
        
        if subject_area:
            query += " AND subject_area = ?"
            params.append(subject_area)
        
        if academic_level:
            query += " AND academic_level = ?"
            params.append(academic_level.value)
        
        if language:
            query += " AND language = ?"
            params.append(language)
        
        query += " ORDER BY quality_score DESC LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_pdf_metadata(row) for row in cursor.fetchall()]
    
    # Source Configuration CRUD Operations
    
    def insert_source_config(self, config: SourceConfig) -> int:
        """Insert source configuration. Returns the inserted ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO source_configs (
                    name, base_url, description, scraping_strategy, rate_limit,
                    max_depth, pdf_patterns, exclude_patterns, subject_areas,
                    languages, academic_levels, is_active, priority, last_crawled,
                    headers, auth_required, api_key, concurrent_downloads, timeout,
                    retry_attempts, min_file_size, max_file_size, min_pages,
                    institution, country, source_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.name, config.base_url, config.description,
                config.scraping_strategy.value, config.rate_limit, config.max_depth,
                json.dumps(config.pdf_patterns), json.dumps(config.exclude_patterns),
                json.dumps(config.subject_areas), json.dumps(config.languages),
                json.dumps([level.value for level in config.academic_levels]),
                config.is_active, config.priority, config.last_crawled,
                json.dumps(config.headers), config.auth_required, config.api_key,
                config.concurrent_downloads, config.timeout, config.retry_attempts,
                config.min_file_size, config.max_file_size, config.min_pages,
                config.institution, config.country, config.source_type
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_source_config(self, source_id: int) -> Optional[SourceConfig]:
        """Get source configuration by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM source_configs WHERE id = ?", (source_id,))
            row = cursor.fetchone()
            return self._row_to_source_config(row) if row else None
    
    def get_active_sources(self) -> List[SourceConfig]:
        """Get all active source configurations."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM source_configs WHERE is_active = TRUE ORDER BY priority DESC"
            )
            return [self._row_to_source_config(row) for row in cursor.fetchall()]
    
    def update_source_last_crawled(self, source_id: int, timestamp: datetime) -> bool:
        """Update last crawled timestamp for a source."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE source_configs SET last_crawled = ? WHERE id = ?",
                (timestamp, source_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    # Processing Tasks CRUD Operations
    
    def insert_processing_task(self, task: ProcessingTask) -> int:
        """Insert processing task. Returns the inserted ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO processing_tasks (
                    task_id, task_type, url, priority, retry_count, max_retries,
                    metadata, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id, task.task_type.value, task.url, task.priority,
                task.retry_count, task.max_retries, json.dumps(task.metadata),
                task.created_at, task.status
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_pending_tasks(self, limit: int = 100) -> List[ProcessingTask]:
        """Get pending tasks ordered by priority."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM processing_tasks 
                WHERE status = 'pending' 
                ORDER BY priority DESC, created_at ASC 
                LIMIT ?
            """, (limit,))
            return [self._row_to_processing_task(row) for row in cursor.fetchall()]
    
    def update_task_status(self, task_id: str, status: str, error_message: str = "") -> bool:
        """Update task status and error message."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                UPDATE processing_tasks 
                SET status = ?, error_message = ?, 
                    started_at = CASE WHEN status = 'processing' THEN CURRENT_TIMESTAMP ELSE started_at END,
                    completed_at = CASE WHEN status IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE task_id = ?
            """, (status, error_message, task_id))
            conn.commit()
            return cursor.rowcount > 0
    
    # Helper methods for row conversion
    
    def _row_to_pdf_metadata(self, row: sqlite3.Row) -> PDFMetadata:
        """Convert database row to PDFMetadata object."""
        return PDFMetadata(
            filename=row['filename'],
            file_path=row['file_path'],
            file_size=row['file_size'],
            title=row['title'] or "",
            authors=json.loads(row['authors'] or '[]'),
            institution=row['institution'] or "",
            subject_area=row['subject_area'] or "",
            academic_level=AcademicLevel(row['academic_level']) if row['academic_level'] else AcademicLevel.UNKNOWN,
            language=row['language'] or "en",
            language_confidence=row['language_confidence'] or 0.0,
            keywords=json.loads(row['keywords'] or '[]'),
            page_count=row['page_count'] or 0,
            text_ratio=row['text_ratio'] or 0.0,
            quality_score=row['quality_score'] or 0.0,
            readability_score=row['readability_score'] or 0.0,
            content_hash=row['content_hash'] or "",
            similarity_hash=row['similarity_hash'] or "",
            download_source=row['download_source'] or "",
            source_url=row['source_url'] or "",
            created_date=datetime.fromisoformat(row['created_date']) if row['created_date'] else datetime.now(),
            last_modified=datetime.fromisoformat(row['last_modified']) if row['last_modified'] else datetime.now(),
            download_date=datetime.fromisoformat(row['download_date']) if row['download_date'] else datetime.now(),
            description=row['description'] or "",
            tags=json.loads(row['tags'] or '[]'),
            course_code=row['course_code'] or "",
            isbn=row['isbn'] or "",
            doi=row['doi'] or "",
            is_processed=bool(row['is_processed']),
            processing_errors=json.loads(row['processing_errors'] or '[]')
        )
    
    def _row_to_source_config(self, row: sqlite3.Row) -> SourceConfig:
        """Convert database row to SourceConfig object."""
        return SourceConfig(
            name=row['name'],
            base_url=row['base_url'],
            description=row['description'] or "",
            scraping_strategy=ScrapingStrategy(row['scraping_strategy']) if row['scraping_strategy'] else ScrapingStrategy.STATIC_HTML,
            rate_limit=row['rate_limit'] or 1.0,
            max_depth=row['max_depth'] or 3,
            pdf_patterns=json.loads(row['pdf_patterns'] or '[]'),
            exclude_patterns=json.loads(row['exclude_patterns'] or '[]'),
            subject_areas=json.loads(row['subject_areas'] or '[]'),
            languages=json.loads(row['languages'] or '["en"]'),
            academic_levels=[AcademicLevel(level) for level in json.loads(row['academic_levels'] or '[]')],
            is_active=bool(row['is_active']),
            priority=row['priority'] or 1,
            last_crawled=datetime.fromisoformat(row['last_crawled']) if row['last_crawled'] else None,
            headers=json.loads(row['headers'] or '{}'),
            auth_required=bool(row['auth_required']),
            api_key=row['api_key'] or "",
            concurrent_downloads=row['concurrent_downloads'] or 5,
            timeout=row['timeout'] or 30,
            retry_attempts=row['retry_attempts'] or 3,
            min_file_size=row['min_file_size'] or 1024,
            max_file_size=row['max_file_size'] or 100 * 1024 * 1024,
            min_pages=row['min_pages'] or 5,
            institution=row['institution'] or "",
            country=row['country'] or "",
            source_type=row['source_type'] or "university"
        )
    
    def _row_to_processing_task(self, row: sqlite3.Row) -> ProcessingTask:
        """Convert database row to ProcessingTask object."""
        return ProcessingTask(
            task_id=row['task_id'],
            task_type=TaskType(row['task_type']),
            url=row['url'],
            priority=row['priority'] or 1,
            retry_count=row['retry_count'] or 0,
            max_retries=row['max_retries'] or 3,
            metadata=json.loads(row['metadata'] or '{}'),
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            status=row['status'] or "pending",
            error_message=row['error_message'] or ""
        )


class DatabaseMigration:
    """
    Database migration and upgrade utilities.
    Handles schema changes and data migrations between versions.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def migrate_to_version(self, target_version: int) -> bool:
        """
        Migrate database to target version.
        Returns True if migration was successful.
        """
        with self.db_manager.get_connection() as conn:
            current_version = self.db_manager._get_schema_version(conn)
            
            if current_version >= target_version:
                self.logger.info(f"Database already at version {current_version}")
                return True
            
            self.logger.info(f"Migrating database from version {current_version} to {target_version}")
            
            # Apply migrations sequentially
            for version in range(current_version + 1, target_version + 1):
                if not self._apply_migration(conn, version):
                    self.logger.error(f"Failed to migrate to version {version}")
                    return False
                
                # Update version
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                conn.commit()
                self.logger.info(f"Successfully migrated to version {version}")
            
            return True
    
    def _apply_migration(self, conn: sqlite3.Connection, version: int) -> bool:
        """Apply specific migration for given version."""
        try:
            if version == 1:
                # Initial schema - already handled in _create_schema
                return True
            
            # Future migrations would be added here
            # elif version == 2:
            #     return self._migrate_v2(conn)
            
            self.logger.warning(f"No migration defined for version {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration to version {version} failed: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the current database."""
        try:
            import shutil
            shutil.copy2(self.db_manager.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    def validate_schema(self) -> bool:
        """Validate database schema integrity."""
        try:
            with self.db_manager.get_connection() as conn:
                # Check if all required tables exist
                required_tables = ['pdf_metadata', 'source_configs', 'processing_tasks', 'collection_stats']
                
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                missing_tables = set(required_tables) - existing_tables
                if missing_tables:
                    self.logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                # Check indexes
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes = {row[0] for row in cursor.fetchall()}
                
                # Verify some critical indexes exist
                critical_indexes = ['idx_pdf_content_hash', 'idx_pdf_quality_score', 'idx_source_is_active']
                missing_indexes = set(critical_indexes) - indexes
                if missing_indexes:
                    self.logger.warning(f"Missing indexes: {missing_indexes}")
                
                self.logger.info("Database schema validation completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
            return False