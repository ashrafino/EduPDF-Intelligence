"""
Main scraper controller that integrates all components including the advanced organization system.
Manages the complete workflow from PDF discovery to organized storage.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from config.settings import config_manager
from scrapers.source_manager import SourceManager
from scrapers.intelligent_crawler import IntelligentCrawler
from processors.content_analyzer import ContentAnalyzer
from processors.metadata_extractor import PDFMetadataExtractor
from data.models import PDFMetadata, ProcessingTask, TaskType
from utils.advanced_organization import AdvancedOrganizationSystem
from data.database import DatabaseManager
from utils.error_handling import with_error_handling


class ScraperController:
    """
    Main controller that orchestrates the entire PDF scraping and organization workflow.
    Integrates all components including the advanced organization system.
    """
    
    def __init__(self, config_path: str = "config"):
        """Initialize the scraper controller with all components."""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.app_config = config_manager.load_app_config()
        self.sources = config_manager.load_sources()
        
        # Initialize core components
        self.source_manager = SourceManager()
        self.content_analyzer = ContentAnalyzer()
        self.metadata_extractor = PDFMetadataExtractor()
        self.database_manager = DatabaseManager()
        
        # Crawler will be initialized per source during scraping
        self.crawler = None
        
        # Initialize advanced organization system
        self.organization_system = AdvancedOrganizationSystem(
            base_output_dir=self.app_config.base_output_dir
        )
        
        # Processing state
        self.processed_pdfs: List[PDFMetadata] = []
        self.organization_enabled = True
        self.auto_organize_threshold = 10  # Organize after every 10 PDFs
        
        self.logger.info("Scraper controller initialized with advanced organization system")
    
    @with_error_handling()
    async def start_scraping(self, max_pdfs_per_source: int = 50, 
                           organize_immediately: bool = False) -> Dict[str, any]:
        """
        Start the complete scraping and organization workflow.
        
        Args:
            max_pdfs_per_source: Maximum PDFs to scrape per source
            organize_immediately: Whether to organize PDFs immediately after each download
            
        Returns:
            Dictionary with scraping and organization results
        """
        self.logger.info("Starting enhanced PDF scraping workflow...")
        
        results = {
            "scraping_started": datetime.now(),
            "sources_processed": 0,
            "pdfs_discovered": 0,
            "pdfs_downloaded": 0,
            "pdfs_processed": 0,
            "organization_results": None,
            "errors": []
        }
        
        try:
            # Step 1: Initialize sources and validate
            await self._initialize_sources()
            
            # Step 2: Discover and download PDFs
            discovered_pdfs = await self._discover_and_download_pdfs(max_pdfs_per_source)
            results["pdfs_discovered"] = len(discovered_pdfs)
            
            # Step 3: Process downloaded PDFs
            processed_pdfs = await self._process_downloaded_pdfs(discovered_pdfs)
            results["pdfs_processed"] = len(processed_pdfs)
            self.processed_pdfs.extend(processed_pdfs)
            
            # Step 4: Organize PDFs using advanced organization system
            if self.organization_enabled and self.processed_pdfs:
                self.logger.info("Starting advanced organization of PDF collection...")
                organization_results = await self._organize_pdf_collection()
                results["organization_results"] = organization_results
            
            # Step 5: Generate final reports
            await self._generate_reports(results)
            
            results["scraping_completed"] = datetime.now()
            self.logger.info(f"Scraping workflow completed. Processed {len(self.processed_pdfs)} PDFs")
            
        except Exception as e:
            self.logger.error(f"Error in scraping workflow: {e}")
            results["errors"].append(str(e))
            raise
        
        return results
    
    async def _initialize_sources(self):
        """Initialize and validate PDF sources."""
        self.logger.info("Initializing PDF sources...")
        
        # Load sources from configuration
        await self.source_manager.load_sources()
        
        # Validate source health
        await self.source_manager.check_source_health()
        
        active_sources = self.source_manager.get_active_sources()
        self.logger.info(f"Initialized {len(active_sources)} active sources")
    
    async def _discover_and_download_pdfs(self, max_per_source: int) -> List[str]:
        """Discover and download PDFs from all sources."""
        self.logger.info("Starting PDF discovery and download...")
        
        discovered_pdfs = []
        active_sources = self.source_manager.get_active_sources()
        
        for source_name, source_config in active_sources.items():
            try:
                self.logger.info(f"Processing source: {source_name}")
                
                # Initialize crawler for this source
                from scrapers.intelligent_crawler import IntelligentCrawler
                source_crawler = IntelligentCrawler(source_config)
                
                # Discover PDFs from source
                pdf_urls = await source_crawler.discover_pdfs(
                    source_config, 
                    max_pdfs=max_per_source
                )
                
                # Download discovered PDFs
                for pdf_url in pdf_urls[:max_per_source]:
                    try:
                        downloaded_path = await source_crawler.download_pdf(
                            pdf_url, 
                            source_config
                        )
                        
                        if downloaded_path:
                            discovered_pdfs.append(downloaded_path)
                            self.logger.info(f"Downloaded: {Path(downloaded_path).name}")
                            
                            # Organize immediately if requested
                            if self.organization_enabled and len(discovered_pdfs) % self.auto_organize_threshold == 0:
                                await self._organize_recent_pdfs(discovered_pdfs[-self.auto_organize_threshold:])
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to download {pdf_url}: {e}")
                        continue
            
            except Exception as e:
                self.logger.error(f"Error processing source {source_name}: {e}")
                continue
        
        self.logger.info(f"Discovered and downloaded {len(discovered_pdfs)} PDFs")
        return discovered_pdfs
    
    async def _process_downloaded_pdfs(self, pdf_paths: List[str]) -> List[PDFMetadata]:
        """Process downloaded PDFs to extract metadata and analyze content."""
        self.logger.info("Processing downloaded PDFs...")
        
        processed_pdfs = []
        
        for pdf_path in pdf_paths:
            try:
                # Extract metadata
                metadata = await self.metadata_extractor.extract_metadata(pdf_path)
                
                # Analyze content
                analysis_results = await self.content_analyzer.analyze_content(metadata)
                
                # Update metadata with analysis results
                if analysis_results:
                    metadata.keywords.extend(analysis_results.get('keywords', []))
                    metadata.subject_area = analysis_results.get('subject_area', metadata.subject_area)
                    metadata.academic_level = analysis_results.get('academic_level', metadata.academic_level)
                    metadata.quality_score = analysis_results.get('quality_score', metadata.quality_score)
                
                # Store in database
                await self.database_manager.store_pdf_metadata(metadata)
                
                processed_pdfs.append(metadata)
                self.logger.debug(f"Processed: {metadata.filename}")
                
            except Exception as e:
                self.logger.warning(f"Failed to process {pdf_path}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_pdfs)} PDFs")
        return processed_pdfs
    
    async def _organize_pdf_collection(self) -> Dict[str, any]:
        """Organize the complete PDF collection using the advanced organization system."""
        self.logger.info("Organizing PDF collection with advanced organization system...")
        
        try:
            # Organize the collection
            organized_collection = self.organization_system.organize_collection(self.processed_pdfs)
            
            # Get organization summary
            summary = self.organization_system.get_organization_summary(organized_collection)
            
            # Save organization data
            org_data_path = Path(self.app_config.base_output_dir) / "organization_data.json"
            self.organization_system.save_organization(organized_collection, str(org_data_path))
            
            self.logger.info(f"Organization complete: {summary['total_pdfs']} PDFs organized into {summary['content_groups']} groups")
            
            return {
                "organized_collection": organized_collection,
                "summary": summary,
                "organization_file": str(org_data_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error during organization: {e}")
            raise
    
    async def _organize_recent_pdfs(self, recent_pdf_paths: List[str]):
        """Organize a subset of recently downloaded PDFs."""
        try:
            # Convert paths to metadata (simplified for incremental organization)
            recent_metadata = []
            for pdf_path in recent_pdf_paths:
                try:
                    metadata = await self.metadata_extractor.extract_metadata(pdf_path)
                    recent_metadata.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to extract metadata for incremental organization: {e}")
            
            if recent_metadata:
                # Perform incremental organization
                self.organization_system.organize_collection(recent_metadata)
                self.logger.info(f"Incrementally organized {len(recent_metadata)} PDFs")
                
        except Exception as e:
            self.logger.warning(f"Error in incremental organization: {e}")
    
    async def _generate_reports(self, results: Dict[str, any]):
        """Generate comprehensive reports about the scraping and organization process."""
        self.logger.info("Generating scraping and organization reports...")
        
        try:
            reports_dir = Path(self.app_config.base_output_dir) / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Generate scraping report
            scraping_report = self._create_scraping_report(results)
            scraping_report_path = reports_dir / f"scraping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(scraping_report_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(scraping_report, f, indent=2, default=str, ensure_ascii=False)
            
            # Generate organization report if available
            if results.get("organization_results"):
                org_report = self._create_organization_report(results["organization_results"])
                org_report_path = reports_dir / f"organization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(org_report_path, 'w', encoding='utf-8') as f:
                    json.dump(org_report, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Reports generated in {reports_dir}")
            
        except Exception as e:
            self.logger.warning(f"Error generating reports: {e}")
    
    def _create_scraping_report(self, results: Dict[str, any]) -> Dict[str, any]:
        """Create a comprehensive scraping report."""
        return {
            "scraping_session": {
                "started": results.get("scraping_started"),
                "completed": results.get("scraping_completed"),
                "duration_minutes": (
                    (results.get("scraping_completed", datetime.now()) - 
                     results.get("scraping_started", datetime.now())).total_seconds() / 60
                ) if results.get("scraping_started") else 0
            },
            "statistics": {
                "sources_processed": results.get("sources_processed", 0),
                "pdfs_discovered": results.get("pdfs_discovered", 0),
                "pdfs_downloaded": results.get("pdfs_downloaded", 0),
                "pdfs_processed": results.get("pdfs_processed", 0),
                "success_rate": (
                    results.get("pdfs_processed", 0) / max(results.get("pdfs_discovered", 1), 1) * 100
                )
            },
            "configuration": {
                "base_output_dir": self.app_config.base_output_dir,
                "sources_configured": len(self.sources),
                "organization_enabled": self.organization_enabled
            },
            "errors": results.get("errors", [])
        }
    
    def _create_organization_report(self, org_results: Dict[str, any]) -> Dict[str, any]:
        """Create a comprehensive organization report."""
        summary = org_results.get("summary", {})
        
        return {
            "organization_summary": summary,
            "folder_structure": {
                "hierarchical_folders": summary.get("hierarchical_folders", 0),
                "content_groups": summary.get("content_groups", 0),
                "base_path": summary.get("base_path", "")
            },
            "content_distribution": {
                "by_subject": summary.get("subject_distribution", {}),
                "by_level": summary.get("level_distribution", {}),
                "by_group_type": summary.get("group_types", {})
            },
            "organization_metadata": {
                "organization_date": summary.get("organization_date", ""),
                "total_pdfs_organized": summary.get("total_pdfs", 0)
            }
        }
    
    async def get_collection_status(self) -> Dict[str, any]:
        """Get current status of the PDF collection and organization."""
        try:
            # Get database statistics
            db_stats = await self.database_manager.get_collection_stats()
            
            # Get organization statistics if available
            org_stats = {}
            if self.processed_pdfs:
                # Create a temporary organization to get current stats
                temp_org = self.organization_system.organize_collection(self.processed_pdfs)
                org_stats = self.organization_system.get_organization_summary(temp_org)
            
            return {
                "collection_stats": db_stats,
                "organization_stats": org_stats,
                "processed_pdfs_count": len(self.processed_pdfs),
                "organization_enabled": self.organization_enabled,
                "last_update": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection status: {e}")
            return {"error": str(e)}
    
    def enable_organization(self, enabled: bool = True):
        """Enable or disable the advanced organization system."""
        self.organization_enabled = enabled
        self.logger.info(f"Advanced organization system {'enabled' if enabled else 'disabled'}")
    
    def set_auto_organize_threshold(self, threshold: int):
        """Set the threshold for automatic organization during scraping."""
        self.auto_organize_threshold = max(1, threshold)
        self.logger.info(f"Auto-organization threshold set to {threshold} PDFs")
    
    async def reorganize_collection(self) -> Dict[str, any]:
        """Reorganize the entire PDF collection with current settings."""
        self.logger.info("Reorganizing entire PDF collection...")
        
        try:
            # Load all PDFs from database
            all_pdfs = await self.database_manager.get_all_pdf_metadata()
            
            if not all_pdfs:
                self.logger.warning("No PDFs found in database for reorganization")
                return {"error": "No PDFs found"}
            
            # Update processed PDFs list
            self.processed_pdfs = all_pdfs
            
            # Reorganize collection
            organization_results = await self._organize_pdf_collection()
            
            self.logger.info(f"Reorganization complete: {len(all_pdfs)} PDFs reorganized")
            return organization_results
            
        except Exception as e:
            self.logger.error(f"Error during reorganization: {e}")
            raise