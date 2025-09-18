"""
Main entry point for the enhanced educational PDF scraper.
Integrates all components including the advanced organization system.
"""

import asyncio
import logging
import argparse
from pathlib import Path

from config.settings import config_manager
from utils.logging_setup import setup_logging
from controllers.simple_scraper_controller import SimplifiedScraperController


async def run_scraper(max_pdfs: int = 20, organize_immediately: bool = True):
    """Run the complete scraping and organization workflow."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize the simplified scraper controller
        controller = SimplifiedScraperController()
        
        # Start the complete scrape and organize workflow
        logger.info("🚀 Starting complete scrape, download, and organize workflow...")
        results = await controller.scrape_and_organize(max_pdfs_per_source=max_pdfs)
        
        # Display formatted results
        print("\n" + controller.get_results_summary(results))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in scraping workflow: {e}")
        raise


async def reorganize_collection():
    """Reorganize existing PDF collection with current settings."""
    logger = logging.getLogger(__name__)
    
    try:
        from utils.advanced_organization import AdvancedOrganizationSystem
        from config.settings import config_manager
        
        app_config = config_manager.load_app_config()
        org_system = AdvancedOrganizationSystem(app_config.base_output_dir)
        
        logger.info("Reorganizing existing PDF collection...")
        
        # Find existing PDFs in downloads directory
        downloads_dir = Path(app_config.base_output_dir) / "downloads"
        if not downloads_dir.exists():
            logger.warning("No downloads directory found. Run scraping first.")
            return {"error": "No downloads directory found"}
        
        # Find all PDFs
        pdf_files = list(downloads_dir.rglob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found to reorganize")
            return {"error": "No PDF files found"}
        
        logger.info(f"Found {len(pdf_files)} PDFs to reorganize")
        
        # Process PDFs and organize
        from processors.metadata_extractor import PDFMetadataExtractor
        extractor = PDFMetadataExtractor()
        
        processed_pdfs = []
        for pdf_file in pdf_files:
            try:
                metadata = extractor.extract_pdf_metadata(str(pdf_file))
                processed_pdfs.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to process {pdf_file.name}: {e}")
        
        # Organize collection
        organized_collection = org_system.organize_collection(processed_pdfs)
        summary = org_system.get_organization_summary(organized_collection)
        
        logger.info("Reorganization completed successfully!")
        logger.info(f"PDFs reorganized: {summary.get('total_pdfs', 0)}")
        logger.info(f"Content groups: {summary.get('content_groups', 0)}")
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Error in reorganization: {e}")
        raise


async def get_collection_status():
    """Get current status of the PDF collection."""
    logger = logging.getLogger(__name__)
    
    try:
        from config.settings import config_manager
        
        app_config = config_manager.load_app_config()
        base_dir = Path(app_config.base_output_dir)
        
        # Check downloads directory
        downloads_dir = base_dir / "downloads"
        pdf_count = len(list(downloads_dir.rglob("*.pdf"))) if downloads_dir.exists() else 0
        
        # Check organization data
        org_data_file = base_dir / "organization_data.json"
        org_info = {}
        if org_data_file.exists():
            try:
                import json
                with open(org_data_file, 'r') as f:
                    org_data = json.load(f)
                    org_info = org_data.get('organization_metadata', {})
            except:
                pass
        
        logger.info("📊 Collection Status:")
        logger.info(f"   Downloaded PDFs: {pdf_count}")
        logger.info(f"   Output Directory: {base_dir}")
        logger.info(f"   Organization Data: {'✅ Available' if org_data_file.exists() else '❌ Not found'}")
        
        if org_info:
            logger.info(f"   Organized PDFs: {org_info.get('total_pdfs', 0)}")
            logger.info(f"   Content Groups: {org_info.get('created_groups', 0)}")
        
        # Check folder structure
        if base_dir.exists():
            subdirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
            logger.info(f"   Directories: {', '.join(subdirs) if subdirs else 'None'}")
        
        return {
            "downloaded_pdfs": pdf_count,
            "base_directory": str(base_dir),
            "organization_available": org_data_file.exists(),
            "organization_info": org_info
        }
        
    except Exception as e:
        logger.error(f"Error getting collection status: {e}")
        raise


def main():
    """Main function to run the enhanced PDF scraper."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Educational PDF Scraper with Advanced Organization')
    parser.add_argument('--max-pdfs', type=int, default=50, 
                       help='Maximum PDFs to scrape per source (default: 50)')
    parser.add_argument('--organize-immediately', action='store_true',
                       help='Organize PDFs immediately during scraping')
    parser.add_argument('--reorganize', action='store_true',
                       help='Reorganize existing PDF collection')
    parser.add_argument('--status', action='store_true',
                       help='Show current collection status')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Load configuration
    app_config = config_manager.load_app_config()
    sources = config_manager.load_sources()
    
    # Setup logging
    setup_logging(args.log_level, app_config.log_file)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(app_config.base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Enhanced Educational PDF Scraper with Advanced Organization starting...")
    logger.info(f"Configuration loaded: {len(sources)} sources configured")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Run the appropriate workflow
    try:
        if args.status:
            # Show collection status
            asyncio.run(get_collection_status())
        elif args.reorganize:
            # Reorganize existing collection
            asyncio.run(reorganize_collection())
        else:
            # Run normal scraping workflow
            asyncio.run(run_scraper(
                max_pdfs=args.max_pdfs,
                organize_immediately=args.organize_immediately
            ))
        
        logger.info("Workflow completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()