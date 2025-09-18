"""
Main entry point for the enhanced educational PDF scraper.
This replaces the original scrapper.py with a modular, configurable system.
"""

import asyncio
import logging
from pathlib import Path

from config.settings import config_manager
from utils.logging_setup import setup_logging


def main():
    """Main function to run the enhanced PDF scraper."""
    # Load configuration
    app_config = config_manager.load_app_config()
    sources = config_manager.load_sources()
    
    # Setup logging
    setup_logging(app_config.log_level, app_config.log_file)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(app_config.base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Enhanced Educational PDF Scraper starting...")
    logger.info(f"Configuration loaded: {len(sources)} sources configured")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # TODO: Initialize and run the scraper controller
    # This will be implemented in subsequent tasks
    logger.info("Scraper initialization complete. Ready for implementation of scraping components.")


if __name__ == "__main__":
    main()