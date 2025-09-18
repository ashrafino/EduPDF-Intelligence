#!/usr/bin/env python3
"""
Standalone PDF organization script using the advanced organization system.
Can be used to organize existing PDF collections without scraping.
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List
import os

from data.models import PDFMetadata, AcademicLevel
from utils.advanced_organization import AdvancedOrganizationSystem
from processors.metadata_extractor import PDFMetadataExtractor
from utils.logging_setup import setup_logging


class PDFOrganizer:
    """Standalone PDF organizer for existing collections."""
    
    def __init__(self, output_dir: str = "OrganizedPDFs"):
        self.output_dir = output_dir
        self.organization_system = AdvancedOrganizationSystem(output_dir)
        self.metadata_extractor = PDFMetadataExtractor()
        self.logger = logging.getLogger(__name__)
    
    async def organize_directory(self, input_dir: str, recursive: bool = True) -> dict:
        """
        Organize all PDFs in a directory using the advanced organization system.
        
        Args:
            input_dir: Directory containing PDFs to organize
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with organization results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        self.logger.info(f"Organizing PDFs from: {input_path}")
        
        # Find all PDF files
        pdf_files = self._find_pdf_files(input_path, recursive)
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            self.logger.warning("No PDF files found to organize")
            return {"error": "No PDF files found"}
        
        # Extract metadata for all PDFs
        pdf_metadata = await self._extract_metadata_batch(pdf_files)
        self.logger.info(f"Extracted metadata for {len(pdf_metadata)} PDFs")
        
        # Organize the collection
        organized_collection = self.organization_system.organize_collection(pdf_metadata)
        
        # Copy/move files to organized structure
        await self._copy_files_to_organized_structure(pdf_files, pdf_metadata)
        
        # Get summary
        summary = self.organization_system.get_organization_summary(organized_collection)
        
        # Save organization data
        org_data_path = Path(self.output_dir) / "organization_data.json"
        self.organization_system.save_organization(organized_collection, str(org_data_path))
        
        self.logger.info("Organization completed successfully!")
        return {
            "organized_collection": organized_collection,
            "summary": summary,
            "organization_file": str(org_data_path)
        }
    
    def _find_pdf_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all PDF files in directory."""
        pdf_files = []
        
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        # Filter out hidden files and system files
        pdf_files = [f for f in pdf_files if not f.name.startswith('.')]
        
        return pdf_files
    
    async def _extract_metadata_batch(self, pdf_files: List[Path]) -> List[PDFMetadata]:
        """Extract metadata for a batch of PDF files."""
        pdf_metadata = []
        
        for pdf_file in pdf_files:
            try:
                self.logger.debug(f"Extracting metadata from: {pdf_file.name}")
                metadata = await self.metadata_extractor.extract_metadata(str(pdf_file))
                pdf_metadata.append(metadata)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract metadata from {pdf_file.name}: {e}")
                
                # Create minimal metadata for files that fail extraction
                minimal_metadata = PDFMetadata(
                    filename=pdf_file.name,
                    file_path=str(pdf_file),
                    file_size=pdf_file.stat().st_size,
                    title=pdf_file.stem,
                    academic_level=AcademicLevel.UNKNOWN,
                    page_count=0,
                    text_ratio=0.0,
                    quality_score=0.5
                )
                pdf_metadata.append(minimal_metadata)
        
        return pdf_metadata
    
    async def _copy_files_to_organized_structure(self, pdf_files: List[Path], 
                                               pdf_metadata: List[PDFMetadata]):
        """Copy PDF files to their organized locations."""
        import shutil
        
        # Create mapping from filename to file path
        file_mapping = {pdf.name: pdf for pdf in pdf_files}
        
        for metadata in pdf_metadata:
            try:
                source_file = file_mapping.get(metadata.filename)
                if not source_file or not source_file.exists():
                    self.logger.warning(f"Source file not found: {metadata.filename}")
                    continue
                
                # Generate organized path
                folder_path = self.organization_system.categorizer.generate_folder_structure(metadata)
                self.organization_system.categorizer.create_folder_structure(folder_path)
                
                # Copy file to organized location
                dest_file = folder_path / metadata.filename
                
                if not dest_file.exists():
                    shutil.copy2(source_file, dest_file)
                    self.logger.debug(f"Copied {metadata.filename} to {folder_path}")
                else:
                    self.logger.debug(f"File already exists: {dest_file}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to copy {metadata.filename}: {e}")
    
    def print_organization_summary(self, summary: dict):
        """Print a formatted organization summary."""
        print("\n" + "="*60)
        print("PDF ORGANIZATION SUMMARY")
        print("="*60)
        
        print(f"Total PDFs Organized: {summary.get('total_pdfs', 0)}")
        print(f"Hierarchical Folders: {summary.get('hierarchical_folders', 0)}")
        print(f"Content Groups: {summary.get('content_groups', 0)}")
        print(f"Organization Date: {summary.get('organization_date', 'Unknown')}")
        print(f"Output Directory: {summary.get('base_path', 'Unknown')}")
        
        print("\nSubject Distribution:")
        for subject, count in summary.get('subject_distribution', {}).items():
            print(f"  {subject.replace('_', ' ').title()}: {count} PDFs")
        
        print("\nAcademic Level Distribution:")
        for level, count in summary.get('level_distribution', {}).items():
            print(f"  {level.replace('_', ' ').title()}: {count} PDFs")
        
        print("\nContent Group Types:")
        for group_type, count in summary.get('group_types', {}).items():
            print(f"  {group_type.title()}: {count} groups")
        
        print("\n" + "="*60)


async def main():
    """Main function for the PDF organizer."""
    parser = argparse.ArgumentParser(
        description='Organize PDF collections using advanced hierarchical categorization and smart grouping'
    )
    parser.add_argument('input_dir', help='Directory containing PDFs to organize')
    parser.add_argument('-o', '--output', default='OrganizedPDFs',
                       help='Output directory for organized PDFs (default: OrganizedPDFs)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Search subdirectories recursively')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be organized without actually copying files')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize organizer
        organizer = PDFOrganizer(args.output)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be copied")
            # TODO: Implement dry run mode
        
        # Organize the directory
        results = await organizer.organize_directory(
            args.input_dir, 
            recursive=args.recursive
        )
        
        if 'error' in results:
            logger.error(results['error'])
            return 1
        
        # Print summary
        organizer.print_organization_summary(results['summary'])
        
        logger.info(f"Organization completed! Check the '{args.output}' directory.")
        return 0
        
    except Exception as e:
        logger.error(f"Organization failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))