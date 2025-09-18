#!/usr/bin/env python3
"""
Test script for the PDF metadata extractor.
Tests the comprehensive metadata extraction functionality.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from processors.metadata_extractor import PDFMetadataExtractor
from data.models import AcademicLevel


def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_metadata_extractor():
    """Test the PDF metadata extractor with sample files."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    extractor = PDFMetadataExtractor()
    
    # Test with any PDF files in the current directory
    pdf_files = list(Path('.').glob('**/*.pdf'))
    
    if not pdf_files:
        logger.info("No PDF files found for testing. Creating a simple test...")
        
        # Test with a non-existent file to check error handling
        test_path = Path('nonexistent.pdf')
        metadata = extractor.extract_comprehensive_metadata(test_path)
        
        print(f"Test with non-existent file:")
        print(f"  Filename: {metadata.filename}")
        print(f"  Is processed: {metadata.is_processed}")
        print(f"  Errors: {metadata.processing_errors}")
        print()
        
        return
    
    # Test with found PDF files
    for pdf_file in pdf_files[:3]:  # Test first 3 PDFs
        logger.info(f"Testing metadata extraction for: {pdf_file}")
        
        try:
            metadata = extractor.extract_comprehensive_metadata(pdf_file)
            
            print(f"Metadata for {pdf_file.name}:")
            print(f"  Title: {metadata.title}")
            print(f"  Authors: {metadata.authors}")
            print(f"  Subject: {metadata.subject_area}")
            print(f"  Academic Level: {metadata.academic_level.value}")
            print(f"  Language: {metadata.language} (confidence: {metadata.language_confidence:.2f})")
            print(f"  Page Count: {metadata.page_count}")
            print(f"  File Size: {metadata.file_size:,} bytes")
            print(f"  Text Ratio: {metadata.text_ratio:.2f}")
            print(f"  Quality Score: {metadata.quality_score:.2f}")
            print(f"  Readability Score: {metadata.readability_score:.2f}")
            print(f"  Content Hash: {metadata.content_hash[:16]}...")
            print(f"  Keywords: {metadata.keywords[:5]}")  # First 5 keywords
            print(f"  Is Processed: {metadata.is_processed}")
            
            if metadata.processing_errors:
                print(f"  Errors: {metadata.processing_errors}")
            
            print("-" * 50)
            
        except Exception as e:
            logger.error(f"Error testing {pdf_file}: {e}")


def test_academic_level_classification():
    """Test academic level classification logic."""
    extractor = PDFMetadataExtractor()
    
    test_cases = [
        ("Introduction to Computer Science - High School Course", "high_school"),
        ("Advanced Algorithms and Data Structures - Graduate Seminar", "graduate"),
        ("Fundamentals of Programming - Undergraduate Course", "undergraduate"),
        ("PhD Dissertation: Machine Learning Research", "postgraduate"),
        ("Basic Mathematics for College Students", "undergraduate")
    ]
    
    print("Testing Academic Level Classification:")
    print("-" * 40)
    
    for title, expected in test_cases:
        level = extractor._classify_academic_level(title, title)
        print(f"Title: {title}")
        print(f"  Classified as: {level.value}")
        print(f"  Expected: {expected}")
        print(f"  Match: {'✓' if level.value == expected else '✗'}")
        print()


if __name__ == "__main__":
    print("Testing PDF Metadata Extractor")
    print("=" * 50)
    
    test_metadata_extractor()
    test_academic_level_classification()
    
    print("Testing completed!")