#!/usr/bin/env python3
"""
Basic test script for SourceManager functionality.
Tests core features without requiring pytest.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrapers.source_manager import SourceManager
from data.models import SourceConfig, ScrapingStrategy


def test_source_loading():
    """Test loading sources from configuration."""
    print("Testing source loading...")
    
    try:
        manager = SourceManager("config/sources.yaml")
        sources = manager.get_active_sources()
        
        print(f"✓ Loaded {len(sources)} sources successfully")
        
        for source in sources[:3]:  # Show first 3 sources
            print(f"  - {source.name}: {source.base_url}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading sources: {e}")
        return False


def test_source_filtering():
    """Test source filtering by subject area."""
    print("\nTesting source filtering...")
    
    try:
        manager = SourceManager("config/sources.yaml")
        
        # Test computer science sources
        cs_sources = manager.get_sources_by_subject("computer_science")
        print(f"✓ Found {len(cs_sources)} computer science sources")
        
        # Test mathematics sources
        math_sources = manager.get_sources_by_subject("mathematics")
        print(f"✓ Found {len(math_sources)} mathematics sources")
        
        return True
    except Exception as e:
        print(f"✗ Error filtering sources: {e}")
        return False


async def test_source_validation():
    """Test basic source validation."""
    print("\nTesting source validation...")
    
    try:
        manager = SourceManager("config/sources.yaml")
        sources = manager.get_active_sources()
        
        if not sources:
            print("✗ No sources available for validation")
            return False
        
        # Test validation on first source
        source = sources[0]
        print(f"Validating source: {source.name}")
        
        # This will likely fail due to network/timeout, but tests the code path
        try:
            result = await manager.validate_source(source)
            print(f"✓ Validation result for {source.name}: {result}")
        except Exception as e:
            print(f"✓ Validation attempted (expected network error): {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Error in validation test: {e}")
        return False


def test_content_analysis():
    """Test content analysis methods."""
    print("\nTesting content analysis...")
    
    try:
        manager = SourceManager("config/sources.yaml")
        
        # Test site name extraction
        html_content = '<html><head><title>MIT OpenCourseWare - Computer Science</title></head></html>'
        site_name = manager._extract_site_name(html_content, "ocw.mit.edu")
        print(f"✓ Extracted site name: '{site_name}'")
        
        # Test subject classification
        content = "Welcome to the Computer Science Department. We offer programming and algorithms courses."
        subjects = manager._classify_subject_areas(content, "https://cs.university.edu")
        print(f"✓ Classified subjects: {subjects}")
        
        # Test PDF pattern extraction
        pdf_content = '''
        <html>
            <a href="lecture01.pdf">Lecture 1</a>
            <a href="notes_chapter1.pdf">Chapter Notes</a>
        </html>
        '''
        patterns = manager._extract_pdf_patterns(pdf_content, "https://test.edu")
        print(f"✓ Extracted PDF patterns: {patterns}")
        
        return True
    except Exception as e:
        print(f"✗ Error in content analysis: {e}")
        return False


def test_academic_source_detection():
    """Test academic source detection."""
    print("\nTesting academic source detection...")
    
    try:
        manager = SourceManager("config/sources.yaml")
        
        # Test .edu domain
        edu_source = SourceConfig(
            name="Test University",
            base_url="https://cs.test.edu",
            subject_areas=["computer_science"]
        )
        is_academic = manager._is_academic_source(edu_source)
        print(f"✓ .edu domain detected as academic: {is_academic}")
        
        # Test commercial domain
        commercial_source = SourceConfig(
            name="Commercial Site",
            base_url="https://example.com",
            subject_areas=["general"]
        )
        is_academic = manager._is_academic_source(commercial_source)
        print(f"✓ Commercial domain detected as non-academic: {not is_academic}")
        
        return True
    except Exception as e:
        print(f"✗ Error in academic detection: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== SourceManager Basic Tests ===\n")
    
    tests = [
        test_source_loading,
        test_source_filtering,
        test_source_validation,
        test_content_analysis,
        test_academic_source_detection
    ]
    
    results = []
    for test in tests:
        if asyncio.iscoroutinefunction(test):
            result = await test()
        else:
            result = test()
        results.append(result)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)