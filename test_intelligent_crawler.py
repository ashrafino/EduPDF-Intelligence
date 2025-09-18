#!/usr/bin/env python3
"""
Basic test script for IntelligentCrawler functionality.
Tests core crawling logic and academic site recognition.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrapers.intelligent_crawler import IntelligentCrawler
from data.models import SourceConfig, ScrapingStrategy


def test_crawler_initialization():
    """Test crawler initialization and configuration."""
    print("Testing crawler initialization...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://cs.test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            pdf_patterns=[r'\.pdf$', r'lecture.*\.pdf'],
            exclude_patterns=[r'/admin/', r'/private/'],
            max_depth=3
        )
        
        crawler = IntelligentCrawler(config)
        
        assert crawler.source_config.name == "Test University"
        assert len(crawler.academic_patterns) > 0
        assert 'course_pages' in crawler.academic_patterns
        assert 'faculty_directories' in crawler.academic_patterns
        print("✓ Crawler initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error in crawler initialization: {e}")
        return False


def test_academic_pattern_recognition():
    """Test recognition of academic URL patterns."""
    print("\nTesting academic pattern recognition...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML
        )
        
        crawler = IntelligentCrawler(config)
        
        # Test course page recognition
        assert crawler._is_academic_entry_point("https://test.edu/courses/cs101/") == True
        assert crawler._is_academic_entry_point("https://test.edu/classes/math/") == True
        assert crawler._is_academic_entry_point("https://test.edu/curriculum/") == True
        print("✓ Course page patterns recognized correctly")
        
        # Test faculty directory recognition
        assert crawler._is_academic_entry_point("https://test.edu/faculty/") == True
        assert crawler._is_academic_entry_point("https://test.edu/people/professors/") == True
        print("✓ Faculty directory patterns recognized correctly")
        
        # Test resource page recognition
        assert crawler._is_academic_entry_point("https://test.edu/resources/") == True
        assert crawler._is_academic_entry_point("https://test.edu/materials/") == True
        assert crawler._is_academic_entry_point("https://test.edu/library/") == True
        print("✓ Resource page patterns recognized correctly")
        
        # Test non-academic URLs
        assert crawler._is_academic_entry_point("https://test.edu/admin/") == False
        assert crawler._is_academic_entry_point("https://test.edu/contact/") == False
        print("✓ Non-academic URLs correctly excluded")
        
        return True
    except Exception as e:
        print(f"✗ Error in academic pattern recognition: {e}")
        return False


def test_url_validation():
    """Test URL validation and filtering logic."""
    print("\nTesting URL validation...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            pdf_patterns=[r'\.pdf$', r'lecture.*\.pdf'],
            exclude_patterns=[r'/private/', r'/admin/']
        )
        
        crawler = IntelligentCrawler(config)
        
        # Test PDF URL validation
        assert crawler._is_valid_pdf_url("https://test.edu/lecture01.pdf") == True
        assert crawler._is_valid_pdf_url("https://test.edu/notes.pdf") == True
        assert crawler._is_valid_pdf_url("https://test.edu/private/secret.pdf") == False
        assert crawler._is_valid_pdf_url("https://test.edu/image.jpg") == False
        print("✓ PDF URL validation works correctly")
        
        # Test domain checking
        assert crawler._is_same_domain("https://test.edu/page.html") == True
        assert crawler._is_same_domain("https://other.edu/page.html") == False
        print("✓ Domain validation works correctly")
        
        # Test content type checking
        assert crawler._is_crawlable_content_type("https://test.edu/page.html") == True
        assert crawler._is_crawlable_content_type("https://test.edu/document.pdf") == False
        assert crawler._is_crawlable_content_type("https://test.edu/image.jpg") == False
        print("✓ Content type validation works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in URL validation: {e}")
        return False


def test_academic_content_detection():
    """Test detection of academic content in URLs."""
    print("\nTesting academic content detection...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML
        )
        
        crawler = IntelligentCrawler(config)
        
        # Test academic content URLs
        academic_urls = [
            "https://test.edu/courses/cs101/lectures/",
            "https://test.edu/faculty/smith/research/",
            "https://test.edu/departments/math/assignments/",
            "https://test.edu/classes/physics/syllabus.html",
            "https://test.edu/research/publications/"
        ]
        
        for url in academic_urls:
            assert crawler._is_academic_content(url) == True
        print("✓ Academic content URLs detected correctly")
        
        # Test non-academic content URLs
        non_academic_urls = [
            "https://test.edu/about/",
            "https://test.edu/contact/",
            "https://test.edu/news/",
            "https://test.edu/events/",
            "https://test.edu/admissions/"
        ]
        
        for url in non_academic_urls:
            assert crawler._is_academic_content(url) == False
        print("✓ Non-academic content URLs correctly excluded")
        
        return True
    except Exception as e:
        print(f"✗ Error in academic content detection: {e}")
        return False


def test_url_prioritization():
    """Test URL prioritization logic."""
    print("\nTesting URL prioritization...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://cs.test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            subject_areas=["computer_science", "mathematics"]
        )
        
        crawler = IntelligentCrawler(config)
        
        # Test priority calculation
        high_priority_url = "https://cs.test.edu/courses/cs101/"
        medium_priority_url = "https://cs.test.edu/faculty/"
        low_priority_url = "https://cs.test.edu/about/"
        
        high_priority = crawler._calculate_url_priority(high_priority_url)
        medium_priority = crawler._calculate_url_priority(medium_priority_url)
        low_priority = crawler._calculate_url_priority(low_priority_url)
        
        assert high_priority > medium_priority > low_priority
        print("✓ URL prioritization works correctly")
        
        # Test subject area bonus
        cs_url = "https://cs.test.edu/computer-science/courses/"
        math_url = "https://cs.test.edu/mathematics/lectures/"
        generic_url = "https://cs.test.edu/general/info/"
        
        cs_priority = crawler._calculate_url_priority(cs_url)
        math_priority = crawler._calculate_url_priority(math_url)
        generic_priority = crawler._calculate_url_priority(generic_url)
        
        assert cs_priority > generic_priority
        assert math_priority > generic_priority
        print("✓ Subject area prioritization works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in URL prioritization: {e}")
        return False


def test_crawl_statistics():
    """Test crawl statistics tracking."""
    print("\nTesting crawl statistics...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML
        )
        
        crawler = IntelligentCrawler(config)
        
        # Simulate some crawling activity
        crawler.visited_urls.add("https://test.edu/page1")
        crawler.visited_urls.add("https://test.edu/page2")
        crawler.discovered_pdfs.add("https://test.edu/doc1.pdf")
        crawler.crawl_queue.append(("https://test.edu/page3", 1, "https://test.edu"))
        
        stats = crawler.get_crawl_statistics()
        
        assert stats['total_pages_visited'] == 2
        assert stats['total_pdfs_discovered'] == 1
        assert stats['pages_in_queue'] == 1
        assert stats['discovery_rate'] == 0.5
        assert stats['source_name'] == "Test University"
        print("✓ Crawl statistics calculated correctly")
        
        # Test reset functionality
        crawler.reset_crawl_state()
        
        assert len(crawler.visited_urls) == 0
        assert len(crawler.discovered_pdfs) == 0
        assert len(crawler.crawl_queue) == 0
        print("✓ Crawl state reset works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in crawl statistics: {e}")
        return False


def test_general_crawlability():
    """Test general crawlability checks."""
    print("\nTesting general crawlability...")
    
    try:
        config = SourceConfig(
            name="Test University",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML
        )
        
        crawler = IntelligentCrawler(config)
        
        # Test crawlable URLs
        crawlable_urls = [
            "https://test.edu/page.html",
            "https://test.edu/info/",
            "https://test.edu/content/article"
        ]
        
        for url in crawlable_urls:
            assert crawler._is_general_crawlable(url) == True
        print("✓ Crawlable URLs identified correctly")
        
        # Test non-crawlable URLs
        non_crawlable_urls = [
            "https://test.edu/admin/",
            "https://test.edu/login",
            "https://test.edu/api/data",
            "https://test.edu/search?q=test",
            "https://test.edu/cgi-bin/script"
        ]
        
        for url in non_crawlable_urls:
            assert crawler._is_general_crawlable(url) == False
        print("✓ Non-crawlable URLs correctly excluded")
        
        return True
    except Exception as e:
        print(f"✗ Error in general crawlability test: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== Intelligent Crawler Tests ===\n")
    
    tests = [
        test_crawler_initialization,
        test_academic_pattern_recognition,
        test_url_validation,
        test_academic_content_detection,
        test_url_prioritization,
        test_crawl_statistics,
        test_general_crawlability
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