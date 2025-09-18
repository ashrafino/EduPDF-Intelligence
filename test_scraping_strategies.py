#!/usr/bin/env python3
"""
Basic test script for scraping strategies.
Tests core functionality without requiring external dependencies.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrapers.scraping_strategies import (
    StaticHTMLStrategy, 
    DynamicJavaScriptStrategy, 
    APIEndpointStrategy,
    StrategyFactory
)
from data.models import SourceConfig, ScrapingStrategy


def test_strategy_factory():
    """Test strategy factory creation."""
    print("Testing strategy factory...")
    
    try:
        # Test static HTML strategy creation
        static_config = SourceConfig(
            name="Test Static",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML
        )
        
        static_strategy = StrategyFactory.create_strategy(static_config)
        assert isinstance(static_strategy, StaticHTMLStrategy)
        print("✓ Static HTML strategy created successfully")
        
        # Test dynamic JavaScript strategy creation
        js_config = SourceConfig(
            name="Test JS",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.DYNAMIC_JAVASCRIPT
        )
        
        js_strategy = StrategyFactory.create_strategy(js_config)
        assert isinstance(js_strategy, DynamicJavaScriptStrategy)
        print("✓ Dynamic JavaScript strategy created successfully")
        
        # Test API endpoint strategy creation
        api_config = SourceConfig(
            name="Test API",
            base_url="https://api.test.edu",
            scraping_strategy=ScrapingStrategy.API_ENDPOINT
        )
        
        api_strategy = StrategyFactory.create_strategy(api_config)
        assert isinstance(api_strategy, APIEndpointStrategy)
        print("✓ API endpoint strategy created successfully")
        
        # Test available strategies
        available = StrategyFactory.get_available_strategies()
        assert len(available) >= 3
        print(f"✓ Found {len(available)} available strategies")
        
        return True
    except Exception as e:
        print(f"✗ Error in strategy factory test: {e}")
        return False


def test_static_html_strategy():
    """Test static HTML strategy methods."""
    print("\nTesting static HTML strategy...")
    
    try:
        config = SourceConfig(
            name="Test Static",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            pdf_patterns=[r'\.pdf$', r'lecture.*\.pdf'],
            exclude_patterns=[r'/admin/', r'/private/']
        )
        
        strategy = StaticHTMLStrategy(config)
        
        # Test PDF pattern matching
        assert strategy._is_valid_pdf_url("https://test.edu/lecture01.pdf") == True
        assert strategy._is_valid_pdf_url("https://test.edu/notes.pdf") == True
        assert strategy._is_valid_pdf_url("https://test.edu/admin/secret.pdf") == False
        assert strategy._is_valid_pdf_url("https://test.edu/image.jpg") == False
        print("✓ PDF pattern matching works correctly")
        
        # Test crawlable page detection
        assert strategy._is_crawlable_page("https://test.edu/courses/") == True
        assert strategy._is_crawlable_page("https://test.edu/lecture.pdf") == False
        assert strategy._is_crawlable_page("https://other.edu/page.html") == False
        print("✓ Crawlable page detection works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in static HTML strategy test: {e}")
        return False


def test_api_strategy_methods():
    """Test API strategy helper methods."""
    print("\nTesting API strategy methods...")
    
    try:
        config = SourceConfig(
            name="Test API",
            base_url="https://api.test.edu",
            scraping_strategy=ScrapingStrategy.API_ENDPOINT
        )
        
        strategy = APIEndpointStrategy(config)
        
        # Test JSON PDF extraction
        test_json = {
            "results": [
                {"title": "Paper 1", "pdf_url": "https://test.edu/paper1.pdf"},
                {"title": "Paper 2", "download_url": "https://test.edu/paper2.pdf"}
            ]
        }
        
        pdf_urls = strategy._extract_pdfs_from_json(test_json)
        assert len(pdf_urls) == 2
        assert "https://test.edu/paper1.pdf" in pdf_urls
        assert "https://test.edu/paper2.pdf" in pdf_urls
        print("✓ JSON PDF extraction works correctly")
        
        # Test JSON metadata parsing
        metadata = strategy._parse_json_metadata({
            "title": "Test Paper",
            "author": "Test Author",
            "abstract": "Test description"
        }, "https://api.test.edu/paper/1")
        
        assert metadata['title'] == "Test Paper"
        assert metadata['author'] == "Test Author"
        assert metadata['description'] == "Test description"
        print("✓ JSON metadata parsing works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in API strategy test: {e}")
        return False


async def test_strategy_initialization():
    """Test strategy initialization and cleanup."""
    print("\nTesting strategy initialization...")
    
    try:
        config = SourceConfig(
            name="Test Init",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML,
            timeout=10,
            concurrent_downloads=3
        )
        
        # Test context manager usage
        async with StaticHTMLStrategy(config) as strategy:
            assert strategy.session is not None
            print("✓ Strategy initialized with session")
        
        # Session should be closed after context exit
        print("✓ Strategy cleanup completed")
        
        return True
    except Exception as e:
        print(f"✗ Error in strategy initialization test: {e}")
        return False


def test_content_extraction_methods():
    """Test content extraction helper methods."""
    print("\nTesting content extraction methods...")
    
    try:
        config = SourceConfig(
            name="Test Extract",
            base_url="https://test.edu",
            scraping_strategy=ScrapingStrategy.STATIC_HTML
        )
        
        strategy = StaticHTMLStrategy(config)
        
        # Mock BeautifulSoup object for testing
        class MockSoup:
            def find(self, tag, attrs=None):
                if tag == 'title':
                    return MockTag('MIT OpenCourseWare - Computer Science')
                elif tag == 'meta' and attrs and attrs.get('name') == 'description':
                    return MockTag('', {'content': 'Educational content for students'})
                return None
            
            def find_all(self, tags):
                if 'h1' in tags:
                    return [MockTag('Introduction to Algorithms')]
                return []
            
            def get_text(self):
                return 'Computer Science Department at MIT University offers programming courses.'
        
        class MockTag:
            def __init__(self, text, attrs=None):
                self.text = text
                self.attrs = attrs or {}
            
            def get_text(self):
                return self.text
            
            def get(self, key):
                return self.attrs.get(key)
        
        mock_soup = MockSoup()
        
        # Test title extraction
        title = strategy._extract_title(mock_soup)
        assert title == 'MIT OpenCourseWare - Computer Science'
        print("✓ Title extraction works correctly")
        
        # Test description extraction
        description = strategy._extract_description(mock_soup)
        assert description == 'Educational content for students'
        print("✓ Description extraction works correctly")
        
        # Test keywords extraction
        keywords = strategy._extract_keywords(mock_soup)
        assert 'Introduction to Algorithms' in keywords
        print("✓ Keywords extraction works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error in content extraction test: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== Scraping Strategies Tests ===\n")
    
    tests = [
        test_strategy_factory,
        test_static_html_strategy,
        test_api_strategy_methods,
        test_strategy_initialization,
        test_content_extraction_methods
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