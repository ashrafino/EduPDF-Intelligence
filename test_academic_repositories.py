#!/usr/bin/env python3
"""
Test script for academic repository integrations.
Tests arXiv API, ResearchGate scraping, institutional repositories, and international sources.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scrapers.academic_repositories import (
    ArxivRepository, ResearchGateRepository, InstitutionalRepository,
    GoogleScholarRepository, AcademicRepositoryManager
)
from scrapers.international_sources import InternationalSourceManager
from scrapers.source_manager import SourceManager


async def test_arxiv_integration():
    """Test arXiv API integration."""
    print("\n=== Testing arXiv Integration ===")
    
    try:
        async with ArxivRepository() as arxiv:
            # Test search
            print("Searching arXiv for computer science papers...")
            papers = await arxiv.search_papers("cat:cs.AI", max_results=5)
            
            print(f"Found {len(papers)} papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"{i}. {paper.get('title', 'No title')}")
                print(f"   Authors: {', '.join(paper.get('authors', []))}")
                print(f"   arXiv ID: {paper.get('arxiv_id', 'N/A')}")
                
                # Test PDF URL extraction
                pdf_urls = await arxiv.get_pdf_urls(paper)
                if pdf_urls:
                    print(f"   PDF URL: {pdf_urls[0]}")
                print()
            
            # Test category search
            print("Testing category search...")
            cs_papers = await arxiv.search_by_category("cs.LG", max_results=3)
            print(f"Found {len(cs_papers)} machine learning papers")
            
            # Test recent papers
            print("Testing recent papers...")
            recent = await arxiv.get_recent_papers(days=7, categories=["cs.*"])
            print(f"Found {len(recent)} recent CS papers")
            
    except Exception as e:
        print(f"Error testing arXiv: {e}")


async def test_researchgate_integration():
    """Test ResearchGate integration."""
    print("\n=== Testing ResearchGate Integration ===")
    
    try:
        async with ResearchGateRepository() as rg:
            print("Searching ResearchGate for machine learning papers...")
            papers = await rg.search_papers("machine learning", max_results=3)
            
            print(f"Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.get('title', 'No title')}")
                print(f"   Authors: {', '.join(paper.get('authors', []))}")
                
                pdf_urls = await rg.get_pdf_urls(paper)
                if pdf_urls:
                    print(f"   PDF URL: {pdf_urls[0]}")
                print()
                
    except Exception as e:
        print(f"Error testing ResearchGate: {e}")


async def test_institutional_repository():
    """Test institutional repository integration."""
    print("\n=== Testing Institutional Repository ===")
    
    try:
        # Test with MIT DSpace configuration
        config = {
            'base_url': 'https://dspace.mit.edu',
            'type': 'dspace'
        }
        
        async with InstitutionalRepository(config) as repo:
            print("Searching MIT DSpace...")
            papers = await repo.search_papers("computer science", max_results=3)
            
            print(f"Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.get('title', 'No title')}")
                print(f"   Authors: {', '.join(paper.get('authors', []))}")
                
                pdf_urls = await repo.get_pdf_urls(paper)
                if pdf_urls:
                    print(f"   PDF URL: {pdf_urls[0]}")
                print()
                
    except Exception as e:
        print(f"Error testing institutional repository: {e}")


async def test_google_scholar():
    """Test Google Scholar integration (use with caution)."""
    print("\n=== Testing Google Scholar Integration ===")
    
    try:
        async with GoogleScholarRepository() as scholar:
            print("Searching Google Scholar (limited test)...")
            papers = await scholar.search_papers("machine learning PDF", max_results=2)
            
            print(f"Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.get('title', 'No title')}")
                print(f"   Authors: {', '.join(paper.get('authors', []))}")
                print(f"   Citations: {paper.get('citation_count', 'N/A')}")
                
                pdf_urls = await scholar.get_pdf_urls(paper)
                if pdf_urls:
                    print(f"   PDF URL: {pdf_urls[0]}")
                print()
                
    except Exception as e:
        print(f"Error testing Google Scholar: {e}")


async def test_repository_manager():
    """Test the academic repository manager."""
    print("\n=== Testing Academic Repository Manager ===")
    
    try:
        config = {
            'repositories': {
                'arxiv': {'enabled': True},
                'researchgate': {'enabled': False},  # Disabled for testing
                'google_scholar': {'enabled': False}  # Disabled for testing
            }
        }
        
        manager = AcademicRepositoryManager(config)
        
        print("Searching all enabled repositories...")
        results = await manager.search_all_repositories("machine learning", max_results_per_repo=3)
        
        for repo_name, papers in results.items():
            print(f"\n{repo_name}: {len(papers)} papers")
            for i, paper in enumerate(papers[:2], 1):
                print(f"  {i}. {paper.get('title', 'No title')[:60]}...")
        
        # Test PDF URL extraction
        print("\nExtracting PDF URLs...")
        pdf_urls = await manager.get_all_pdf_urls(results)
        print(f"Found {len(pdf_urls)} PDF URLs")
        
        # Show repository stats
        stats = manager.get_repository_stats()
        print(f"\nRepository Stats:")
        print(f"  Total repositories: {stats['total_repositories']}")
        print(f"  Enabled: {', '.join(stats['enabled_repositories'])}")
        
    except Exception as e:
        print(f"Error testing repository manager: {e}")


async def test_international_sources():
    """Test international source support."""
    print("\n=== Testing International Sources ===")
    
    try:
        async with InternationalSourceManager() as intl_manager:
            # Test encoding detection
            print("Testing encoding detection...")
            test_content = "这是中文测试内容".encode('utf-8')
            encoding = await intl_manager.detect_encoding(test_content)
            print(f"Detected encoding: {encoding}")
            
            # Test language detection
            print("Testing language detection...")
            chinese_text = "这是一篇关于机器学习的学术论文"
            lang, confidence = intl_manager.detect_language(chinese_text)
            print(f"Detected language: {lang} (confidence: {confidence:.2f})")
            
            # Test language-specific patterns
            print("Testing language-specific patterns...")
            zh_patterns = intl_manager.get_language_specific_patterns('zh')
            print(f"Chinese PDF patterns: {zh_patterns.get('pdf_patterns', [])}")
            
            # Test regional databases
            print("Testing regional databases...")
            china_dbs = await intl_manager.get_regional_databases('china')
            print(f"Found {len(china_dbs)} Chinese databases:")
            for db in china_dbs:
                print(f"  - {db.get('name', 'Unknown')}")
            
    except Exception as e:
        print(f"Error testing international sources: {e}")


async def test_source_manager_integration():
    """Test integration with the main source manager."""
    print("\n=== Testing Source Manager Integration ===")
    
    try:
        source_manager = SourceManager()
        
        # Test expanding source coverage
        print("Expanding source coverage...")
        stats = await source_manager.expand_source_coverage()
        print(f"Source expansion stats: {stats}")
        
        # Test academic repository search
        print("Testing academic repository search...")
        results = await source_manager.search_academic_repositories("computer science", max_results_per_repo=2)
        
        for repo_name, papers in results.items():
            print(f"{repo_name}: {len(papers)} papers")
        
        # Test arXiv integration
        print("Testing arXiv integration...")
        arxiv_papers = await source_manager.get_arxiv_papers(categories=["cs.AI"], max_results=3)
        print(f"Found {len(arxiv_papers)} arXiv AI papers")
        
        # Test regional databases
        print("Testing regional databases...")
        china_dbs = await source_manager.get_regional_academic_databases('china')
        print(f"Found {len(china_dbs)} Chinese academic databases")
        
    except Exception as e:
        print(f"Error testing source manager integration: {e}")


async def main():
    """Run all tests."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Academic Repository Integrations")
    print("=" * 50)
    
    # Run tests
    await test_arxiv_integration()
    await test_researchgate_integration()
    await test_institutional_repository()
    # await test_google_scholar()  # Commented out to avoid rate limiting
    await test_repository_manager()
    await test_international_sources()
    await test_source_manager_integration()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())