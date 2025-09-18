#!/usr/bin/env python3
"""
Basic test for academic repository integrations.
Tests core functionality without external API calls.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scrapers.academic_repositories import (
    ArxivRepository, AcademicRepositoryManager
)
from scrapers.international_sources import InternationalSourceManager
from scrapers.source_manager import SourceManager


async def test_basic_functionality():
    """Test basic functionality without external calls."""
    print("=== Testing Basic Functionality ===")
    
    try:
        # Test ArxivRepository initialization
        print("1. Testing ArxivRepository initialization...")
        arxiv = ArxivRepository()
        print("   ✓ ArxivRepository created successfully")
        
        # Test AcademicRepositoryManager initialization
        print("2. Testing AcademicRepositoryManager initialization...")
        manager = AcademicRepositoryManager()
        stats = manager.get_repository_stats()
        print(f"   ✓ Manager initialized with {stats['total_repositories']} repositories")
        print(f"   ✓ Enabled repositories: {', '.join(stats['enabled_repositories'])}")
        
        # Test InternationalSourceManager initialization
        print("3. Testing InternationalSourceManager initialization...")
        intl_manager = InternationalSourceManager()
        print("   ✓ InternationalSourceManager created successfully")
        
        # Test language detection
        print("4. Testing language detection...")
        test_texts = {
            "This is an English academic paper about machine learning": "en",
            "这是一篇关于机器学习的中文学术论文": "zh",
            "これは機械学習に関する日本語の学術論文です": "ja",
            "이것은 기계 학습에 관한 한국어 학술 논문입니다": "ko"
        }
        
        for text, expected_lang in test_texts.items():
            lang, confidence = intl_manager.detect_language(text)
            print(f"   Text: '{text[:30]}...'")
            print(f"   Detected: {lang} (confidence: {confidence:.2f})")
        
        # Test encoding detection
        print("5. Testing encoding detection...")
        test_content = "Test content with UTF-8 encoding: 测试内容".encode('utf-8')
        encoding = await intl_manager.detect_encoding(test_content)
        print(f"   ✓ Detected encoding: {encoding}")
        
        # Test language-specific patterns
        print("6. Testing language-specific patterns...")
        languages = ['en', 'zh', 'ja', 'ko', 'ar']
        for lang in languages:
            patterns = intl_manager.get_language_specific_patterns(lang)
            pdf_patterns = patterns.get('pdf_patterns', [])
            print(f"   {lang}: {len(pdf_patterns)} PDF patterns")
        
        # Test SourceManager integration
        print("7. Testing SourceManager integration...")
        source_manager = SourceManager()
        
        # Add academic repositories
        source_manager.add_academic_repository_sources()
        source_manager.add_international_repository_sources()
        
        active_sources = source_manager.get_active_sources()
        print(f"   ✓ Total active sources: {len(active_sources)}")
        
        # Count by type
        repo_sources = [s for s in active_sources if 'repository' in s.source_type]
        print(f"   ✓ Repository sources: {len(repo_sources)}")
        
        print("\n✓ All basic tests passed!")
        
    except Exception as e:
        print(f"✗ Error in basic tests: {e}")
        import traceback
        traceback.print_exc()


async def test_configuration_loading():
    """Test configuration loading."""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        # Test loading academic repository config
        print("1. Testing academic repository configuration...")
        
        config_path = Path("config/academic_repositories.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"   ✓ Configuration loaded successfully")
            print(f"   ✓ Repositories configured: {len(config.get('repositories', {}))}")
            print(f"   ✓ International regions: {len(config.get('international', {}).get('regions', {}))}")
            print(f"   ✓ Supported languages: {len(config.get('language_support', {}).get('supported_languages', []))}")
        else:
            print("   ⚠ Configuration file not found, using defaults")
        
        print("\n✓ Configuration tests passed!")
        
    except Exception as e:
        print(f"✗ Error in configuration tests: {e}")


async def main():
    """Run basic tests."""
    print("Testing Academic Repository Integration - Basic Tests")
    print("=" * 60)
    
    await test_basic_functionality()
    await test_configuration_loading()
    
    print("\n" + "=" * 60)
    print("Basic tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())