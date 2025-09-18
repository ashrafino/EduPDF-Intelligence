"""
Tests for the SourceManager class.
Validates source loading, discovery, and health checking functionality.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from scrapers.source_manager import SourceManager
from data.models import SourceConfig, ScrapingStrategy


class TestSourceManager:
    """Test cases for SourceManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_sources.yaml"
        
        # Sample configuration data
        self.test_config = {
            'test_source': {
                'name': 'Test University',
                'base_url': 'https://test.edu',
                'scraping_strategy': 'static_html',
                'rate_limit': 1.0,
                'max_depth': 2,
                'pdf_patterns': [r'\.pdf$'],
                'subject_areas': ['computer_science'],
                'languages': ['en'],
                'is_active': True,
                'institution': 'Test University'
            }
        }
        
        # Write test config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def test_load_sources_from_yaml_config(self):
        """Test loading sources from YAML configuration."""
        manager = SourceManager(str(self.config_path))
        
        assert len(manager.sources) == 1
        assert 'test_source' in manager.sources
        
        source = manager.sources['test_source']
        assert source.name == 'Test University'
        assert source.base_url == 'https://test.edu'
        assert source.scraping_strategy == ScrapingStrategy.STATIC_HTML
        assert source.is_active is True
    
    def test_get_active_sources(self):
        """Test retrieving only active sources."""
        manager = SourceManager(str(self.config_path))
        active_sources = manager.get_active_sources()
        
        assert len(active_sources) == 1
        assert active_sources[0].name == 'Test University'
    
    def test_get_sources_by_subject(self):
        """Test filtering sources by subject area."""
        manager = SourceManager(str(self.config_path))
        cs_sources = manager.get_sources_by_subject('computer_science')
        
        assert len(cs_sources) == 1
        assert cs_sources[0].name == 'Test University'
        
        # Test non-existent subject
        math_sources = manager.get_sources_by_subject('mathematics')
        assert len(math_sources) == 0
    
    @pytest.mark.asyncio
    async def test_validate_source_success(self):
        """Test successful source validation."""
        manager = SourceManager(str(self.config_path))
        source = manager.sources['test_source']
        
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='<html><a href="test.pdf">PDF</a></html>')
        mock_response.headers = {}
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await manager.validate_source(source)
            assert result is True
            assert 'Test University' in manager.health_status
            assert manager.health_status['Test University']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_validate_source_failure(self):
        """Test source validation failure."""
        manager = SourceManager(str(self.config_path))
        source = manager.sources['test_source']
        
        # Mock failed HTTP response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await manager.validate_source(source)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_source_health(self):
        """Test comprehensive source health checking."""
        manager = SourceManager(str(self.config_path))
        
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='<html><a href="test.pdf">PDF</a></html>')
        mock_response.headers = {
            'content-type': 'text/html',
            'server': 'nginx'
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            health_info = await manager.check_source_health('test_source')
            
            assert health_info['status'] == 'healthy'
            assert health_info['response_code'] == 200
            assert 'response_time' in health_info
            assert health_info['pdf_links_found'] == 1
    
    def test_get_source_health_summary(self):
        """Test health summary generation."""
        manager = SourceManager(str(self.config_path))
        
        # Add some mock health data
        manager.health_status = {
            'source1': {'status': 'healthy'},
            'source2': {'status': 'unhealthy'},
            'source3': {'status': 'healthy'}
        }
        
        summary = manager.get_source_health_summary()
        
        assert summary['total_sources'] == 1  # Only one source in config
        assert summary['healthy_sources'] == 0  # No health checks run yet
        assert 'health_percentage' in summary
    
    def test_extract_site_name(self):
        """Test site name extraction from HTML content."""
        manager = SourceManager(str(self.config_path))
        
        # Test with title tag
        html_with_title = '<html><head><title>MIT OpenCourseWare</title></head></html>'
        name = manager._extract_site_name(html_with_title, 'ocw.mit.edu')
        assert name == 'MIT OpenCourseWare'
        
        # Test fallback to domain
        html_no_title = '<html><head></head></html>'
        name = manager._extract_site_name(html_no_title, 'test.edu')
        assert name == 'Test'
    
    def test_classify_subject_areas(self):
        """Test subject area classification from content."""
        manager = SourceManager(str(self.config_path))
        
        # Test computer science content
        cs_content = 'Welcome to Computer Science Department. We offer programming courses.'
        subjects = manager._classify_subject_areas(cs_content, 'https://cs.test.edu')
        assert 'computer_science' in subjects
        
        # Test mathematics content
        math_content = 'Mathematics Department offers calculus and algebra courses.'
        subjects = manager._classify_subject_areas(math_content, 'https://math.test.edu')
        assert 'mathematics' in subjects
    
    def test_extract_pdf_patterns(self):
        """Test PDF pattern extraction from content."""
        manager = SourceManager(str(self.config_path))
        
        content = '''
        <html>
            <a href="lecture01.pdf">Lecture 1</a>
            <a href="notes_chapter1.pdf">Notes</a>
            <a href="slides_intro.pdf">Slides</a>
        </html>
        '''
        
        patterns = manager._extract_pdf_patterns(content, 'https://test.edu')
        
        assert r'\.pdf$' in patterns
        assert r'lecture.*\.pdf' in patterns
        assert r'notes.*\.pdf' in patterns
        assert r'slides.*\.pdf' in patterns
    
    def test_is_academic_source(self):
        """Test academic source identification."""
        manager = SourceManager(str(self.config_path))
        
        # Test .edu domain
        edu_source = SourceConfig(
            name="Test University",
            base_url="https://test.edu",
            subject_areas=["computer_science"]
        )
        assert manager._is_academic_source(edu_source) is True
        
        # Test non-academic domain
        commercial_source = SourceConfig(
            name="Commercial Site",
            base_url="https://example.com",
            subject_areas=["general"]
        )
        assert manager._is_academic_source(commercial_source) is False
        
        # Test academic repository
        arxiv_source = SourceConfig(
            name="arXiv",
            base_url="https://arxiv.org",
            subject_areas=["computer_science"]
        )
        assert manager._is_academic_source(arxiv_source) is True


if __name__ == '__main__':
    pytest.main([__file__])