# Academic Repositories Integration

This document describes the implementation of expanded source integration for the educational PDF scraper, including major academic repositories and international source support.

## Overview

The expanded source integration adds support for:

1. **Major Academic Repositories**: arXiv, ResearchGate, institutional repositories, and Google Scholar
2. **International Source Support**: Multi-language content, non-Latin character encodings, and regional academic databases
3. **Cultural Context Awareness**: Language-specific scraping strategies and academic system understanding

## Implementation Details

### 1. Academic Repository Integration (`scrapers/academic_repositories.py`)

#### ArxivRepository
- **API Integration**: Uses arXiv's official API for paper discovery
- **Features**:
  - Search by categories (cs.*, math.*, physics.*, etc.)
  - Recent papers retrieval
  - Comprehensive metadata extraction
  - Direct PDF URL access
- **Rate Limiting**: 3 seconds between requests (respects arXiv guidelines)

#### ResearchGateRepository
- **Web Scraping**: Limited functionality due to anti-bot measures
- **Features**:
  - Basic paper search
  - Author and title extraction
  - PDF link discovery when available
- **Rate Limiting**: 5 seconds between requests

#### InstitutionalRepository
- **Multi-Platform Support**: DSpace, EPrints, and generic repositories
- **Features**:
  - API integration for supported platforms
  - Web scraping fallback for generic repositories
  - Metadata extraction from repository formats
- **Supported Types**: DSpace, EPrints, generic institutional repositories

#### GoogleScholarRepository
- **Limited Use**: Disabled by default due to strict anti-bot measures
- **Features**:
  - Paper search with citation counts
  - Author and publication information
  - PDF link extraction
- **Rate Limiting**: 10 seconds between requests (very conservative)

#### AcademicRepositoryManager
- **Unified Interface**: Coordinates multiple repository integrations
- **Features**:
  - Concurrent searching across repositories
  - PDF URL aggregation
  - Repository statistics and health monitoring
  - Configurable repository selection

### 2. International Source Support (`scrapers/international_sources.py`)

#### Character Encoding Support
- **Auto-Detection**: Uses `chardet` library for encoding detection
- **Fallback Strategy**: UTF-8 with error handling for corrupted content
- **Regional Mappings**: Predefined encoding sets for different regions
- **Supported Encodings**:
  - East Asia: UTF-8, GB2312, GBK, GB18030, Big5, Shift_JIS, EUC-JP, EUC-KR
  - Middle East: UTF-8, ISO-8859-6, Windows-1256
  - Eastern Europe: UTF-8, Windows-1251, KOI8-R, ISO-8859-5
  - Western Europe: UTF-8, ISO-8859-1, Windows-1252

#### Language Detection and Processing
- **Primary Method**: Uses `langdetect` library for accurate detection
- **Fallback Method**: Keyword-based detection using academic terms
- **Supported Languages**: English, Chinese, Japanese, Korean, Arabic, Russian, Spanish, French, German
- **Language-Specific Features**:
  - PDF pattern matching
  - Academic keyword recognition
  - Exclude pattern filtering
  - Cultural context classification

#### Regional Academic Databases
- **China**: CNKI, Wanfang Data
- **Japan**: CiNii, J-STAGE
- **Korea**: KISS, RISS
- **Europe**: HAL (France), SSOAR (Germany)
- **Latin America**: SciELO, Redalyc

#### Cultural Context Awareness
- **Academic System Mapping**: Different grade level classifications
- **Subject Classification**: Culture-specific academic subjects
- **Writing System Support**: Latin, Chinese characters, Arabic script, Cyrillic, etc.
- **Text Direction**: Left-to-right and right-to-left text support

### 3. Enhanced Source Manager Integration

#### New Methods Added to SourceManager
- `initialize_academic_repositories()`: Set up repository integrations
- `initialize_international_sources()`: Set up international support
- `search_academic_repositories()`: Search across all repositories
- `get_arxiv_papers()`: Specific arXiv integration
- `discover_institutional_repositories()`: Auto-discover repositories
- `scrape_international_source()`: Language-aware scraping
- `get_regional_academic_databases()`: Access regional databases
- `expand_source_coverage()`: Add all new source types

#### Source Configuration Enhancements
- Added academic repository sources (arXiv, ResearchGate, Google Scholar)
- Added international repository sources (HAL, CiNii, CNKI, SciELO)
- Language-specific PDF patterns and filtering
- Cultural context metadata

## Configuration

### Academic Repositories Configuration (`config/academic_repositories.yaml`)

```yaml
repositories:
  arxiv:
    enabled: true
    rate_limit: 3.0
    categories: ["cs.*", "math.*", "stat.*", "physics.*"]
  
  researchgate:
    enabled: false  # Limited due to anti-bot measures
    
  institutional:
    - name: "MIT DSpace"
      url: "https://dspace.mit.edu/"
      type: "dspace"
      enabled: true

international:
  regions:
    china:
      enabled: true
      databases:
        - name: "CNKI"
          url: "https://www.cnki.net/"
          language: "zh"
```

### Language Support Configuration

```yaml
language_support:
  supported_languages:
    - code: "zh"
      name: "Chinese"
      priority: 2
    - code: "ja"
      name: "Japanese"
      priority: 3
```

## Usage Examples

### Basic Academic Repository Search

```python
from scrapers.source_manager import SourceManager

async def search_repositories():
    source_manager = SourceManager()
    
    # Search across all academic repositories
    results = await source_manager.search_academic_repositories(
        "machine learning", 
        max_results_per_repo=50
    )
    
    for repo_name, papers in results.items():
        print(f"{repo_name}: {len(papers)} papers found")
```

### arXiv Integration

```python
# Get recent arXiv papers
arxiv_papers = await source_manager.get_arxiv_papers(
    categories=["cs.AI", "cs.LG"], 
    max_results=100
)

# Search specific categories
cs_papers = await source_manager.search_academic_repositories(
    "cat:cs.AI OR cat:cs.LG"
)
```

### International Source Processing

```python
from scrapers.international_sources import InternationalSourceManager

async def process_international_source():
    async with InternationalSourceManager() as intl_manager:
        # Fetch with encoding detection
        content, encoding = await intl_manager.fetch_with_encoding_detection(url)
        
        # Detect language
        language, confidence = intl_manager.detect_language(content)
        
        # Get language-specific patterns
        patterns = intl_manager.get_language_specific_patterns(language)
```

### Institutional Repository Discovery

```python
# Discover institutional repositories
seed_urls = [
    "https://dspace.mit.edu/",
    "https://dash.harvard.edu/",
    "https://purl.stanford.edu/"
]

discovered_repos = await source_manager.discover_institutional_repositories(seed_urls)
print(f"Discovered {len(discovered_repos)} institutional repositories")
```

## Testing

### Test Files
- `test_academic_repositories.py`: Comprehensive integration tests
- `test_basic_integration.py`: Basic functionality verification

### Running Tests

```bash
# Basic functionality test (no external API calls)
python test_basic_integration.py

# Full integration test (requires internet connection)
python test_academic_repositories.py
```

### Test Coverage
- ✅ ArxivRepository API integration
- ✅ ResearchGate web scraping
- ✅ Institutional repository support
- ✅ Google Scholar integration (limited)
- ✅ International source encoding detection
- ✅ Language detection and classification
- ✅ Cultural context awareness
- ✅ Source manager integration

## Performance Considerations

### Rate Limiting
- **arXiv**: 3 seconds between requests
- **ResearchGate**: 5 seconds between requests
- **Google Scholar**: 10 seconds between requests (very conservative)
- **Institutional Repositories**: 2-3 seconds between requests

### Concurrent Processing
- Repository searches run concurrently
- Encoding detection is optimized for performance
- Language detection uses efficient algorithms

### Memory Management
- Streaming content processing for large files
- Automatic cleanup of HTTP sessions
- Efficient text processing for language detection

## Error Handling

### Network Errors
- Exponential backoff for failed requests
- Circuit breaker pattern for problematic sources
- Graceful degradation when repositories are unavailable

### Encoding Errors
- Automatic fallback to UTF-8 with error replacement
- Multiple encoding attempts for difficult content
- Logging of encoding detection confidence

### Language Detection Errors
- Fallback to keyword-based detection
- Default to English for undetectable content
- Confidence scoring for detection quality

## Future Enhancements

### Planned Features
1. **Additional Repositories**: IEEE Xplore, ACM Digital Library, SpringerLink
2. **Enhanced Language Support**: Thai, Hindi, Hebrew, more regional languages
3. **API Integrations**: More repositories with official APIs
4. **Machine Learning**: Improved content classification using ML models
5. **Caching**: Repository response caching for improved performance

### Scalability Improvements
1. **Distributed Processing**: Support for multiple worker nodes
2. **Database Optimization**: Better indexing for international content
3. **Content Deduplication**: Cross-repository duplicate detection
4. **Quality Scoring**: Enhanced quality metrics for international content

## Dependencies

### New Dependencies Added
- `chardet>=5.0.0`: Character encoding detection
- `charset-normalizer>=3.0.0`: Character encoding normalization
- `langdetect>=1.0.9`: Language detection (already present)

### Optional Dependencies
- `playwright`: For JavaScript-heavy sites (ResearchGate, Google Scholar)
- `selenium`: Alternative browser automation

## Compliance and Ethics

### Rate Limiting Compliance
- All integrations respect robots.txt and rate limits
- Conservative request timing to avoid overloading servers
- User-Agent headers identify the scraper appropriately

### Content Licensing
- Only accesses publicly available content
- Respects copyright and fair use guidelines
- Focuses on academic and educational materials

### Privacy Considerations
- No personal data collection
- Anonymized usage patterns
- Secure handling of API keys and credentials

## Troubleshooting

### Common Issues

1. **Encoding Detection Failures**
   - Solution: Falls back to UTF-8 with error handling
   - Check: Content-Type headers for declared encoding

2. **Language Detection Low Confidence**
   - Solution: Uses keyword-based fallback detection
   - Check: Content length and language-specific terms

3. **Repository API Failures**
   - Solution: Implements retry logic and circuit breakers
   - Check: Network connectivity and API status

4. **Rate Limiting Blocks**
   - Solution: Increases delay between requests
   - Check: Repository-specific rate limit policies

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This provides detailed information about:
- HTTP requests and responses
- Encoding detection results
- Language detection confidence
- Repository search results
- Error conditions and recovery attempts