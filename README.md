# Enhanced Educational PDF Scraper

A comprehensive, modular system for collecting, analyzing, and organizing high-quality educational PDFs from diverse academic sources. This project implements intelligent content filtering, academic level classification, and multilingual support for educational content discovery.

## 🚀 Features Implemented

### ✅ Core Architecture

- **Modular Design**: Clean separation of concerns with dedicated modules for scraping, processing, filtering, and data management
- **Configuration Management**: Flexible JSON/YAML-based configuration system with runtime parameter control
- **Database Integration**: SQLite-based metadata storage with comprehensive indexing and migration support
- **Logging System**: Structured logging with rotation and configurable levels

### ✅ Content Processing Pipeline

- **PDF Metadata Extraction**: Comprehensive extraction of titles, authors, institutions, and technical metadata
- **Content Analysis**: Advanced text analysis using TF-IDF, keyword extraction, and topic classification
- **Language Detection**: Multi-language support with confidence scoring using langdetect
- **Academic Level Classification**: Intelligent classification into high school, undergraduate, and graduate levels
- **Quality Assessment**: Readability analysis and educational relevance scoring

### ✅ Intelligent Filtering System

- **Educational Relevance Filter**: Identifies educational content using keyword analysis and ML models
- **Academic Classifier**: Determines academic level using readability metrics and course code analysis
- **Content Quality Scoring**: Multi-factor quality assessment including text ratio and complexity analysis
- **Deduplication**: Content hashing for duplicate detection and removal

### ✅ Data Management

- **Comprehensive Data Models**: Rich metadata structures supporting all educational content types
- **Database Schema**: Optimized SQLite schema with proper indexing and relationships
- **Migration System**: Database versioning and upgrade capabilities
- **Statistics Tracking**: Collection progress monitoring and quality metrics

### ✅ Source Management

- **Multiple Scraping Strategies**: Support for static HTML, dynamic JavaScript, and API endpoints
- **Rate Limiting**: Configurable request throttling to respect source policies
- **Source Configuration**: YAML-based source definitions with flexible parameters
- **Academic Repository Support**: Pre-configured support for major educational institutions

## 📁 Project Structure

```
├── config/                    # Configuration management
│   ├── settings.py           # Configuration classes and manager ✅
│   ├── app_config.json       # Application settings ✅
│   ├── sources.yaml          # Source configurations ✅
│   └── academic_repositories.yaml # Academic source definitions ✅
├── data/                     # Database and models
│   ├── models.py            # Data models and enums ✅
│   ├── database.py          # Database manager and operations ✅
│   ├── init_db.py           # Database initialization ✅
│   └── scraper.db           # SQLite database (created at runtime)
├── processors/              # Content processing
│   ├── content_analyzer.py  # Text analysis and classification ✅
│   ├── metadata_extractor.py # PDF metadata extraction ✅
│   ├── pdf_processor.py     # PDF processing functions ✅
│   ├── download_manager.py  # Download management ✅
│   └── worker_pool.py       # Parallel processing ✅
├── filters/                 # Content filtering
│   ├── intelligent_filter.py # Main filtering system ✅
│   ├── academic_classifier.py # Academic level classification ✅
│   └── content_filter.py    # Educational relevance filtering ✅
├── utils/                   # Utilities
│   └── logging_setup.py     # Logging configuration ✅
├── tests/                   # Test suite
│   ├── test_*.py           # Comprehensive test coverage ✅
├── main.py                  # Main application entry point ✅
├── scrapper.py             # Legacy scraper (functional) ✅
└── requirements.txt        # Dependencies ✅
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Quick Start

1. **Clone and setup**:

```bash
git clone <repository-url>
cd educational-pdf-scraper
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Initialize database**:

```bash
python -m data.init_db
```

4. **Run the scraper**:

```bash
python main.py
```

### Alternative: Use Legacy Scraper

For immediate functionality, you can use the original scraper:

```bash
python scrapper.py
```

## 📋 Dependencies

### Core Libraries

- **requests** (≥2.31.0): HTTP client for web scraping
- **aiohttp** (≥3.8.0): Async HTTP client for concurrent downloads
- **beautifulsoup4** (≥4.12.0): HTML parsing and content extraction
- **PyYAML** (≥6.0): Configuration file parsing

### PDF Processing

- **PyPDF2** (≥3.0.0): PDF metadata and text extraction
- **pdfplumber** (≥0.9.0): Advanced PDF content analysis
- **pytesseract** (≥0.3.10): OCR for scanned documents

### Text Analysis & ML

- **scikit-learn** (≥1.3.0): Machine learning for classification
- **nltk** (≥3.8.1): Natural language processing
- **langdetect** (≥1.0.9): Language detection
- **numpy** (≥1.24.0): Numerical computations

## ⚙️ Configuration

### Application Settings (`config/app_config.json`)

```json
{
  "base_output_dir": "EducationalPDFs",
  "processing": {
    "max_workers": 4,
    "max_concurrent_downloads": 10,
    "download_timeout": 30,
    "max_file_size_mb": 100
  },
  "filtering": {
    "min_quality_score": 0.6,
    "supported_languages": ["en", "fr", "ar", "es", "de"]
  }
}
```

### Source Configuration (`config/sources.yaml`)

```yaml
mit_ocw_cs:
  name: "MIT OCW Computer Science"
  base_url: "https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/"
  scraping_strategy: "static_html"
  rate_limit: 2.0
  subject_areas: ["computer_science"]
  languages: ["en"]
```

## 🔧 Usage Examples

### Basic Usage

```python
from config.settings import config_manager
from data.database import DatabaseManager
from processors.content_analyzer import ContentAnalyzer

# Load configuration
config = config_manager.load_app_config()
sources = config_manager.load_sources()

# Initialize database
db = DatabaseManager()

# Analyze content
analyzer = ContentAnalyzer()
```

### Running Tests

```bash
# Run specific test modules
python test_content_analyzer.py
python test_intelligent_filter.py
python test_basic_integration.py

# Run academic repository tests
python test_academic_repositories.py
```

## 🧪 Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Content Analysis Tests**: Text processing and classification
- **Filter Tests**: Educational relevance and quality assessment
- **Database Tests**: Data persistence and retrieval

## 🎯 Key Features in Detail

### Intelligent Content Analysis

- **Keyword Extraction**: TF-IDF based keyword identification
- **Subject Classification**: Multi-class classification for academic subjects
- **Language Detection**: Confidence-scored language identification
- **Readability Analysis**: Flesch-Kincaid and ARI scoring

### Academic Level Classification

- **Readability Metrics**: Multiple algorithms for complexity assessment
- **Course Code Analysis**: Pattern recognition for academic level indicators
- **Keyword-based Classification**: Subject-specific vocabulary analysis
- **ML Classification**: Trained models for level prediction

### Quality Assessment

- **Educational Relevance**: Scoring based on educational keywords and patterns
- **Content Quality**: Text-to-image ratio and structural analysis
- **Duplicate Detection**: Content hashing for similarity identification
- **Source Credibility**: Institution and source type weighting

## 🌍 Multilingual Support

The system supports multiple languages with automatic detection:

- **English** (primary)
- **French** (français)
- **Arabic** (العربية)
- **Spanish** (español)
- **German** (deutsch)

## 📊 Supported Academic Sources

Pre-configured support for major educational institutions:

- MIT OpenCourseWare
- Stanford Computer Science
- UC Berkeley EECS
- Carnegie Mellon SCS
- arXiv.org
- INRIA HAL

## 🔄 Processing Pipeline

1. **Source Discovery**: Identify and validate educational sources
2. **Content Extraction**: Download and parse PDF documents
3. **Metadata Analysis**: Extract comprehensive document metadata
4. **Content Classification**: Determine subject area and academic level
5. **Quality Assessment**: Score educational relevance and content quality
6. **Deduplication**: Identify and handle duplicate content
7. **Storage**: Persist metadata and organize files

## 📈 Performance Features

- **Async Processing**: Concurrent downloads and processing
- **Rate Limiting**: Respectful scraping with configurable delays
- **Caching**: Intelligent caching to avoid redundant processing
- **Progress Tracking**: Real-time monitoring of collection progress
- **Error Handling**: Robust error recovery and retry mechanisms

## 🚧 Current Status

This project represents a significant enhancement over the original `scrapper.py` with:

- ✅ Complete modular architecture implementation
- ✅ Advanced content analysis and classification
- ✅ Comprehensive database schema and management
- ✅ Intelligent filtering and quality assessment
- ✅ Extensive test coverage and validation
- ✅ Production-ready configuration management

The system is ready for deployment and can be extended with additional sources, processing strategies, and analysis capabilities.
