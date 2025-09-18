"""
Comprehensive PDF metadata extraction system.
Implements robust PDF parsing, OCR fallback, and error handling.
"""

import logging
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import io

# PDF processing libraries
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image

# Data models
from data.models import PDFMetadata, AcademicLevel

# Content analysis
from processors.content_analyzer import ContentAnalyzer


class PDFMetadataExtractor:
    """
    Robust PDF metadata extractor with OCR fallback and comprehensive error handling.
    Implements requirements 3.1 and 3.5 for metadata extraction.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configure OCR settings
        self.ocr_config = '--oem 3 --psm 6'
        
        # Initialize content analyzer
        self.content_analyzer = ContentAnalyzer()
        
        # Academic level keywords for classification
        self.academic_keywords = {
            AcademicLevel.HIGH_SCHOOL: [
                'high school', 'secondary', 'grade 9', 'grade 10', 'grade 11', 'grade 12',
                'freshman year', 'sophomore year', 'junior year', 'senior year', 'ap course', 'advanced placement',
                '9th grade', '10th grade', '11th grade', '12th grade'
            ],
            AcademicLevel.UNDERGRADUATE: [
                'undergraduate', 'bachelor', 'college', 'university', 'intro to', 'introduction to',
                'fundamentals', 'principles of', 'basic', 'survey of', 'bsc', 'ba', 'bs',
                'college students', 'university students'
            ],
            AcademicLevel.GRADUATE: [
                'graduate', 'master', 'masters', 'msc', 'ma', 'ms', 'advanced', 'seminar',
                'research methods', 'thesis', 'graduate level', 'graduate course'
            ],
            AcademicLevel.POSTGRADUATE: [
                'phd', 'doctoral', 'postdoc', 'post-doctoral', 'dissertation', 'advanced research',
                'ph.d', 'doctorate', 'postgraduate'
            ]
        }
    
    def extract_comprehensive_metadata(self, pdf_path: Path) -> PDFMetadata:
        """
        Extract comprehensive metadata from a PDF file with error handling.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFMetadata object with extracted information
        """
        self.logger.info(f"Extracting metadata from: {pdf_path}")
        
        # Initialize metadata with basic file information
        metadata = PDFMetadata(
            filename=pdf_path.name,
            file_path=str(pdf_path),
            file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
            title="",  # Will be extracted later
            created_date=datetime.fromtimestamp(pdf_path.stat().st_ctime) if pdf_path.exists() else datetime.now(),
            last_modified=datetime.fromtimestamp(pdf_path.stat().st_mtime) if pdf_path.exists() else datetime.now()
        )
        
        try:
            # Extract metadata using multiple methods
            pypdf2_metadata = self._extract_with_pypdf2(pdf_path)
            pdfplumber_metadata = self._extract_with_pdfplumber(pdf_path)
            
            # Merge metadata from different sources
            self._merge_metadata(metadata, pypdf2_metadata, pdfplumber_metadata)
            
            # Calculate content hash
            metadata.content_hash = self._calculate_content_hash(pdf_path)
            
            # Extract text content for analysis
            text_content = self._extract_text_content(pdf_path)
            
            # Classify academic level based on content
            metadata.academic_level = self._classify_academic_level(text_content, metadata.title)
            
            # Perform comprehensive content analysis
            metadata = self.content_analyzer.analyze_content(text_content, metadata)
            
            # Calculate quality metrics
            self._calculate_quality_metrics(metadata, text_content)
            
            metadata.is_processed = True
            self.logger.info(f"Successfully extracted metadata for: {pdf_path.name}")
            
        except Exception as e:
            error_msg = f"Error extracting metadata from {pdf_path}: {str(e)}"
            self.logger.error(error_msg)
            metadata.processing_errors.append(error_msg)
            metadata.is_processed = False
        
        return metadata
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata using PyPDF2."""
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Basic document info
                if reader.metadata:
                    metadata['title'] = reader.metadata.get('/Title', '').strip()
                    metadata['author'] = reader.metadata.get('/Author', '').strip()
                    metadata['subject'] = reader.metadata.get('/Subject', '').strip()
                    metadata['creator'] = reader.metadata.get('/Creator', '').strip()
                    metadata['producer'] = reader.metadata.get('/Producer', '').strip()
                
                # Page count
                metadata['page_count'] = len(reader.pages)
                
                # Check if encrypted
                metadata['is_encrypted'] = reader.is_encrypted
                
                # Extract text from first few pages for title detection
                text_sample = ""
                for i in range(min(3, len(reader.pages))):
                    try:
                        page_text = reader.pages[i].extract_text()
                        text_sample += page_text + "\n"
                    except Exception as e:
                        self.logger.warning(f"Could not extract text from page {i}: {e}")
                
                metadata['text_sample'] = text_sample
                
        except PyPDF2.errors.PdfReadError as e:
            self.logger.warning(f"PyPDF2 could not read {pdf_path}: {e}")
            metadata['pypdf2_error'] = str(e)
        except Exception as e:
            self.logger.error(f"Unexpected error with PyPDF2 for {pdf_path}: {e}")
            metadata['pypdf2_error'] = str(e)
        
        return metadata
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata using pdfplumber for better text extraction."""
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Basic info
                metadata['page_count'] = len(pdf.pages)
                
                # Extract text from all pages
                full_text = ""
                text_length = 0
                image_count = 0
                
                for page in pdf.pages:
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                            text_length += len(page_text)
                        
                        # Count images for text ratio calculation
                        if hasattr(page, 'images'):
                            image_count += len(page.images)
                            
                    except Exception as e:
                        self.logger.warning(f"Error extracting from page: {e}")
                
                metadata['full_text'] = full_text
                metadata['text_length'] = text_length
                metadata['image_count'] = image_count
                
                # Try to extract title from text if not in metadata
                if full_text and not metadata.get('title'):
                    metadata['extracted_title'] = self._extract_title_from_text(full_text)
                
        except Exception as e:
            self.logger.warning(f"pdfplumber could not process {pdf_path}: {e}")
            metadata['pdfplumber_error'] = str(e)
        
        return metadata
    
    def _extract_text_content(self, pdf_path: Path) -> str:
        """
        Extract text content with OCR fallback for scanned documents.
        """
        text_content = ""
        
        try:
            # First try standard text extraction
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            # If very little text extracted, try OCR
            if len(text_content.strip()) < 100:
                self.logger.info(f"Low text content detected, attempting OCR for {pdf_path}")
                ocr_text = self._extract_with_ocr(pdf_path)
                if len(ocr_text) > len(text_content):
                    text_content = ocr_text
                    
        except Exception as e:
            self.logger.error(f"Error extracting text content from {pdf_path}: {e}")
        
        return text_content
    
    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """
        Extract text using OCR for scanned documents.
        """
        ocr_text = ""
        
        try:
            # Convert PDF pages to images and run OCR
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages[:5]):  # Limit to first 5 pages for performance
                    try:
                        # Convert page to image
                        page_image = page.to_image(resolution=300)
                        
                        # Run OCR on the image
                        page_text = pytesseract.image_to_string(
                            page_image.original, 
                            config=self.ocr_config
                        )
                        
                        ocr_text += page_text + "\n"
                        
                    except Exception as e:
                        self.logger.warning(f"OCR failed for page {i} of {pdf_path}: {e}")
                        
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {pdf_path}: {e}")
        
        return ocr_text
    
    def _merge_metadata(self, metadata: PDFMetadata, pypdf2_data: Dict, pdfplumber_data: Dict):
        """
        Merge metadata from different extraction methods.
        """
        # Title - prefer PyPDF2 metadata, fallback to extracted title
        if pypdf2_data.get('title'):
            metadata.title = pypdf2_data['title']
        elif pdfplumber_data.get('extracted_title'):
            metadata.title = pdfplumber_data['extracted_title']
        else:
            # Use filename as last resort
            metadata.title = metadata.filename.replace('.pdf', '').replace('_', ' ').title()
        
        # Authors - parse from author field
        if pypdf2_data.get('author'):
            metadata.authors = self._parse_authors(pypdf2_data['author'])
        
        # Subject
        if pypdf2_data.get('subject'):
            metadata.subject_area = pypdf2_data['subject']
        
        # Page count - prefer pdfplumber as it's more reliable
        if pdfplumber_data.get('page_count'):
            metadata.page_count = pdfplumber_data['page_count']
        elif pypdf2_data.get('page_count'):
            metadata.page_count = pypdf2_data['page_count']
        
        # Text ratio calculation
        if pdfplumber_data.get('text_length') and pdfplumber_data.get('image_count') is not None:
            text_length = pdfplumber_data['text_length']
            image_count = pdfplumber_data['image_count']
            
            # Simple heuristic: more text = higher ratio
            if image_count == 0:
                metadata.text_ratio = 1.0 if text_length > 1000 else 0.5
            else:
                metadata.text_ratio = min(1.0, text_length / (image_count * 1000))
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """
        Parse author string into list of individual authors.
        """
        if not author_string:
            return []
        
        # Common separators for multiple authors
        separators = [';', ',', ' and ', ' & ', '\n']
        
        authors = [author_string]
        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend([a.strip() for a in author.split(sep) if a.strip()])
            authors = new_authors
        
        # Clean up author names
        cleaned_authors = []
        for author in authors:
            # Remove common prefixes/suffixes
            author = re.sub(r'^(Dr\.?|Prof\.?|Professor)\s+', '', author, flags=re.IGNORECASE)
            author = re.sub(r'\s+(Ph\.?D\.?|M\.?D\.?|Ph\.D\.?)$', '', author, flags=re.IGNORECASE)
            
            if len(author.strip()) > 2:  # Avoid single letters or very short strings
                cleaned_authors.append(author.strip())
        
        return cleaned_authors[:10]  # Limit to 10 authors
    
    def _extract_title_from_text(self, text: str) -> str:
        """
        Extract likely title from document text.
        """
        lines = text.split('\n')
        
        # Look for title in first few lines
        for line in lines[:10]:
            line = line.strip()
            
            # Skip very short lines or lines with mostly numbers/symbols
            if len(line) < 10 or len(re.sub(r'[^a-zA-Z\s]', '', line)) < 5:
                continue
            
            # Skip lines that look like headers/footers
            if any(keyword in line.lower() for keyword in ['page', 'chapter', 'section', 'figure', 'table']):
                continue
            
            # This looks like a potential title
            if len(line) < 200:  # Reasonable title length
                return line
        
        return ""
    
    def _classify_academic_level(self, text_content: str, title: str) -> AcademicLevel:
        """
        Classify academic level based on content and title analysis.
        """
        combined_text = (title + " " + text_content[:2000]).lower()
        
        level_scores = {level: 0 for level in AcademicLevel}
        
        # Score based on keyword matches
        for level, keywords in self.academic_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    level_scores[level] += 1
        
        # Additional heuristics with stronger weights
        if 'dissertation' in combined_text or 'phd' in combined_text:
            level_scores[AcademicLevel.POSTGRADUATE] += 5
        
        if any(word in combined_text for word in ['research', 'methodology', 'analysis']):
            level_scores[AcademicLevel.GRADUATE] += 2
            level_scores[AcademicLevel.POSTGRADUATE] += 1
        
        if any(word in combined_text for word in ['introduction', 'basic', 'fundamentals']):
            level_scores[AcademicLevel.UNDERGRADUATE] += 2
            level_scores[AcademicLevel.HIGH_SCHOOL] += 1
        
        if 'high school' in combined_text:
            level_scores[AcademicLevel.HIGH_SCHOOL] += 5
        
        # Return level with highest score
        best_level = max(level_scores.items(), key=lambda x: x[1])
        
        return best_level[0] if best_level[1] > 0 else AcademicLevel.UNKNOWN
    
    def _calculate_quality_metrics(self, metadata: PDFMetadata, text_content: str):
        """
        Calculate quality metrics for the PDF.
        """
        # Basic quality score based on multiple factors
        quality_score = 0.0
        
        # Page count factor (5-50 pages is ideal)
        if 5 <= metadata.page_count <= 50:
            quality_score += 0.3
        elif metadata.page_count > 50:
            quality_score += 0.2
        elif metadata.page_count >= 3:
            quality_score += 0.1
        
        # Text ratio factor
        quality_score += metadata.text_ratio * 0.3
        
        # Content length factor
        if len(text_content) > 5000:
            quality_score += 0.2
        elif len(text_content) > 1000:
            quality_score += 0.1
        
        # Title quality factor
        if metadata.title and len(metadata.title) > 10:
            quality_score += 0.1
        
        # Author information factor
        if metadata.authors:
            quality_score += 0.1
        
        metadata.quality_score = min(1.0, quality_score)
        
        # Simple readability score (average sentence length)
        sentences = re.split(r'[.!?]+', text_content)
        if sentences:
            words = text_content.split()
            avg_sentence_length = len(words) / len(sentences)
            # Normalize to 0-1 scale (10-20 words per sentence is ideal)
            metadata.readability_score = max(0, min(1.0, 1.0 - abs(avg_sentence_length - 15) / 15))
    
    def _calculate_content_hash(self, pdf_path: Path) -> str:
        """
        Calculate SHA-256 hash of PDF content for deduplication.
        """
        try:
            with open(pdf_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {pdf_path}: {e}")
            return ""