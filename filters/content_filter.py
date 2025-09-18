"""
Educational relevance filter for PDF content analysis.
Implements content analysis, institution credibility scoring, and quality filtering.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data.models import PDFMetadata, AcademicLevel


@dataclass
class FilterResult:
    """Result of content filtering analysis."""
    is_educational: bool
    relevance_score: float
    quality_score: float
    reasons: List[str]
    institution_score: float
    author_credibility: float


class EducationalRelevanceFilter:
    """
    Filter for determining educational relevance of PDF content.
    Implements keyword matching, institution scoring, and quality analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Educational keywords by category
        self.educational_keywords = {
            'academic_terms': [
                'course', 'syllabus', 'curriculum', 'lecture', 'tutorial', 'assignment',
                'homework', 'exam', 'quiz', 'test', 'study', 'textbook', 'chapter',
                'lesson', 'module', 'unit', 'semester', 'academic', 'education',
                'learning', 'teaching', 'instruction', 'pedagogy', 'student',
                'professor', 'instructor', 'faculty', 'university', 'college',
                'school', 'department', 'research', 'thesis', 'dissertation'
            ],
            'subject_indicators': [
                'mathematics', 'physics', 'chemistry', 'biology', 'computer science',
                'engineering', 'history', 'literature', 'psychology', 'sociology',
                'economics', 'philosophy', 'political science', 'geography',
                'anthropology', 'linguistics', 'statistics', 'calculus', 'algebra',
                'geometry', 'programming', 'algorithms', 'data structures'
            ],
            'document_types': [
                'notes', 'slides', 'presentation', 'handout', 'worksheet',
                'lab manual', 'guide', 'reference', 'handbook', 'primer',
                'introduction', 'overview', 'summary', 'review', 'analysis'
            ]
        }
        
        # Credible educational institutions (partial list)
        self.credible_institutions = {
            'tier_1': [  # Top-tier universities
                'harvard', 'mit', 'stanford', 'cambridge', 'oxford', 'caltech',
                'princeton', 'yale', 'columbia', 'chicago', 'berkeley', 'ucla'
            ],
            'tier_2': [  # Well-known universities
                'university', 'college', 'institute', 'school', 'academy',
                'polytechnic', 'technical', 'state university', 'community college'
            ],
            'publishers': [  # Academic publishers
                'springer', 'elsevier', 'wiley', 'pearson', 'mcgraw-hill',
                'cambridge university press', 'oxford university press',
                'mit press', 'academic press'
            ]
        }
        
        # Course code patterns
        self.course_patterns = [
            r'\b[A-Z]{2,4}\s*\d{3,4}\b',  # CS101, MATH 2301, etc.
            r'\b[A-Z]{2,4}-\d{3,4}\b',    # CS-101, MATH-2301
            r'\bCourse\s+\d+\b',          # Course 101
            r'\bLecture\s+\d+\b',         # Lecture 1
            r'\bChapter\s+\d+\b'          # Chapter 1
        ]
        
        # Quality thresholds
        self.min_file_size = 1024  # 1KB minimum
        self.max_file_size = 100 * 1024 * 1024  # 100MB maximum
        self.min_pages = 5
        self.min_text_ratio = 0.1  # At least 10% text content
        
    def analyze_educational_relevance(self, metadata: PDFMetadata, 
                                    content_text: Optional[str] = None) -> FilterResult:
        """
        Analyze PDF for educational relevance using multiple criteria.
        
        Args:
            metadata: PDF metadata object
            content_text: Optional extracted text content
            
        Returns:
            FilterResult with relevance analysis
        """
        reasons = []
        scores = {}
        
        # 1. Keyword analysis
        keyword_score = self._analyze_keywords(metadata, content_text)
        scores['keywords'] = keyword_score
        if keyword_score > 0.3:
            reasons.append(f"Contains educational keywords (score: {keyword_score:.2f})")
        
        # 2. Institution credibility
        institution_score = self._score_institution_credibility(metadata)
        scores['institution'] = institution_score
        if institution_score > 0.5:
            reasons.append(f"From credible institution (score: {institution_score:.2f})")
        
        # 3. Author credibility
        author_score = self._score_author_credibility(metadata)
        scores['author'] = author_score
        if author_score > 0.3:
            reasons.append(f"Credible author indicators (score: {author_score:.2f})")
        
        # 4. File quality validation
        quality_score = self._validate_file_quality(metadata)
        scores['quality'] = quality_score
        if quality_score < 0.5:
            reasons.append(f"Quality issues detected (score: {quality_score:.2f})")
        
        # 5. Course/academic structure detection
        structure_score = self._detect_academic_structure(metadata, content_text)
        scores['structure'] = structure_score
        if structure_score > 0.4:
            reasons.append(f"Academic structure detected (score: {structure_score:.2f})")
        
        # Calculate overall relevance score
        relevance_score = (
            keyword_score * 0.3 +
            institution_score * 0.25 +
            author_score * 0.15 +
            structure_score * 0.3
        )
        
        # Determine if educational (threshold: 0.4)
        is_educational = relevance_score >= 0.4 and quality_score >= 0.5
        
        if not is_educational and relevance_score < 0.4:
            reasons.append(f"Low educational relevance (score: {relevance_score:.2f})")
        
        return FilterResult(
            is_educational=is_educational,
            relevance_score=relevance_score,
            quality_score=quality_score,
            reasons=reasons,
            institution_score=institution_score,
            author_credibility=author_score
        )
    
    def _analyze_keywords(self, metadata: PDFMetadata, 
                         content_text: Optional[str] = None) -> float:
        """Analyze educational keywords in title, description, and content."""
        text_sources = []
        
        # Collect text from various sources
        if metadata.title:
            text_sources.append(metadata.title.lower())
        if metadata.description:
            text_sources.append(metadata.description.lower())
        if metadata.keywords:
            text_sources.append(' '.join(metadata.keywords).lower())
        if content_text:
            # Use first 1000 characters to avoid processing huge documents
            text_sources.append(content_text[:1000].lower())
        
        if not text_sources:
            return 0.0
        
        combined_text = ' '.join(text_sources)
        
        # Count matches in each category
        total_matches = 0
        total_keywords = 0
        
        for category, keywords in self.educational_keywords.items():
            category_matches = 0
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    category_matches += 1
                total_keywords += 1
            
            # Weight different categories
            if category == 'academic_terms':
                total_matches += category_matches * 1.5
            elif category == 'subject_indicators':
                total_matches += category_matches * 1.2
            else:
                total_matches += category_matches
        
        # Normalize score (0-1)
        if total_keywords == 0:
            return 0.0
        
        score = min(total_matches / (total_keywords * 0.1), 1.0)
        return score
    
    def _score_institution_credibility(self, metadata: PDFMetadata) -> float:
        """Score institution credibility based on known educational institutions."""
        if not metadata.institution and not metadata.download_source:
            return 0.0
        
        # Combine institution and source URL for analysis
        text_to_analyze = []
        if metadata.institution:
            text_to_analyze.append(metadata.institution.lower())
        if metadata.download_source:
            text_to_analyze.append(metadata.download_source.lower())
        
        combined_text = ' '.join(text_to_analyze)
        
        # Check against credible institution lists
        for tier, institutions in self.credible_institutions.items():
            for institution in institutions:
                if institution.lower() in combined_text:
                    if tier == 'tier_1':
                        return 1.0
                    elif tier == 'tier_2':
                        return 0.7
                    elif tier == 'publishers':
                        return 0.8
        
        # Check for educational domain indicators
        edu_indicators = ['.edu', '.ac.', 'university', 'college', 'school']
        for indicator in edu_indicators:
            if indicator in combined_text:
                return 0.6
        
        return 0.2  # Default low score for unknown sources
    
    def _score_author_credibility(self, metadata: PDFMetadata) -> float:
        """Score author credibility based on academic indicators."""
        if not metadata.authors:
            return 0.0
        
        credibility_score = 0.0
        author_text = ' '.join(metadata.authors).lower()
        
        # Academic titles and credentials
        academic_titles = [
            'dr.', 'prof.', 'professor', 'ph.d', 'phd', 'md', 'm.d.',
            'assistant professor', 'associate professor', 'lecturer',
            'researcher', 'scientist', 'fellow'
        ]
        
        for title in academic_titles:
            if title in author_text:
                credibility_score += 0.3
        
        # Multiple authors (collaboration indicator)
        if len(metadata.authors) > 1:
            credibility_score += 0.2
        
        # Author from known institution
        if metadata.institution:
            credibility_score += 0.2
        
        return min(credibility_score, 1.0)
    
    def _validate_file_quality(self, metadata: PDFMetadata) -> float:
        """Validate file quality based on size, pages, and text ratio."""
        quality_score = 1.0
        
        # File size validation
        if metadata.file_size < self.min_file_size:
            quality_score -= 0.5  # Too small
        elif metadata.file_size > self.max_file_size:
            quality_score -= 0.3  # Very large, might be scanned
        
        # Page count validation
        if metadata.page_count > 0:
            if metadata.page_count < self.min_pages:
                quality_score -= 0.4  # Too few pages
            elif metadata.page_count > 1000:
                quality_score -= 0.2  # Unusually long
        
        # Text ratio validation (text vs images)
        if metadata.text_ratio < self.min_text_ratio:
            quality_score -= 0.6  # Mostly images, likely scanned
        
        return max(quality_score, 0.0)
    
    def _detect_academic_structure(self, metadata: PDFMetadata, 
                                 content_text: Optional[str] = None) -> float:
        """Detect academic document structure patterns."""
        structure_score = 0.0
        
        # Check for course codes in title or content
        text_to_check = []
        if metadata.title:
            text_to_check.append(metadata.title)
        if metadata.course_code:
            text_to_check.append(metadata.course_code)
        if content_text:
            text_to_check.append(content_text[:500])  # First 500 chars
        
        combined_text = ' '.join(text_to_check)
        
        # Look for course code patterns
        for pattern in self.course_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                structure_score += 0.3
        
        # Check for academic document structure keywords
        structure_keywords = [
            'table of contents', 'bibliography', 'references', 'abstract',
            'introduction', 'conclusion', 'methodology', 'results',
            'discussion', 'appendix', 'syllabus', 'assignment'
        ]
        
        for keyword in structure_keywords:
            if keyword.lower() in combined_text.lower():
                structure_score += 0.1
        
        return min(structure_score, 1.0)
    
    def filter_batch(self, pdf_list: List[PDFMetadata], 
                    content_texts: Optional[Dict[str, str]] = None) -> List[Tuple[PDFMetadata, FilterResult]]:
        """
        Filter a batch of PDFs for educational relevance.
        
        Args:
            pdf_list: List of PDF metadata objects
            content_texts: Optional dictionary mapping filenames to extracted text
            
        Returns:
            List of tuples (metadata, filter_result) for educational PDFs
        """
        results = []
        
        for pdf_metadata in pdf_list:
            content_text = None
            if content_texts and pdf_metadata.filename in content_texts:
                content_text = content_texts[pdf_metadata.filename]
            
            filter_result = self.analyze_educational_relevance(pdf_metadata, content_text)
            
            if filter_result.is_educational:
                results.append((pdf_metadata, filter_result))
                self.logger.info(f"Educational PDF found: {pdf_metadata.filename} "
                               f"(relevance: {filter_result.relevance_score:.2f})")
            else:
                self.logger.debug(f"Non-educational PDF filtered: {pdf_metadata.filename} "
                                f"(relevance: {filter_result.relevance_score:.2f})")
        
        return results