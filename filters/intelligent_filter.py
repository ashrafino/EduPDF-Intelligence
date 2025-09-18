"""
Main intelligent content filtering system.
Combines educational relevance filtering and academic level classification.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data.models import PDFMetadata, AcademicLevel
from filters.content_filter import EducationalRelevanceFilter, FilterResult
from filters.academic_classifier import AcademicLevelClassifier, ClassificationResult


@dataclass
class IntelligentFilterResult:
    """Combined result of intelligent content filtering."""
    pdf_metadata: PDFMetadata
    is_educational: bool
    relevance_score: float
    predicted_level: AcademicLevel
    classification_confidence: float
    quality_score: float
    complexity_score: float
    should_include: bool
    filter_reasons: List[str]
    classification_reasons: List[str]


class IntelligentContentFilter:
    """
    Main intelligent content filtering system that combines educational relevance
    filtering with academic level classification for comprehensive PDF analysis.
    """
    
    def __init__(self, min_relevance_score: float = 0.4, 
                 min_quality_score: float = 0.5,
                 min_classification_confidence: float = 0.3):
        """
        Initialize the intelligent content filter.
        
        Args:
            min_relevance_score: Minimum educational relevance score (0-1)
            min_quality_score: Minimum quality score (0-1)
            min_classification_confidence: Minimum classification confidence (0-1)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize component filters
        self.relevance_filter = EducationalRelevanceFilter()
        self.level_classifier = AcademicLevelClassifier()
        
        # Filter thresholds
        self.min_relevance_score = min_relevance_score
        self.min_quality_score = min_quality_score
        self.min_classification_confidence = min_classification_confidence
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'educational_found': 0,
            'high_school': 0,
            'undergraduate': 0,
            'graduate': 0,
            'unknown_level': 0,
            'filtered_out': 0
        }
    
    def filter_pdf(self, metadata: PDFMetadata, 
                   content_text: Optional[str] = None) -> IntelligentFilterResult:
        """
        Apply intelligent filtering to a single PDF.
        
        Args:
            metadata: PDF metadata object
            content_text: Optional extracted text content
            
        Returns:
            IntelligentFilterResult with comprehensive analysis
        """
        self.stats['total_processed'] += 1
        
        # Step 1: Educational relevance filtering
        filter_result = self.relevance_filter.analyze_educational_relevance(
            metadata, content_text
        )
        
        # Step 2: Academic level classification (only if educational)
        classification_result = None
        if filter_result.is_educational:
            classification_result = self.level_classifier.classify_academic_level(
                metadata, content_text
            )
            
            # Update metadata with classification results
            metadata.academic_level = classification_result.predicted_level
            metadata.readability_score = classification_result.readability_score
            metadata.quality_score = filter_result.quality_score
        else:
            # Create default classification result for non-educational content
            classification_result = ClassificationResult(
                predicted_level=AcademicLevel.UNKNOWN,
                confidence=0.0,
                complexity_score=0.0,
                readability_score=0.0,
                features={},
                reasoning=["Content not classified as educational"]
            )
        
        # Step 3: Make final inclusion decision
        should_include = self._should_include_pdf(filter_result, classification_result)
        
        # Update statistics
        self._update_statistics(filter_result, classification_result, should_include)
        
        # Create combined result
        result = IntelligentFilterResult(
            pdf_metadata=metadata,
            is_educational=filter_result.is_educational,
            relevance_score=filter_result.relevance_score,
            predicted_level=classification_result.predicted_level,
            classification_confidence=classification_result.confidence,
            quality_score=filter_result.quality_score,
            complexity_score=classification_result.complexity_score,
            should_include=should_include,
            filter_reasons=filter_result.reasons,
            classification_reasons=classification_result.reasoning
        )
        
        # Log result
        self._log_filter_result(result)
        
        return result
    
    def filter_batch(self, pdf_list: List[PDFMetadata],
                    content_texts: Optional[Dict[str, str]] = None) -> List[IntelligentFilterResult]:
        """
        Apply intelligent filtering to a batch of PDFs.
        
        Args:
            pdf_list: List of PDF metadata objects
            content_texts: Optional dictionary mapping filenames to extracted text
            
        Returns:
            List of IntelligentFilterResult objects
        """
        results = []
        
        self.logger.info(f"Starting intelligent filtering of {len(pdf_list)} PDFs")
        
        for i, pdf_metadata in enumerate(pdf_list):
            content_text = None
            if content_texts and pdf_metadata.filename in content_texts:
                content_text = content_texts[pdf_metadata.filename]
            
            result = self.filter_pdf(pdf_metadata, content_text)
            results.append(result)
            
            # Log progress for large batches
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(pdf_list)} PDFs")
        
        # Log final statistics
        self._log_batch_statistics(len(pdf_list))
        
        return results
    
    def get_educational_pdfs(self, pdf_list: List[PDFMetadata],
                           content_texts: Optional[Dict[str, str]] = None) -> List[PDFMetadata]:
        """
        Filter and return only educational PDFs that meet quality criteria.
        
        Args:
            pdf_list: List of PDF metadata objects
            content_texts: Optional dictionary mapping filenames to extracted text
            
        Returns:
            List of PDFMetadata objects for educational PDFs
        """
        results = self.filter_batch(pdf_list, content_texts)
        
        educational_pdfs = []
        for result in results:
            if result.should_include:
                educational_pdfs.append(result.pdf_metadata)
        
        self.logger.info(f"Found {len(educational_pdfs)} educational PDFs out of {len(pdf_list)} total")
        
        return educational_pdfs
    
    def _should_include_pdf(self, filter_result: FilterResult, 
                          classification_result: ClassificationResult) -> bool:
        """
        Determine if PDF should be included based on all criteria.
        
        Args:
            filter_result: Educational relevance filter result
            classification_result: Academic level classification result
            
        Returns:
            Boolean indicating if PDF should be included
        """
        # Must be educational
        if not filter_result.is_educational:
            return False
        
        # Must meet minimum relevance score
        if filter_result.relevance_score < self.min_relevance_score:
            return False
        
        # Must meet minimum quality score
        if filter_result.quality_score < self.min_quality_score:
            return False
        
        # Must have reasonable classification confidence (unless unknown is acceptable)
        if (classification_result.predicted_level != AcademicLevel.UNKNOWN and 
            classification_result.confidence < self.min_classification_confidence):
            return False
        
        return True
    
    def _update_statistics(self, filter_result: FilterResult,
                         classification_result: ClassificationResult,
                         should_include: bool):
        """Update internal statistics tracking."""
        if filter_result.is_educational:
            self.stats['educational_found'] += 1
            
            # Count by academic level
            level = classification_result.predicted_level
            if level == AcademicLevel.HIGH_SCHOOL:
                self.stats['high_school'] += 1
            elif level == AcademicLevel.UNDERGRADUATE:
                self.stats['undergraduate'] += 1
            elif level == AcademicLevel.GRADUATE:
                self.stats['graduate'] += 1
            else:
                self.stats['unknown_level'] += 1
        
        if not should_include:
            self.stats['filtered_out'] += 1
    
    def _log_filter_result(self, result: IntelligentFilterResult):
        """Log the filtering result for debugging and monitoring."""
        if result.should_include:
            self.logger.info(
                f"INCLUDED: {result.pdf_metadata.filename} - "
                f"Level: {result.predicted_level.value}, "
                f"Relevance: {result.relevance_score:.2f}, "
                f"Quality: {result.quality_score:.2f}, "
                f"Confidence: {result.classification_confidence:.2f}"
            )
        else:
            self.logger.debug(
                f"FILTERED: {result.pdf_metadata.filename} - "
                f"Educational: {result.is_educational}, "
                f"Relevance: {result.relevance_score:.2f}, "
                f"Quality: {result.quality_score:.2f}"
            )
    
    def _log_batch_statistics(self, total_processed: int):
        """Log statistics for the batch processing."""
        stats = self.stats
        
        self.logger.info("=== Intelligent Filtering Statistics ===")
        self.logger.info(f"Total PDFs processed: {total_processed}")
        self.logger.info(f"Educational PDFs found: {stats['educational_found']}")
        self.logger.info(f"High School level: {stats['high_school']}")
        self.logger.info(f"Undergraduate level: {stats['undergraduate']}")
        self.logger.info(f"Graduate level: {stats['graduate']}")
        self.logger.info(f"Unknown level: {stats['unknown_level']}")
        self.logger.info(f"Filtered out: {stats['filtered_out']}")
        
        if total_processed > 0:
            educational_rate = (stats['educational_found'] / total_processed) * 100
            inclusion_rate = ((stats['educational_found'] - stats['filtered_out']) / total_processed) * 100
            self.logger.info(f"Educational rate: {educational_rate:.1f}%")
            self.logger.info(f"Final inclusion rate: {inclusion_rate:.1f}%")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get current filtering statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
    
    def update_thresholds(self, min_relevance_score: Optional[float] = None,
                         min_quality_score: Optional[float] = None,
                         min_classification_confidence: Optional[float] = None):
        """
        Update filtering thresholds.
        
        Args:
            min_relevance_score: New minimum relevance score
            min_quality_score: New minimum quality score
            min_classification_confidence: New minimum classification confidence
        """
        if min_relevance_score is not None:
            self.min_relevance_score = min_relevance_score
            self.logger.info(f"Updated minimum relevance score to {min_relevance_score}")
        
        if min_quality_score is not None:
            self.min_quality_score = min_quality_score
            self.logger.info(f"Updated minimum quality score to {min_quality_score}")
        
        if min_classification_confidence is not None:
            self.min_classification_confidence = min_classification_confidence
            self.logger.info(f"Updated minimum classification confidence to {min_classification_confidence}")