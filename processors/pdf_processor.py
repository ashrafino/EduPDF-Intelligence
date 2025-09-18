"""
PDF processing functions for worker pool tasks.
Handles metadata extraction, content analysis, and quality assessment.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from data.models import ProcessingTask, TaskType, PDFMetadata
from processors.metadata_extractor import PDFMetadataExtractor


def process_pdf_task(task: ProcessingTask) -> Dict[str, Any]:
    """
    Process a PDF task based on its type.
    
    Args:
        task: Processing task to execute
        
    Returns:
        Dictionary containing processing results
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        if task.task_type == TaskType.EXTRACT_METADATA:
            result = extract_pdf_metadata(task)
        elif task.task_type == TaskType.CLASSIFY_CONTENT:
            result = classify_pdf_content(task)
        elif task.task_type == TaskType.DEDUPLICATE:
            result = check_pdf_duplicates(task)
        else:
            result = {'error': f'Unknown task type: {task.task_type}'}
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        logger.debug(f"Processed task {task.task_id} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error processing task {task.task_id}: {e}")
        return {
            'error': str(e),
            'processing_time': time.time() - start_time
        }


def extract_pdf_metadata(task: ProcessingTask) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a PDF file using the enhanced extractor.
    
    Args:
        task: Processing task containing PDF file information
        
    Returns:
        Dictionary containing extracted metadata
    """
    logger = logging.getLogger(__name__)
    
    try:
        file_path = task.metadata.get('file_path', '')
        if not file_path:
            return {'error': 'No file path provided in task metadata'}
        
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            return {'error': f'PDF file not found: {file_path}'}
        
        # Use the comprehensive metadata extractor
        extractor = PDFMetadataExtractor()
        pdf_metadata = extractor.extract_comprehensive_metadata(pdf_path)
        
        # Convert PDFMetadata to dictionary for return
        result = {
            'task_type': 'extract_metadata',
            'file_path': file_path,
            'metadata': {
                'filename': pdf_metadata.filename,
                'title': pdf_metadata.title,
                'authors': pdf_metadata.authors,
                'institution': pdf_metadata.institution,
                'subject_area': pdf_metadata.subject_area,
                'academic_level': pdf_metadata.academic_level.value,
                'language': pdf_metadata.language,
                'language_confidence': pdf_metadata.language_confidence,
                'keywords': pdf_metadata.keywords,
                'page_count': pdf_metadata.page_count,
                'text_ratio': pdf_metadata.text_ratio,
                'quality_score': pdf_metadata.quality_score,
                'readability_score': pdf_metadata.readability_score,
                'content_hash': pdf_metadata.content_hash,
                'file_size': pdf_metadata.file_size,
                'is_processed': pdf_metadata.is_processed,
                'processing_errors': pdf_metadata.processing_errors
            },
            'extracted_at': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully extracted metadata for {pdf_metadata.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_pdf_metadata: {e}")
        return {
            'error': str(e),
            'task_type': 'extract_metadata',
            'file_path': task.metadata.get('file_path', ''),
            'extracted_at': datetime.now().isoformat()
        }


def classify_pdf_content(task: ProcessingTask) -> Dict[str, Any]:
    """
    Classify PDF content by academic level and subject area using the content analyzer.
    
    Args:
        task: Processing task containing PDF file information
        
    Returns:
        Dictionary containing classification results
    """
    logger = logging.getLogger(__name__)
    
    try:
        file_path = task.metadata.get('file_path', '')
        if not file_path:
            return {'error': 'No file path provided in task metadata'}
        
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            return {'error': f'PDF file not found: {file_path}'}
        
        # Extract metadata first (which includes content analysis)
        extractor = PDFMetadataExtractor()
        pdf_metadata = extractor.extract_comprehensive_metadata(pdf_path)
        
        # Return classification results
        result = {
            'task_type': 'classify_content',
            'file_path': file_path,
            'subject_area': pdf_metadata.subject_area,
            'academic_level': pdf_metadata.academic_level.value,
            'language': pdf_metadata.language,
            'language_confidence': pdf_metadata.language_confidence,
            'keywords': pdf_metadata.keywords,
            'tags': pdf_metadata.tags,
            'quality_score': pdf_metadata.quality_score,
            'classified_at': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully classified content for {pdf_metadata.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in classify_pdf_content: {e}")
        return {
            'error': str(e),
            'task_type': 'classify_content',
            'file_path': task.metadata.get('file_path', ''),
            'classified_at': datetime.now().isoformat()
        }


def check_pdf_duplicates(task: ProcessingTask) -> Dict[str, Any]:
    """
    Check for duplicate PDFs using content hashing.
    
    Args:
        task: Processing task containing PDF file information
        
    Returns:
        Dictionary containing duplicate check results
    """
    # Placeholder implementation
    # In a real implementation, this would calculate content hashes
    
    file_path = task.metadata.get('file_path', '')
    
    # Simulate duplicate checking
    time.sleep(0.05)
    
    return {
        'task_type': 'deduplicate',
        'file_path': file_path,
        'content_hash': 'abc123def456',
        'is_duplicate': False,
        'similar_files': [],
        'checked_at': datetime.now().isoformat()
    }