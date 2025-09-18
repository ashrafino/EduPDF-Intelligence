#!/usr/bin/env python3
"""
Test script for the content analyzer.
Tests keyword extraction, language detection, and subject classification.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from processors.content_analyzer import ContentAnalyzer
from data.models import PDFMetadata, AcademicLevel


def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_keyword_extraction():
    """Test keyword extraction functionality."""
    analyzer = ContentAnalyzer()
    
    sample_texts = [
        """
        This document provides an introduction to machine learning algorithms and data structures.
        We will cover supervised learning, unsupervised learning, and reinforcement learning.
        Key topics include neural networks, decision trees, and support vector machines.
        Students will learn to implement algorithms in Python and analyze their performance.
        """,
        """
        Calculus is a branch of mathematics that deals with derivatives and integrals.
        This course covers differential calculus, integral calculus, and multivariable calculus.
        Students will learn about limits, continuity, and the fundamental theorem of calculus.
        Applications include optimization problems and area calculations.
        """,
        """
        Physics explores the fundamental laws of nature and the behavior of matter and energy.
        This textbook covers mechanics, thermodynamics, and electromagnetism.
        Topics include Newton's laws, conservation of energy, and electromagnetic fields.
        Students will solve problems involving motion, forces, and wave phenomena.
        """
    ]
    
    print("Testing Keyword Extraction:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts, 1):
        keywords = analyzer.extract_keywords(text, max_keywords=10)
        print(f"Sample {i} keywords: {keywords}")
    
    print()


def test_language_detection():
    """Test language detection functionality."""
    analyzer = ContentAnalyzer()
    
    sample_texts = [
        ("This is an English text about computer science and programming.", "en"),
        ("Este es un texto en español sobre matemáticas y ciencias.", "es"),
        ("Ceci est un texte français sur la physique et la chimie.", "fr"),
        ("Dies ist ein deutscher Text über Ingenieurwissenschaften.", "de"),
        ("Short text", "en")  # Should have low confidence
    ]
    
    print("Testing Language Detection:")
    print("-" * 40)
    
    for text, expected in sample_texts:
        lang, confidence = analyzer.detect_language_with_confidence(text)
        print(f"Text: {text[:50]}...")
        print(f"  Detected: {lang} (confidence: {confidence:.2f})")
        print(f"  Expected: {expected}")
        print(f"  Match: {'✓' if lang == expected else '✗'}")
        print()


def test_subject_classification():
    """Test subject area classification."""
    analyzer = ContentAnalyzer()
    
    sample_texts = [
        ("Programming algorithms data structures computer software development", "computer_science"),
        ("Calculus derivatives integrals mathematics algebra geometry", "mathematics"),
        ("Physics mechanics thermodynamics electromagnetism quantum energy", "physics"),
        ("Chemistry molecules atoms reactions compounds organic inorganic", "chemistry"),
        ("Biology cells genetics evolution ecology DNA RNA proteins", "biology"),
        ("Engineering design systems mechanical electrical manufacturing", "engineering"),
        ("Business management marketing finance economics strategy", "business"),
        ("Psychology behavior cognitive social development personality", "psychology")
    ]
    
    print("Testing Subject Classification:")
    print("-" * 40)
    
    for text, expected in sample_texts:
        subject = analyzer.classify_subject_area(text)
        print(f"Text: {text}")
        print(f"  Classified as: {subject}")
        print(f"  Expected: {expected}")
        print(f"  Match: {'✓' if subject == expected else '✗'}")
        print()


def test_ml_topic_classification():
    """Test machine learning topic classification."""
    analyzer = ContentAnalyzer()
    
    sample_text = """
    This comprehensive guide covers machine learning algorithms and their applications
    in computer science. Students will learn about supervised learning techniques
    including decision trees, neural networks, and support vector machines.
    The course also covers unsupervised learning methods such as clustering and
    dimensionality reduction. Programming assignments will be completed in Python
    using popular libraries like scikit-learn and TensorFlow.
    """
    
    print("Testing ML Topic Classification:")
    print("-" * 40)
    
    topics = analyzer.classify_topics_ml(sample_text)
    print(f"Sample text: {sample_text[:100]}...")
    print(f"Classified topics:")
    for topic, confidence in topics:
        print(f"  {topic}: {confidence:.3f}")
    
    print()


def test_full_content_analysis():
    """Test complete content analysis pipeline."""
    analyzer = ContentAnalyzer()
    
    # Create sample metadata
    metadata = PDFMetadata(
        filename="test_document.pdf",
        file_path="/path/to/test_document.pdf",
        file_size=1024000,
        title="Introduction to Machine Learning"
    )
    
    sample_text = """
    Machine Learning: A Comprehensive Introduction
    
    This textbook provides a thorough introduction to machine learning algorithms
    and their applications in computer science and data analysis. The book covers
    both theoretical foundations and practical implementations.
    
    Chapter 1: Introduction to Machine Learning
    Machine learning is a subset of artificial intelligence that focuses on
    algorithms that can learn from and make predictions on data. This chapter
    introduces key concepts including supervised learning, unsupervised learning,
    and reinforcement learning.
    
    Chapter 2: Supervised Learning Algorithms
    This chapter covers popular supervised learning algorithms including:
    - Linear regression and logistic regression
    - Decision trees and random forests
    - Support vector machines
    - Neural networks and deep learning
    
    Students will implement these algorithms in Python using libraries such as
    scikit-learn, NumPy, and pandas. Programming exercises are provided to
    reinforce theoretical concepts.
    """
    
    print("Testing Full Content Analysis:")
    print("-" * 40)
    
    # Perform analysis
    analyzed_metadata = analyzer.analyze_content(sample_text, metadata)
    
    print(f"Original title: {metadata.title}")
    print(f"Keywords: {analyzed_metadata.keywords}")
    print(f"Language: {analyzed_metadata.language} (confidence: {analyzed_metadata.language_confidence:.2f})")
    print(f"Subject area: {analyzed_metadata.subject_area}")
    print(f"Tags: {analyzed_metadata.tags}")
    print(f"Processing errors: {analyzed_metadata.processing_errors}")
    
    print()


if __name__ == "__main__":
    print("Testing Content Analyzer")
    print("=" * 50)
    
    setup_logging()
    
    test_keyword_extraction()
    test_language_detection()
    test_subject_classification()
    test_ml_topic_classification()
    test_full_content_analysis()
    
    print("Content analyzer testing completed!")