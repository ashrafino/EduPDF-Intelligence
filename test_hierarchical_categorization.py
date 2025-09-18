#!/usr/bin/env python3
"""
Test script for hierarchical categorization system.
Tests automatic folder structure generation, subject taxonomy, and classification.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.models import PDFMetadata, AcademicLevel
from utils.organization import (
    HierarchicalCategorizer, SubjectArea, TopicCategory, SubjectTaxonomy
)


def create_test_metadata() -> list[PDFMetadata]:
    """Create test PDF metadata for different subjects and levels."""
    test_pdfs = [
        # Mathematics PDFs
        PDFMetadata(
            filename="calculus_textbook.pdf",
            file_path="/test/calculus_textbook.pdf",
            file_size=5000000,
            title="Introduction to Calculus and Differential Equations",
            authors=["Dr. John Smith", "Prof. Jane Doe"],
            institution="MIT",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "derivative", "integral", "differential equations", "limits"],
            page_count=450,
            text_ratio=0.85,
            quality_score=0.92,
            description="Comprehensive textbook covering calculus fundamentals and applications"
        ),
        
        PDFMetadata(
            filename="linear_algebra_notes.pdf",
            file_path="/test/linear_algebra_notes.pdf",
            file_size=2500000,
            title="Linear Algebra and Matrix Theory",
            authors=["Prof. Alice Johnson"],
            institution="Stanford University",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["linear algebra", "matrix", "vector", "eigenvalue", "determinant"],
            page_count=280,
            text_ratio=0.78,
            quality_score=0.88,
            description="Course notes on linear algebra with practical examples"
        ),
        
        # Computer Science PDFs
        PDFMetadata(
            filename="machine_learning_guide.pdf",
            file_path="/test/machine_learning_guide.pdf",
            file_size=8000000,
            title="Machine Learning: Algorithms and Applications",
            authors=["Dr. Robert Chen", "Dr. Sarah Wilson"],
            institution="Carnegie Mellon University",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "neural networks", "deep learning", "algorithms", "AI"],
            page_count=650,
            text_ratio=0.82,
            quality_score=0.95,
            description="Advanced guide to machine learning techniques and implementations"
        ),
        
        PDFMetadata(
            filename="python_programming.pdf",
            file_path="/test/python_programming.pdf",
            file_size=3200000,
            title="Python Programming for Beginners",
            authors=["Prof. Michael Brown"],
            institution="University of California",
            subject_area="computer_science",
            academic_level=AcademicLevel.HIGH_SCHOOL,
            keywords=["python", "programming", "coding", "data structures", "algorithms"],
            page_count=320,
            text_ratio=0.75,
            quality_score=0.85,
            description="Introduction to Python programming with hands-on examples"
        ),
        
        # Physics PDFs
        PDFMetadata(
            filename="quantum_mechanics.pdf",
            file_path="/test/quantum_mechanics.pdf",
            file_size=6500000,
            title="Principles of Quantum Mechanics",
            authors=["Prof. David Einstein"],
            institution="Harvard University",
            subject_area="physics",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["quantum mechanics", "wave function", "schrodinger", "particle", "uncertainty"],
            page_count=520,
            text_ratio=0.88,
            quality_score=0.93,
            description="Advanced quantum mechanics theory and applications"
        ),
        
        PDFMetadata(
            filename="classical_mechanics.pdf",
            file_path="/test/classical_mechanics.pdf",
            file_size=4100000,
            title="Classical Mechanics and Motion",
            authors=["Dr. Lisa Newton"],
            institution="Princeton University",
            subject_area="physics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["mechanics", "motion", "force", "energy", "momentum", "kinematics"],
            page_count=380,
            text_ratio=0.81,
            quality_score=0.89,
            description="Fundamentals of classical mechanics with problem sets"
        ),
        
        # Chemistry PDFs
        PDFMetadata(
            filename="organic_chemistry.pdf",
            file_path="/test/organic_chemistry.pdf",
            file_size=7200000,
            title="Organic Chemistry: Structure and Reactions",
            authors=["Prof. Maria Garcia", "Dr. James Lee"],
            institution="Yale University",
            subject_area="chemistry",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["organic chemistry", "carbon", "synthesis", "reaction mechanism", "functional groups"],
            page_count=580,
            text_ratio=0.79,
            quality_score=0.91,
            description="Comprehensive organic chemistry textbook with reaction mechanisms"
        ),
        
        # Biology PDFs
        PDFMetadata(
            filename="molecular_biology.pdf",
            file_path="/test/molecular_biology.pdf",
            file_size=5800000,
            title="Molecular Biology and Genetics",
            authors=["Dr. Susan Clark"],
            institution="Johns Hopkins University",
            subject_area="biology",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["molecular biology", "DNA", "RNA", "protein", "genetics", "gene expression"],
            page_count=480,
            text_ratio=0.84,
            quality_score=0.94,
            description="Advanced molecular biology covering genetic mechanisms"
        ),
        
        # Engineering PDFs
        PDFMetadata(
            filename="electrical_circuits.pdf",
            file_path="/test/electrical_circuits.pdf",
            file_size=4500000,
            title="Electrical Circuits and Systems",
            authors=["Prof. Kevin Zhang"],
            institution="Georgia Tech",
            subject_area="engineering",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["electrical engineering", "circuits", "electronics", "voltage", "current"],
            page_count=420,
            text_ratio=0.77,
            quality_score=0.87,
            description="Introduction to electrical circuits and electronic systems"
        ),
        
        # Business PDFs
        PDFMetadata(
            filename="strategic_management.pdf",
            file_path="/test/strategic_management.pdf",
            file_size=3800000,
            title="Strategic Management and Leadership",
            authors=["Prof. Amanda White"],
            institution="Wharton School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["management", "strategy", "leadership", "organization", "planning"],
            page_count=350,
            text_ratio=0.83,
            quality_score=0.90,
            description="Advanced strategic management concepts and case studies"
        )
    ]
    
    return test_pdfs


def test_hierarchical_categorization():
    """Test the hierarchical categorization system."""
    print("Testing Hierarchical Categorization System")
    print("=" * 50)
    
    # Initialize categorizer
    categorizer = HierarchicalCategorizer("test_output")
    
    # Create test metadata
    test_pdfs = create_test_metadata()
    
    print(f"Testing with {len(test_pdfs)} sample PDFs\n")
    
    # Test classification and folder generation
    results = []
    for pdf in test_pdfs:
        print(f"Processing: {pdf.filename}")
        print(f"  Title: {pdf.title}")
        print(f"  Subject Area: {pdf.subject_area}")
        print(f"  Academic Level: {pdf.academic_level.value}")
        
        # Classify the PDF
        subject, topic, confidence = categorizer.classify_pdf(pdf)
        print(f"  Classified as: {subject.value} -> {topic} (confidence: {confidence:.2f})")
        
        # Generate folder structure
        folder_path = categorizer.generate_folder_structure(pdf)
        print(f"  Folder Path: {folder_path}")
        
        # Create the folder structure
        success = categorizer.create_folder_structure(folder_path)
        print(f"  Folder Created: {success}")
        
        results.append({
            'pdf': pdf,
            'subject': subject,
            'topic': topic,
            'confidence': confidence,
            'folder_path': folder_path,
            'created': success
        })
        
        print()
    
    # Test folder statistics
    print("Folder Statistics:")
    print("-" * 30)
    stats = categorizer.get_folder_statistics()
    
    print(f"Total folders created: {stats['total_folders']}")
    
    print("\nBy Subject:")
    for subject, count in stats['by_subject'].items():
        print(f"  {subject}: {count} PDFs")
    
    print("\nBy Academic Level:")
    for level, count in stats['by_level'].items():
        print(f"  {level}: {count} PDFs")
    
    print("\nBy Topic:")
    for topic, count in stats['by_topic'].items():
        print(f"  {topic}: {count} PDFs")
    
    # Test custom taxonomy addition
    print("\nTesting Custom Taxonomy Addition:")
    print("-" * 40)
    
    # Add a custom topic to computer science
    custom_topic = TopicCategory(
        name="web_development",
        keywords=["web", "html", "css", "javascript", "frontend", "backend", "react", "node"],
        academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.HIGH_SCHOOL],
        description="Web development technologies and frameworks"
    )
    
    categorizer.add_custom_topic(SubjectArea.COMPUTER_SCIENCE, custom_topic)
    print(f"Added custom topic: {custom_topic.name}")
    
    # Test with a web development PDF
    web_pdf = PDFMetadata(
        filename="web_development_guide.pdf",
        file_path="/test/web_development_guide.pdf",
        file_size=4200000,
        title="Modern Web Development with React and Node.js",
        authors=["Prof. Alex Johnson"],
        institution="Tech University",
        subject_area="computer_science",
        academic_level=AcademicLevel.UNDERGRADUATE,
        keywords=["web development", "react", "javascript", "node.js", "frontend", "backend"],
        page_count=380,
        text_ratio=0.76,
        quality_score=0.86,
        description="Comprehensive guide to modern web development"
    )
    
    subject, topic, confidence = categorizer.classify_pdf(web_pdf)
    print(f"Web dev PDF classified as: {subject.value} -> {topic} (confidence: {confidence:.2f})")
    
    # Test taxonomy persistence
    print("\nTesting Taxonomy Persistence:")
    print("-" * 35)
    
    # Save taxonomies
    categorizer.save_taxonomies("test_taxonomies.json")
    print("Taxonomies saved to test_taxonomies.json")
    
    # Create new categorizer and load taxonomies
    new_categorizer = HierarchicalCategorizer("test_output_2")
    new_categorizer.load_taxonomies("test_taxonomies.json")
    print("Taxonomies loaded in new categorizer")
    
    # Test that custom topic is preserved
    cs_taxonomy = new_categorizer.taxonomies.get(SubjectArea.COMPUTER_SCIENCE)
    if cs_taxonomy and "web_development" in cs_taxonomy.topics:
        print("Custom topic 'web_development' successfully preserved!")
    else:
        print("ERROR: Custom topic not preserved")
    
    # Test edge cases
    print("\nTesting Edge Cases:")
    print("-" * 25)
    
    # PDF with minimal metadata
    minimal_pdf = PDFMetadata(
        filename="unknown_document.pdf",
        file_path="/test/unknown_document.pdf",
        file_size=1000000,
        title="Document",
        academic_level=AcademicLevel.UNKNOWN,
        page_count=10,
        text_ratio=0.5,
        quality_score=0.3
    )
    
    subject, topic, confidence = categorizer.classify_pdf(minimal_pdf)
    print(f"Minimal PDF classified as: {subject.value} -> {topic} (confidence: {confidence:.2f})")
    
    # PDF with mixed keywords
    mixed_pdf = PDFMetadata(
        filename="interdisciplinary_study.pdf",
        file_path="/test/interdisciplinary_study.pdf",
        file_size=3500000,
        title="Mathematical Models in Biology and Physics",
        authors=["Dr. Cross Disciplinary"],
        academic_level=AcademicLevel.GRADUATE,
        keywords=["mathematics", "biology", "physics", "modeling", "differential equations"],
        page_count=250,
        text_ratio=0.82,
        quality_score=0.88,
        description="Interdisciplinary approach to mathematical modeling"
    )
    
    subject, topic, confidence = categorizer.classify_pdf(mixed_pdf)
    print(f"Mixed PDF classified as: {subject.value} -> {topic} (confidence: {confidence:.2f})")
    
    print("\nHierarchical Categorization Test Complete!")
    return results


def test_subject_taxonomy():
    """Test individual subject taxonomy functionality."""
    print("\nTesting Subject Taxonomy Functionality:")
    print("=" * 45)
    
    # Create a custom taxonomy for testing
    test_taxonomy = SubjectTaxonomy(SubjectArea.MATHEMATICS)
    
    # Add test topics
    calculus_topic = TopicCategory(
        name="advanced_calculus",
        keywords=["multivariable", "vector calculus", "partial derivative", "multiple integral"],
        academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE],
        description="Advanced calculus topics"
    )
    
    test_taxonomy.add_topic(calculus_topic)
    
    # Test classification
    test_pdf = PDFMetadata(
        filename="vector_calculus.pdf",
        file_path="/test/vector_calculus.pdf",
        file_size=4000000,
        title="Vector Calculus and Multivariable Analysis",
        keywords=["vector calculus", "partial derivative", "gradient", "divergence"],
        academic_level=AcademicLevel.UNDERGRADUATE,
        page_count=300,
        text_ratio=0.8,
        quality_score=0.9
    )
    
    topic, confidence = test_taxonomy.classify_content(test_pdf)
    print(f"Test PDF classified as: {topic} (confidence: {confidence:.2f})")
    
    # Test folder path generation
    folder_path = test_taxonomy.get_folder_path(topic, test_pdf.academic_level)
    print(f"Generated folder path: {folder_path}")
    
    print("Subject Taxonomy Test Complete!")


if __name__ == "__main__":
    # Run all tests
    try:
        results = test_hierarchical_categorization()
        test_subject_taxonomy()
        
        print(f"\nAll tests completed successfully!")
        print(f"Processed {len(results)} PDFs with hierarchical categorization")
        
        # Clean up test files
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
        if os.path.exists("test_output_2"):
            shutil.rmtree("test_output_2")
        if os.path.exists("test_taxonomies.json"):
            os.remove("test_taxonomies.json")
        
        print("Test cleanup completed.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()