#!/usr/bin/env python3
"""
Test script for smart content grouping system.
Tests related content detection, course material grouping, series detection, and recommendations.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.models import PDFMetadata, AcademicLevel
from utils.content_grouping import (
    SmartContentGrouper, ContentGroup, CourseStructure, SeriesInfo
)


def create_test_pdf_collection() -> List[PDFMetadata]:
    """Create a comprehensive test collection with various grouping scenarios."""
    test_pdfs = [
        # Course materials for CS101
        PDFMetadata(
            filename="cs101_lecture_01.pdf",
            file_path="/test/cs101_lecture_01.pdf",
            file_size=2000000,
            title="CS101 Introduction to Programming - Lecture 1",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "introduction", "variables", "cs101"],
            course_code="CS101",
            page_count=25,
            text_ratio=0.8,
            quality_score=0.9
        ),
        
        PDFMetadata(
            filename="cs101_lecture_02.pdf",
            file_path="/test/cs101_lecture_02.pdf",
            file_size=2100000,
            title="CS101 Control Structures - Lecture 2",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "control structures", "loops", "cs101"],
            course_code="CS101",
            page_count=28,
            text_ratio=0.82,
            quality_score=0.88
        ),
        
        PDFMetadata(
            filename="cs101_assignment_1.pdf",
            file_path="/test/cs101_assignment_1.pdf",
            file_size=500000,
            title="CS101 Programming Assignment 1",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "assignment", "homework", "cs101"],
            course_code="CS101",
            page_count=5,
            text_ratio=0.75,
            quality_score=0.85
        ),
        
        PDFMetadata(
            filename="cs101_midterm_exam.pdf",
            file_path="/test/cs101_midterm_exam.pdf",
            file_size=300000,
            title="CS101 Midterm Examination",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "exam", "midterm", "cs101"],
            course_code="CS101",
            page_count=8,
            text_ratio=0.7,
            quality_score=0.8
        ),
        
        # Series: Machine Learning Textbook
        PDFMetadata(
            filename="ml_textbook_part_1.pdf",
            file_path="/test/ml_textbook_part_1.pdf",
            file_size=8000000,
            title="Machine Learning Fundamentals - Part 1",
            authors=["Dr. Alice Johnson", "Dr. Bob Wilson"],
            institution="AI Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "fundamentals", "algorithms", "textbook"],
            page_count=200,
            text_ratio=0.85,
            quality_score=0.95
        ),
        
        PDFMetadata(
            filename="ml_textbook_part_2.pdf",
            file_path="/test/ml_textbook_part_2.pdf",
            file_size=8500000,
            title="Machine Learning Fundamentals - Part 2",
            authors=["Dr. Alice Johnson", "Dr. Bob Wilson"],
            institution="AI Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "neural networks", "deep learning", "textbook"],
            page_count=220,
            text_ratio=0.87,
            quality_score=0.96
        ),
        
        PDFMetadata(
            filename="ml_textbook_part_3.pdf",
            file_path="/test/ml_textbook_part_3.pdf",
            file_size=7800000,
            title="Machine Learning Fundamentals - Part 3",
            authors=["Dr. Alice Johnson", "Dr. Bob Wilson"],
            institution="AI Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "applications", "case studies", "textbook"],
            page_count=180,
            text_ratio=0.83,
            quality_score=0.94
        ),
        
        # Related content by same author (different courses)
        PDFMetadata(
            filename="advanced_algorithms.pdf",
            file_path="/test/advanced_algorithms.pdf",
            file_size=6000000,
            title="Advanced Algorithms and Data Structures",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["algorithms", "data structures", "complexity", "advanced"],
            page_count=350,
            text_ratio=0.88,
            quality_score=0.92
        ),
        
        PDFMetadata(
            filename="software_engineering.pdf",
            file_path="/test/software_engineering.pdf",
            file_size=5500000,
            title="Software Engineering Principles",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["software engineering", "design patterns", "testing", "agile"],
            page_count=280,
            text_ratio=0.84,
            quality_score=0.89
        ),
        
        # Mathematics course materials
        PDFMetadata(
            filename="calc_chapter_1.pdf",
            file_path="/test/calc_chapter_1.pdf",
            file_size=3000000,
            title="Calculus I - Chapter 1: Limits",
            authors=["Prof. Maria Garcia"],
            institution="Math College",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "limits", "continuity", "math201"],
            course_code="MATH201",
            page_count=45,
            text_ratio=0.9,
            quality_score=0.91
        ),
        
        PDFMetadata(
            filename="calc_chapter_2.pdf",
            file_path="/test/calc_chapter_2.pdf",
            file_size=3200000,
            title="Calculus I - Chapter 2: Derivatives",
            authors=["Prof. Maria Garcia"],
            institution="Math College",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "derivatives", "differentiation", "math201"],
            course_code="MATH201",
            page_count=52,
            text_ratio=0.88,
            quality_score=0.93
        ),
        
        # Physics materials from same institution
        PDFMetadata(
            filename="quantum_mechanics_intro.pdf",
            file_path="/test/quantum_mechanics_intro.pdf",
            file_size=4500000,
            title="Introduction to Quantum Mechanics",
            authors=["Dr. David Einstein"],
            institution="Physics Institute",
            subject_area="physics",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["quantum mechanics", "wave function", "schrodinger", "physics"],
            page_count=180,
            text_ratio=0.85,
            quality_score=0.94
        ),
        
        PDFMetadata(
            filename="statistical_mechanics.pdf",
            file_path="/test/statistical_mechanics.pdf",
            file_size=4200000,
            title="Statistical Mechanics and Thermodynamics",
            authors=["Dr. Sarah Newton"],
            institution="Physics Institute",
            subject_area="physics",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["statistical mechanics", "thermodynamics", "entropy", "physics"],
            page_count=165,
            text_ratio=0.87,
            quality_score=0.92
        ),
        
        # Business course with numbered sections
        PDFMetadata(
            filename="business_strategy_section_1.pdf",
            file_path="/test/business_strategy_section_1.pdf",
            file_size=2500000,
            title="Business Strategy - Section 1 (Strategic Analysis)",
            authors=["Prof. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business strategy", "strategic analysis", "competitive advantage"],
            page_count=40,
            text_ratio=0.82,
            quality_score=0.88
        ),
        
        PDFMetadata(
            filename="business_strategy_section_2.pdf",
            file_path="/test/business_strategy_section_2.pdf",
            file_size=2700000,
            title="Business Strategy - Section 2 (Implementation)",
            authors=["Prof. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business strategy", "implementation", "change management"],
            page_count=45,
            text_ratio=0.84,
            quality_score=0.90
        )
    ]
    
    return test_pdfs


def test_related_content_grouping():
    """Test related content detection and grouping."""
    print("Testing Related Content Grouping")
    print("=" * 40)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Test related content grouping
    related_groups = grouper.group_related_content(test_pdfs)
    
    print(f"Found {len(related_groups)} related content groups:")
    for group_name, pdf_list in related_groups.items():
        print(f"\n{group_name}:")
        for pdf_filename in pdf_list:
            print(f"  - {pdf_filename}")
    
    return related_groups


def test_course_material_detection():
    """Test course material detection and organization."""
    print("\nTesting Course Material Detection")
    print("=" * 40)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Detect course materials
    course_structures = grouper.detect_course_materials(test_pdfs)
    
    print(f"Found {len(course_structures)} courses:")
    for course_code, course in course_structures.items():
        print(f"\nCourse: {course_code}")
        print(f"  Name: {course.course_name}")
        print(f"  Institution: {course.institution}")
        print(f"  Academic Level: {course.academic_level.value}")
        
        if course.lectures:
            print(f"  Lectures ({len(course.lectures)}):")
            for lecture in course.lectures:
                print(f"    - {lecture}")
        
        if course.assignments:
            print(f"  Assignments ({len(course.assignments)}):")
            for assignment in course.assignments:
                print(f"    - {assignment}")
        
        if course.exams:
            print(f"  Exams ({len(course.exams)}):")
            for exam in course.exams:
                print(f"    - {exam}")
        
        if course.textbooks:
            print(f"  Textbooks ({len(course.textbooks)}):")
            for textbook in course.textbooks:
                print(f"    - {textbook}")
        
        if course.supplementary:
            print(f"  Supplementary ({len(course.supplementary)}):")
            for supp in course.supplementary:
                print(f"    - {supp}")
    
    return course_structures


def test_series_detection():
    """Test document series detection."""
    print("\nTesting Document Series Detection")
    print("=" * 40)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Detect series
    series_info = grouper.detect_document_series(test_pdfs)
    
    print(f"Found {len(series_info)} document series:")
    for series_name, series in series_info.items():
        print(f"\nSeries: {series_name}")
        print(f"  Total Parts: {series.total_parts}")
        print(f"  Identified Parts: {series.identified_parts}")
        print(f"  Complete: {series.is_complete()}")
        
        if not series.is_complete():
            missing = series.get_missing_parts()
            print(f"  Missing Parts: {missing}")
        
        print("  Parts Found:")
        for part_num in sorted(series.part_filenames.keys()):
            filename = series.part_filenames[part_num]
            print(f"    Part {part_num}: {filename}")
    
    return series_info


def test_recommendation_system():
    """Test content recommendation system."""
    print("\nTesting Recommendation System")
    print("=" * 40)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Test recommendations for a specific PDF
    target_pdf = test_pdfs[0]  # CS101 lecture 1
    print(f"Generating recommendations for: {target_pdf.filename}")
    print(f"Title: {target_pdf.title}")
    
    recommendations = grouper.generate_recommendations(target_pdf, test_pdfs, max_recommendations=5)
    
    print(f"\nTop {len(recommendations)} recommendations:")
    for i, (filename, similarity) in enumerate(recommendations, 1):
        print(f"  {i}. {filename} (similarity: {similarity:.3f})")
    
    # Test recommendations for different types of content
    print("\n" + "-" * 30)
    
    # Find a machine learning PDF
    ml_pdf = next((pdf for pdf in test_pdfs if "machine learning" in pdf.title.lower()), None)
    if ml_pdf:
        print(f"Generating recommendations for: {ml_pdf.filename}")
        ml_recommendations = grouper.generate_recommendations(ml_pdf, test_pdfs, max_recommendations=3)
        
        print(f"Top {len(ml_recommendations)} recommendations:")
        for i, (filename, similarity) in enumerate(ml_recommendations, 1):
            print(f"  {i}. {filename} (similarity: {similarity:.3f})")
    
    return recommendations


def test_content_group_creation():
    """Test comprehensive content group creation."""
    print("\nTesting Content Group Creation")
    print("=" * 40)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Create all content groups
    content_groups = grouper.create_content_groups(test_pdfs)
    
    print(f"Created {len(content_groups)} content groups:")
    
    for group in content_groups:
        print(f"\nGroup: {group.title}")
        print(f"  Type: {group.group_type}")
        print(f"  ID: {group.group_id}")
        print(f"  Size: {group.get_size()} PDFs")
        
        if group.metadata:
            print("  Metadata:")
            for key, value in group.metadata.items():
                print(f"    {key}: {value}")
        
        print("  PDFs:")
        for pdf_filename in group.pdfs[:5]:  # Show first 5
            print(f"    - {pdf_filename}")
        
        if len(group.pdfs) > 5:
            print(f"    ... and {len(group.pdfs) - 5} more")
    
    return content_groups


def test_group_persistence():
    """Test saving and loading content groups."""
    print("\nTesting Group Persistence")
    print("=" * 30)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Create groups
    content_groups = grouper.create_content_groups(test_pdfs)
    
    # Save groups
    test_file = "test_content_groups.json"
    grouper.save_groups(content_groups, test_file)
    print(f"Saved {len(content_groups)} groups to {test_file}")
    
    # Load groups
    loaded_groups = grouper.load_groups(test_file)
    print(f"Loaded {len(loaded_groups)} groups from {test_file}")
    
    # Verify groups are identical
    if len(content_groups) == len(loaded_groups):
        print("✓ Group count matches")
        
        # Check first group details
        if content_groups and loaded_groups:
            original = content_groups[0]
            loaded = loaded_groups[0]
            
            if (original.group_id == loaded.group_id and 
                original.title == loaded.title and
                original.pdfs == loaded.pdfs):
                print("✓ Group details match")
            else:
                print("✗ Group details don't match")
    else:
        print("✗ Group count doesn't match")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"Cleaned up {test_file}")
    
    return loaded_groups


def test_group_statistics():
    """Test group statistics generation."""
    print("\nTesting Group Statistics")
    print("=" * 30)
    
    grouper = SmartContentGrouper()
    test_pdfs = create_test_pdf_collection()
    
    # Create groups
    content_groups = grouper.create_content_groups(test_pdfs)
    
    # Get statistics
    stats = grouper.get_group_statistics(content_groups)
    
    print("Group Statistics:")
    print(f"  Total Groups: {stats['total_groups']}")
    print(f"  Total PDFs Grouped: {stats['total_pdfs_grouped']}")
    print(f"  Average Group Size: {stats['average_group_size']:.1f}")
    
    print("\nBy Group Type:")
    for group_type, count in stats['by_type'].items():
        print(f"  {group_type}: {count} groups")
    
    print("\nBy Group Size:")
    for size_category, count in stats['by_size'].items():
        print(f"  {size_category}: {count} groups")
    
    return stats


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases")
    print("=" * 25)
    
    grouper = SmartContentGrouper()
    
    # Test with empty list
    empty_groups = grouper.create_content_groups([])
    print(f"Empty collection result: {len(empty_groups)} groups")
    
    # Test with single PDF
    single_pdf = [create_test_pdf_collection()[0]]
    single_groups = grouper.create_content_groups(single_pdf)
    print(f"Single PDF result: {len(single_groups)} groups")
    
    # Test with PDFs with minimal metadata
    minimal_pdf = PDFMetadata(
        filename="minimal.pdf",
        file_path="/test/minimal.pdf",
        file_size=1000000,
        title="Document",
        academic_level=AcademicLevel.UNKNOWN,
        page_count=10,
        text_ratio=0.5,
        quality_score=0.3
    )
    
    minimal_groups = grouper.create_content_groups([minimal_pdf])
    print(f"Minimal metadata result: {len(minimal_groups)} groups")
    
    # Test recommendations with no similar content
    recommendations = grouper.generate_recommendations(minimal_pdf, [minimal_pdf])
    print(f"Self-recommendation result: {len(recommendations)} recommendations")


if __name__ == "__main__":
    # Run all tests
    try:
        print("Smart Content Grouping System Tests")
        print("=" * 50)
        
        related_groups = test_related_content_grouping()
        course_structures = test_course_material_detection()
        series_info = test_series_detection()
        recommendations = test_recommendation_system()
        content_groups = test_content_group_creation()
        loaded_groups = test_group_persistence()
        stats = test_group_statistics()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("All Smart Content Grouping Tests Completed Successfully!")
        print(f"- Found {len(related_groups)} related content groups")
        print(f"- Detected {len(course_structures)} course structures")
        print(f"- Identified {len(series_info)} document series")
        print(f"- Created {len(content_groups)} total content groups")
        print(f"- Generated recommendations with similarity scoring")
        print(f"- Tested persistence and statistics functionality")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()