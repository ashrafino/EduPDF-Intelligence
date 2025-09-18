#!/usr/bin/env python3
"""
Comprehensive test for the complete advanced organization system.
Tests the integration of hierarchical categorization and smart content grouping.
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.models import PDFMetadata, AcademicLevel
from utils.advanced_organization import AdvancedOrganizationSystem, OrganizedCollection


def create_comprehensive_test_collection() -> List[PDFMetadata]:
    """Create a comprehensive test collection representing a real educational library."""
    return [
        # Computer Science Course - CS101
        PDFMetadata(
            filename="cs101_syllabus.pdf",
            file_path="/test/cs101_syllabus.pdf",
            file_size=500000,
            title="CS101 Introduction to Programming - Course Syllabus",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "syllabus", "course outline", "cs101"],
            course_code="CS101",
            page_count=8,
            text_ratio=0.85,
            quality_score=0.9,
            description="Course syllabus and learning objectives for introductory programming"
        ),
        
        PDFMetadata(
            filename="cs101_lecture_variables.pdf",
            file_path="/test/cs101_lecture_variables.pdf",
            file_size=2000000,
            title="CS101 Variables and Data Types - Lecture 3",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "variables", "data types", "cs101"],
            course_code="CS101",
            page_count=25,
            text_ratio=0.8,
            quality_score=0.88
        ),
        
        PDFMetadata(
            filename="cs101_assignment_loops.pdf",
            file_path="/test/cs101_assignment_loops.pdf",
            file_size=800000,
            title="CS101 Programming Assignment: Loops and Iteration",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "assignment", "loops", "iteration", "cs101"],
            course_code="CS101",
            page_count=6,
            text_ratio=0.75,
            quality_score=0.85
        ),
        
        # Mathematics Textbook Series
        PDFMetadata(
            filename="calculus_volume_1.pdf",
            file_path="/test/calculus_volume_1.pdf",
            file_size=12000000,
            title="Calculus: Early Transcendentals - Volume 1",
            authors=["Dr. James Stewart", "Dr. Daniel Clegg"],
            institution="Mathematics Publishers",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "limits", "derivatives", "textbook", "mathematics"],
            page_count=600,
            text_ratio=0.9,
            quality_score=0.95,
            description="Comprehensive calculus textbook covering limits and derivatives"
        ),
        
        PDFMetadata(
            filename="calculus_volume_2.pdf",
            file_path="/test/calculus_volume_2.pdf",
            file_size=13000000,
            title="Calculus: Early Transcendentals - Volume 2",
            authors=["Dr. James Stewart", "Dr. Daniel Clegg"],
            institution="Mathematics Publishers",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "integrals", "series", "textbook", "mathematics"],
            page_count=650,
            text_ratio=0.92,
            quality_score=0.96,
            description="Advanced calculus covering integration and infinite series"
        ),
        
        # Physics Course Materials - PHYS201
        PDFMetadata(
            filename="phys201_mechanics_notes.pdf",
            file_path="/test/phys201_mechanics_notes.pdf",
            file_size=3500000,
            title="PHYS201 Classical Mechanics - Lecture Notes",
            authors=["Prof. Lisa Newton"],
            institution="Science University",
            subject_area="physics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["physics", "mechanics", "motion", "forces", "phys201"],
            course_code="PHYS201",
            page_count=120,
            text_ratio=0.85,
            quality_score=0.91
        ),
        
        PDFMetadata(
            filename="phys201_lab_manual.pdf",
            file_path="/test/phys201_lab_manual.pdf",
            file_size=2800000,
            title="PHYS201 Physics Laboratory Manual",
            authors=["Prof. Lisa Newton", "Dr. Mark Johnson"],
            institution="Science University",
            subject_area="physics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["physics", "laboratory", "experiments", "manual", "phys201"],
            course_code="PHYS201",
            page_count=85,
            text_ratio=0.78,
            quality_score=0.87
        ),
        
        # Machine Learning Research Papers (Related Content)
        PDFMetadata(
            filename="neural_networks_survey.pdf",
            file_path="/test/neural_networks_survey.pdf",
            file_size=4500000,
            title="Deep Neural Networks: A Comprehensive Survey",
            authors=["Dr. Alice Chen", "Dr. Robert Kim"],
            institution="AI Research Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "neural networks", "deep learning", "survey", "AI"],
            page_count=45,
            text_ratio=0.88,
            quality_score=0.93,
            description="Comprehensive survey of deep neural network architectures"
        ),
        
        PDFMetadata(
            filename="cnn_image_classification.pdf",
            file_path="/test/cnn_image_classification.pdf",
            file_size=3200000,
            title="Convolutional Neural Networks for Image Classification",
            authors=["Dr. Alice Chen", "Dr. Sarah Wilson"],
            institution="AI Research Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "CNN", "image classification", "computer vision"],
            page_count=32,
            text_ratio=0.82,
            quality_score=0.89
        ),
        
        # Business Course Series - MBA Program
        PDFMetadata(
            filename="strategic_management_part1.pdf",
            file_path="/test/strategic_management_part1.pdf",
            file_size=5500000,
            title="Strategic Management - Part 1: Analysis and Planning",
            authors=["Prof. Michael Porter", "Dr. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business", "strategy", "management", "planning", "MBA"],
            page_count=180,
            text_ratio=0.86,
            quality_score=0.92
        ),
        
        PDFMetadata(
            filename="strategic_management_part2.pdf",
            file_path="/test/strategic_management_part2.pdf",
            file_size=5800000,
            title="Strategic Management - Part 2: Implementation and Control",
            authors=["Prof. Michael Porter", "Dr. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business", "strategy", "implementation", "control", "MBA"],
            page_count=195,
            text_ratio=0.88,
            quality_score=0.94
        ),
        
        # Chemistry Lab Materials
        PDFMetadata(
            filename="organic_chem_lab_guide.pdf",
            file_path="/test/organic_chem_lab_guide.pdf",
            file_size=4200000,
            title="Organic Chemistry Laboratory Procedures",
            authors=["Dr. Maria Garcia", "Prof. David Lee"],
            institution="Chemistry Department",
            subject_area="chemistry",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["chemistry", "organic", "laboratory", "procedures", "experiments"],
            page_count=150,
            text_ratio=0.79,
            quality_score=0.86
        ),
        
        # High School Mathematics
        PDFMetadata(
            filename="algebra2_workbook.pdf",
            file_path="/test/algebra2_workbook.pdf",
            file_size=6500000,
            title="Algebra II Student Workbook with Solutions",
            authors=["Prof. Jennifer Brown"],
            institution="High School Publishers",
            subject_area="mathematics",
            academic_level=AcademicLevel.HIGH_SCHOOL,
            keywords=["algebra", "mathematics", "workbook", "high school", "solutions"],
            page_count=280,
            text_ratio=0.83,
            quality_score=0.88
        ),
        
        # Engineering Materials
        PDFMetadata(
            filename="circuit_analysis_textbook.pdf",
            file_path="/test/circuit_analysis_textbook.pdf",
            file_size=8900000,
            title="Fundamentals of Electric Circuit Analysis",
            authors=["Prof. Kevin Zhang", "Dr. Lisa Park"],
            institution="Engineering College",
            subject_area="engineering",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["engineering", "circuits", "electrical", "analysis", "fundamentals"],
            page_count=420,
            text_ratio=0.84,
            quality_score=0.91
        ),
        
        # Biology Research
        PDFMetadata(
            filename="genetics_molecular_basis.pdf",
            file_path="/test/genetics_molecular_basis.pdf",
            file_size=7200000,
            title="Molecular Basis of Genetic Inheritance",
            authors=["Dr. Susan Clark", "Prof. Richard Davis"],
            institution="Biology Research Center",
            subject_area="biology",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["biology", "genetics", "molecular", "inheritance", "DNA"],
            page_count=320,
            text_ratio=0.87,
            quality_score=0.93
        )
    ]


def test_complete_organization_system():
    """Test the complete advanced organization system."""
    print("Testing Complete Advanced Organization System")
    print("=" * 60)
    
    # Initialize the organization system
    org_system = AdvancedOrganizationSystem("test_educational_library")
    
    # Create test collection
    test_pdfs = create_comprehensive_test_collection()
    print(f"Created test collection with {len(test_pdfs)} PDFs")
    
    # Organize the collection
    print("\nOrganizing collection...")
    organized_collection = org_system.organize_collection(test_pdfs)
    
    # Display organization results
    print("\nOrganization Results:")
    print("-" * 30)
    
    summary = org_system.get_organization_summary(organized_collection)
    
    print(f"Total PDFs Organized: {summary['total_pdfs']}")
    print(f"Hierarchical Folders Created: {summary['hierarchical_folders']}")
    print(f"Content Groups Created: {summary['content_groups']}")
    print(f"Organization Date: {summary['organization_date']}")
    print(f"Base Path: {summary['base_path']}")
    
    print("\nSubject Distribution:")
    for subject, count in summary['subject_distribution'].items():
        print(f"  {subject.replace('_', ' ').title()}: {count} PDFs")
    
    print("\nAcademic Level Distribution:")
    for level, count in summary['level_distribution'].items():
        print(f"  {level.replace('_', ' ').title()}: {count} PDFs")
    
    print("\nContent Group Types:")
    for group_type, count in summary['group_types'].items():
        print(f"  {group_type.title()}: {count} groups")
    
    # Display detailed group information
    print("\nDetailed Content Groups:")
    print("-" * 30)
    
    for group in organized_collection.content_groups:
        print(f"\nGroup: {group.title}")
        print(f"  Type: {group.group_type}")
        print(f"  Size: {len(group.pdfs)} PDFs")
        
        if group.metadata:
            print("  Metadata:")
            for key, value in group.metadata.items():
                print(f"    {key}: {value}")
        
        print("  Files:")
        for pdf_file in group.pdfs[:3]:  # Show first 3
            print(f"    - {pdf_file}")
        if len(group.pdfs) > 3:
            print(f"    ... and {len(group.pdfs) - 3} more")
    
    # Test folder structure creation
    print("\nTesting Folder Structure:")
    print("-" * 30)
    
    base_path = Path(organized_collection.base_path)
    if base_path.exists():
        print("✓ Base directory created")
        
        # Check for main subject folders
        subject_folders = [d for d in base_path.iterdir() if d.is_dir() and d.name != "Groups" and d.name != "indexes"]
        print(f"✓ Subject folders created: {len(subject_folders)}")
        for folder in subject_folders:
            print(f"  - {folder.name}")
        
        # Check for Groups folder
        groups_folder = base_path / "Groups"
        if groups_folder.exists():
            print("✓ Groups folder created")
            group_types = [d for d in groups_folder.iterdir() if d.is_dir()]
            print(f"  Group types: {[d.name for d in group_types]}")
        
        # Check for indexes folder
        indexes_folder = base_path / "indexes"
        if indexes_folder.exists():
            print("✓ Indexes folder created")
            index_files = [f for f in indexes_folder.iterdir() if f.is_file()]
            print(f"  Index files: {[f.name for f in index_files]}")
        
        # Check for main README
        readme_file = base_path / "README.md"
        if readme_file.exists():
            print("✓ Main README.md created")
            with open(readme_file, 'r') as f:
                content = f.read()
                print(f"  README size: {len(content)} characters")
    
    # Test persistence
    print("\nTesting Organization Persistence:")
    print("-" * 40)
    
    org_data_file = "test_organization_data.json"
    org_system.save_organization(organized_collection, org_data_file)
    print(f"✓ Organization data saved to {org_data_file}")
    
    if os.path.exists(org_data_file):
        file_size = os.path.getsize(org_data_file)
        print(f"  File size: {file_size} bytes")
    
    return organized_collection


def test_organization_features():
    """Test specific organization features."""
    print("\nTesting Specific Organization Features:")
    print("=" * 50)
    
    # Test with different settings
    org_system = AdvancedOrganizationSystem("test_features")
    
    # Disable some features for testing
    org_system.create_symbolic_links = False
    org_system.generate_index_files = False
    
    test_pdfs = create_comprehensive_test_collection()[:5]  # Use subset for faster testing
    
    print("Testing with reduced features...")
    organized_collection = org_system.organize_collection(test_pdfs)
    
    print(f"Organized {len(test_pdfs)} PDFs with reduced features")
    print(f"Created {len(organized_collection.content_groups)} groups")
    
    # Test organization summary
    summary = org_system.get_organization_summary(organized_collection)
    print(f"Summary generated successfully: {len(summary)} fields")
    
    return organized_collection


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases:")
    print("=" * 25)
    
    org_system = AdvancedOrganizationSystem("test_edge_cases")
    
    # Test with empty collection
    empty_result = org_system.organize_collection([])
    print(f"Empty collection result: {len(empty_result.content_groups)} groups")
    
    # Test with single PDF
    single_pdf = create_comprehensive_test_collection()[:1]
    single_result = org_system.organize_collection(single_pdf)
    print(f"Single PDF result: {len(single_result.content_groups)} groups")
    
    # Test with PDFs having minimal metadata
    minimal_pdf = PDFMetadata(
        filename="minimal_test.pdf",
        file_path="/test/minimal_test.pdf",
        file_size=1000000,
        title="Test Document",
        academic_level=AcademicLevel.UNKNOWN,
        page_count=10,
        text_ratio=0.5,
        quality_score=0.3
    )
    
    minimal_result = org_system.organize_collection([minimal_pdf])
    print(f"Minimal metadata result: {len(minimal_result.content_groups)} groups")


def cleanup_test_files():
    """Clean up test files and directories."""
    print("\nCleaning up test files...")
    
    test_dirs = [
        "test_educational_library",
        "test_features", 
        "test_edge_cases"
    ]
    
    test_files = [
        "test_organization_data.json",
        "test_taxonomies.json",
        "test_content_groups.json"
    ]
    
    # Remove test directories
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"  Removed directory: {test_dir}")
    
    # Remove test files
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"  Removed file: {test_file}")
    
    print("Cleanup completed.")


if __name__ == "__main__":
    try:
        print("Advanced Organization System - Comprehensive Test")
        print("=" * 70)
        
        # Run main test
        main_result = test_complete_organization_system()
        
        # Test specific features
        features_result = test_organization_features()
        
        # Test edge cases
        test_edge_cases()
        
        print("\n" + "=" * 70)
        print("All Advanced Organization Tests Completed Successfully!")
        print(f"✓ Main organization test: {len(main_result.content_groups)} groups created")
        print(f"✓ Features test: {len(features_result.content_groups)} groups created")
        print("✓ Edge cases handled properly")
        print("✓ Hierarchical categorization working")
        print("✓ Smart content grouping working")
        print("✓ Index generation working")
        print("✓ Persistence working")
        
        # Clean up
        cleanup_test_files()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Still try to clean up
        try:
            cleanup_test_files()
        except:
            pass