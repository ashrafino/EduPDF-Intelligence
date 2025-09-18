#!/usr/bin/env python3
"""
Demonstration script for the advanced organization system.
Shows the capabilities of hierarchical categorization and smart content grouping.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from data.models import PDFMetadata, AcademicLevel
from utils.advanced_organization import AdvancedOrganizationSystem
from utils.logging_setup import setup_logging


def create_demo_pdf_collection() -> list[PDFMetadata]:
    """Create a realistic demo PDF collection for demonstration."""
    return [
        # Computer Science Course - CS101 (Complete course)
        PDFMetadata(
            filename="cs101_syllabus.pdf",
            file_path="/demo/cs101_syllabus.pdf",
            file_size=500000,
            title="CS101 Introduction to Programming - Course Syllabus",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "syllabus", "course outline", "cs101", "introduction"],
            course_code="CS101",
            page_count=8,
            text_ratio=0.85,
            quality_score=0.9,
            description="Comprehensive course syllabus for introductory programming"
        ),
        
        PDFMetadata(
            filename="cs101_lecture01_intro.pdf",
            file_path="/demo/cs101_lecture01_intro.pdf",
            file_size=2000000,
            title="CS101 Introduction to Programming - Lecture 1",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "introduction", "variables", "cs101", "lecture"],
            course_code="CS101",
            page_count=25,
            text_ratio=0.8,
            quality_score=0.88
        ),
        
        PDFMetadata(
            filename="cs101_lecture02_control.pdf",
            file_path="/demo/cs101_lecture02_control.pdf",
            file_size=2200000,
            title="CS101 Control Structures and Loops - Lecture 2",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "control structures", "loops", "cs101", "lecture"],
            course_code="CS101",
            page_count=28,
            text_ratio=0.82,
            quality_score=0.89
        ),
        
        PDFMetadata(
            filename="cs101_assignment1.pdf",
            file_path="/demo/cs101_assignment1.pdf",
            file_size=800000,
            title="CS101 Programming Assignment 1: Basic Algorithms",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "assignment", "algorithms", "cs101", "homework"],
            course_code="CS101",
            page_count=6,
            text_ratio=0.75,
            quality_score=0.85
        ),
        
        PDFMetadata(
            filename="cs101_midterm_exam.pdf",
            file_path="/demo/cs101_midterm_exam.pdf",
            file_size=400000,
            title="CS101 Midterm Examination",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "exam", "midterm", "cs101", "test"],
            course_code="CS101",
            page_count=8,
            text_ratio=0.7,
            quality_score=0.8
        ),
        
        # Machine Learning Textbook Series
        PDFMetadata(
            filename="ml_fundamentals_vol1.pdf",
            file_path="/demo/ml_fundamentals_vol1.pdf",
            file_size=8000000,
            title="Machine Learning Fundamentals - Volume 1: Supervised Learning",
            authors=["Dr. Alice Johnson", "Dr. Bob Wilson"],
            institution="AI Research Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "supervised learning", "algorithms", "textbook", "AI"],
            page_count=350,
            text_ratio=0.9,
            quality_score=0.95,
            description="Comprehensive introduction to supervised machine learning"
        ),
        
        PDFMetadata(
            filename="ml_fundamentals_vol2.pdf",
            file_path="/demo/ml_fundamentals_vol2.pdf",
            file_size=8500000,
            title="Machine Learning Fundamentals - Volume 2: Unsupervised Learning",
            authors=["Dr. Alice Johnson", "Dr. Bob Wilson"],
            institution="AI Research Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "unsupervised learning", "clustering", "textbook", "AI"],
            page_count=320,
            text_ratio=0.92,
            quality_score=0.96,
            description="Advanced unsupervised learning techniques and applications"
        ),
        
        PDFMetadata(
            filename="ml_fundamentals_vol3.pdf",
            file_path="/demo/ml_fundamentals_vol3.pdf",
            file_size=7800000,
            title="Machine Learning Fundamentals - Volume 3: Deep Learning",
            authors=["Dr. Alice Johnson", "Dr. Bob Wilson"],
            institution="AI Research Institute",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "deep learning", "neural networks", "textbook", "AI"],
            page_count=380,
            text_ratio=0.88,
            quality_score=0.94,
            description="Deep learning architectures and modern applications"
        ),
        
        # Mathematics Course - Calculus
        PDFMetadata(
            filename="calc_chapter1_limits.pdf",
            file_path="/demo/calc_chapter1_limits.pdf",
            file_size=3000000,
            title="Calculus I - Chapter 1: Limits and Continuity",
            authors=["Prof. Maria Garcia"],
            institution="Mathematics College",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "limits", "continuity", "mathematics", "chapter"],
            course_code="MATH201",
            page_count=45,
            text_ratio=0.9,
            quality_score=0.91
        ),
        
        PDFMetadata(
            filename="calc_chapter2_derivatives.pdf",
            file_path="/demo/calc_chapter2_derivatives.pdf",
            file_size=3200000,
            title="Calculus I - Chapter 2: Derivatives and Applications",
            authors=["Prof. Maria Garcia"],
            institution="Mathematics College",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "derivatives", "differentiation", "mathematics", "chapter"],
            course_code="MATH201",
            page_count=52,
            text_ratio=0.88,
            quality_score=0.93
        ),
        
        PDFMetadata(
            filename="calc_chapter3_integrals.pdf",
            file_path="/demo/calc_chapter3_integrals.pdf",
            file_size=3400000,
            title="Calculus I - Chapter 3: Integration Techniques",
            authors=["Prof. Maria Garcia"],
            institution="Mathematics College",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "integrals", "integration", "mathematics", "chapter"],
            course_code="MATH201",
            page_count=48,
            text_ratio=0.89,
            quality_score=0.92
        ),
        
        # Physics Materials
        PDFMetadata(
            filename="quantum_mechanics_intro.pdf",
            file_path="/demo/quantum_mechanics_intro.pdf",
            file_size=4500000,
            title="Introduction to Quantum Mechanics",
            authors=["Dr. David Einstein"],
            institution="Physics Institute",
            subject_area="physics",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["quantum mechanics", "wave function", "schrodinger", "physics", "quantum"],
            page_count=180,
            text_ratio=0.85,
            quality_score=0.94
        ),
        
        PDFMetadata(
            filename="statistical_mechanics.pdf",
            file_path="/demo/statistical_mechanics.pdf",
            file_size=4200000,
            title="Statistical Mechanics and Thermodynamics",
            authors=["Dr. Sarah Newton"],
            institution="Physics Institute",
            subject_area="physics",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["statistical mechanics", "thermodynamics", "entropy", "physics", "thermal"],
            page_count=165,
            text_ratio=0.87,
            quality_score=0.92
        ),
        
        # Business Strategy Series
        PDFMetadata(
            filename="business_strategy_part1.pdf",
            file_path="/demo/business_strategy_part1.pdf",
            file_size=5500000,
            title="Strategic Management - Part 1: Analysis and Planning",
            authors=["Prof. Michael Porter", "Dr. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business", "strategy", "management", "planning", "analysis"],
            page_count=180,
            text_ratio=0.86,
            quality_score=0.92
        ),
        
        PDFMetadata(
            filename="business_strategy_part2.pdf",
            file_path="/demo/business_strategy_part2.pdf",
            file_size=5800000,
            title="Strategic Management - Part 2: Implementation and Control",
            authors=["Prof. Michael Porter", "Dr. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business", "strategy", "implementation", "control", "management"],
            page_count=195,
            text_ratio=0.88,
            quality_score=0.94
        ),
        
        # Related Research Papers (Same Authors)
        PDFMetadata(
            filename="advanced_algorithms_smith.pdf",
            file_path="/demo/advanced_algorithms_smith.pdf",
            file_size=6000000,
            title="Advanced Algorithms and Data Structures",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["algorithms", "data structures", "complexity", "advanced", "computer science"],
            page_count=350,
            text_ratio=0.88,
            quality_score=0.92
        ),
        
        PDFMetadata(
            filename="software_engineering_smith.pdf",
            file_path="/demo/software_engineering_smith.pdf",
            file_size=5500000,
            title="Software Engineering Principles and Practices",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["software engineering", "design patterns", "testing", "agile", "development"],
            page_count=280,
            text_ratio=0.84,
            quality_score=0.89
        ),
        
        # High School Mathematics
        PDFMetadata(
            filename="algebra2_workbook.pdf",
            file_path="/demo/algebra2_workbook.pdf",
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
        
        # Engineering
        PDFMetadata(
            filename="circuit_analysis.pdf",
            file_path="/demo/circuit_analysis.pdf",
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
        )
    ]


async def run_organization_demo():
    """Run a comprehensive demonstration of the organization system."""
    print("🎓 Advanced PDF Organization System - Live Demo")
    print("=" * 60)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # Initialize organization system
    org_system = AdvancedOrganizationSystem("DemoOrganizedPDFs")
    
    # Create demo collection
    print("📚 Creating demo PDF collection...")
    demo_pdfs = create_demo_pdf_collection()
    print(f"   Created {len(demo_pdfs)} demo PDFs representing a realistic educational library")
    
    # Show collection overview
    print("\n📊 Collection Overview:")
    subjects = {}
    levels = {}
    for pdf in demo_pdfs:
        subject = pdf.subject_area or "unknown"
        level = pdf.academic_level.value
        subjects[subject] = subjects.get(subject, 0) + 1
        levels[level] = levels.get(level, 0) + 1
    
    print("   Subjects:")
    for subject, count in subjects.items():
        print(f"     {subject.replace('_', ' ').title()}: {count} PDFs")
    
    print("   Academic Levels:")
    for level, count in levels.items():
        print(f"     {level.replace('_', ' ').title()}: {count} PDFs")
    
    # Organize the collection
    print("\n🔄 Organizing collection with advanced organization system...")
    print("   Step 1: Applying hierarchical categorization...")
    print("   Step 2: Creating smart content groups...")
    print("   Step 3: Generating folder structure...")
    print("   Step 4: Creating indexes and documentation...")
    
    organized_collection = org_system.organize_collection(demo_pdfs)
    
    # Display results
    summary = org_system.get_organization_summary(organized_collection)
    
    print("\n✅ Organization Complete!")
    print("=" * 40)
    print(f"📁 Total PDFs Organized: {summary['total_pdfs']}")
    print(f"📂 Hierarchical Folders: {summary['hierarchical_folders']}")
    print(f"👥 Content Groups: {summary['content_groups']}")
    print(f"📍 Output Directory: {summary['base_path']}")
    
    print("\n📈 Subject Distribution:")
    for subject, count in summary['subject_distribution'].items():
        print(f"   {subject.replace('_', ' ').title()}: {count} PDFs")
    
    print("\n🎯 Academic Level Distribution:")
    for level, count in summary['level_distribution'].items():
        print(f"   {level.replace('_', ' ').title()}: {count} PDFs")
    
    print("\n🏷️  Content Group Types:")
    for group_type, count in summary['group_types'].items():
        print(f"   {group_type.title()}: {count} groups")
    
    # Show detailed groups
    print("\n📋 Detailed Content Groups:")
    print("-" * 40)
    
    for group in organized_collection.content_groups:
        print(f"\n🔖 {group.title}")
        print(f"   Type: {group.group_type}")
        print(f"   Size: {len(group.pdfs)} PDFs")
        
        if group.metadata:
            print("   Metadata:")
            for key, value in group.metadata.items():
                print(f"     {key}: {value}")
        
        print("   Files:")
        for pdf_file in group.pdfs[:3]:  # Show first 3
            print(f"     • {pdf_file}")
        if len(group.pdfs) > 3:
            print(f"     ... and {len(group.pdfs) - 3} more")
    
    # Show folder structure
    print("\n📁 Generated Folder Structure:")
    print("-" * 40)
    base_path = Path(summary['base_path'])
    if base_path.exists():
        print("DemoOrganizedPDFs/")
        
        # Show main directories
        for item in sorted(base_path.iterdir()):
            if item.is_dir():
                print(f"├── {item.name}/")
                
                # Show subdirectories (limited depth)
                try:
                    subdirs = [d for d in item.iterdir() if d.is_dir()][:3]
                    for i, subdir in enumerate(subdirs):
                        prefix = "│   ├──" if i < len(subdirs) - 1 else "│   └──"
                        print(f"{prefix} {subdir.name}/")
                        
                        # Show files in subdirectory (limited)
                        try:
                            files = [f for f in subdir.iterdir() if f.is_file()][:2]
                            for j, file in enumerate(files):
                                file_prefix = "│   │   ├──" if i < len(subdirs) - 1 else "│       ├──"
                                if j == len(files) - 1 and len(files) < 3:
                                    file_prefix = "│   │   └──" if i < len(subdirs) - 1 else "│       └──"
                                print(f"{file_prefix} {file.name}")
                            
                            if len([f for f in subdir.iterdir() if f.is_file()]) > 2:
                                more_prefix = "│   │   └──" if i < len(subdirs) - 1 else "│       └──"
                                print(f"{more_prefix} ...")
                        except:
                            pass
                except:
                    pass
    
    # Demonstrate recommendation system
    print("\n🎯 Content Recommendation Demo:")
    print("-" * 40)
    
    # Pick a sample PDF for recommendations
    sample_pdf = demo_pdfs[0]  # CS101 syllabus
    print(f"📄 Sample PDF: {sample_pdf.title}")
    
    recommendations = org_system.grouper.generate_recommendations(
        sample_pdf, demo_pdfs, max_recommendations=5
    )
    
    print("🔍 Top Recommendations:")
    for i, (filename, similarity) in enumerate(recommendations, 1):
        print(f"   {i}. {filename} (similarity: {similarity:.3f})")
    
    # Show series detection
    print("\n📚 Document Series Detection:")
    print("-" * 40)
    
    series_info = org_system.grouper.detect_document_series(demo_pdfs)
    for series_name, series in series_info.items():
        print(f"📖 Series: {series_name}")
        print(f"   Total Parts: {series.total_parts}")
        print(f"   Found Parts: {series.identified_parts}")
        print(f"   Complete: {'✅' if series.is_complete() else '❌'}")
        
        if not series.is_complete():
            missing = series.get_missing_parts()
            print(f"   Missing: {missing}")
    
    print("\n🎉 Demo completed successfully!")
    print(f"📂 Check the '{summary['base_path']}' directory to see the organized structure")
    print("📄 Index files and documentation have been generated automatically")
    
    return organized_collection


if __name__ == "__main__":
    asyncio.run(run_organization_demo())