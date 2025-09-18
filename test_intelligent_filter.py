"""
Test script for the intelligent content filtering system.
Tests educational relevance filtering and academic level classification.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.models import PDFMetadata, AcademicLevel
from filters.intelligent_filter import IntelligentContentFilter


def create_test_metadata(filename: str, title: str, institution: str = "", 
                        authors: list = None, course_code: str = "") -> PDFMetadata:
    """Create test PDF metadata."""
    return PDFMetadata(
        filename=filename,
        file_path=f"/test/{filename}",
        file_size=1024000,  # 1MB
        title=title,
        authors=authors or [],
        institution=institution,
        course_code=course_code,
        page_count=20,
        text_ratio=0.8,  # 80% text content
        created_date=datetime.now()
    )


def test_educational_relevance_filter():
    """Test the educational relevance filtering."""
    print("=== Testing Educational Relevance Filter ===")
    
    # Create test cases
    test_cases = [
        # Educational PDFs
        create_test_metadata(
            "calculus_textbook.pdf",
            "Introduction to Calculus - Chapter 1: Limits and Derivatives",
            "MIT",
            ["Dr. John Smith", "Prof. Jane Doe"]
        ),
        create_test_metadata(
            "cs101_notes.pdf", 
            "CS101 Lecture Notes: Programming Fundamentals",
            "Stanford University",
            ["Prof. Alice Johnson"],
            "CS101"
        ),
        create_test_metadata(
            "biology_lab.pdf",
            "Biology Lab Manual: Cell Structure and Function",
            "Harvard University",
            ["Dr. Bob Wilson"]
        ),
        
        # Non-educational PDFs
        create_test_metadata(
            "recipe_book.pdf",
            "Delicious Recipes for Every Occasion",
            "",
            ["Chef Mary"]
        ),
        create_test_metadata(
            "marketing_brochure.pdf",
            "Our Amazing Product Features",
            "TechCorp Inc",
            ["Marketing Team"]
        )
    ]
    
    # Test content texts
    content_texts = {
        "calculus_textbook.pdf": """
        This textbook provides a comprehensive introduction to calculus for undergraduate students.
        Chapter 1 covers limits and derivatives with detailed examples and exercises.
        Students will learn fundamental concepts including continuity, differentiation rules,
        and applications of derivatives in physics and engineering.
        """,
        "cs101_notes.pdf": """
        Course: Computer Science 101 - Programming Fundamentals
        Instructor: Prof. Alice Johnson
        These lecture notes cover basic programming concepts including variables,
        control structures, functions, and object-oriented programming principles.
        Assignment 1 is due next week. Quiz on Friday covers chapters 1-3.
        """,
        "biology_lab.pdf": """
        Laboratory Manual for Biology 201
        Experiment 3: Observing Cell Structure Under Microscope
        Objective: Students will examine plant and animal cells to identify
        key cellular components including nucleus, mitochondria, and cell membrane.
        """,
        "recipe_book.pdf": """
        Welcome to our collection of delicious recipes!
        Try our famous chocolate chip cookies or savory pasta dishes.
        Perfect for family dinners and special occasions.
        """,
        "marketing_brochure.pdf": """
        Introducing our revolutionary new software solution!
        Increase productivity by 300% with our cutting-edge technology.
        Contact our sales team today for a free demo and special pricing.
        """
    }
    
    # Initialize filter
    filter_system = IntelligentContentFilter()
    
    # Test each PDF
    for metadata in test_cases:
        content_text = content_texts.get(metadata.filename, "")
        result = filter_system.filter_pdf(metadata, content_text)
        
        print(f"\nFile: {metadata.filename}")
        print(f"  Educational: {result.is_educational}")
        print(f"  Relevance Score: {result.relevance_score:.2f}")
        print(f"  Quality Score: {result.quality_score:.2f}")
        print(f"  Academic Level: {result.predicted_level.value}")
        print(f"  Classification Confidence: {result.classification_confidence:.2f}")
        print(f"  Should Include: {result.should_include}")
        print(f"  Reasons: {', '.join(result.filter_reasons[:2])}")  # Show first 2 reasons
    
    # Print statistics
    print(f"\n=== Statistics ===")
    stats = filter_system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_academic_level_classification():
    """Test academic level classification specifically."""
    print("\n=== Testing Academic Level Classification ===")
    
    test_cases = [
        # High School
        create_test_metadata(
            "algebra_basics.pdf",
            "Algebra I: Basic Equations and Graphing",
            "Lincoln High School",
            ["Ms. Sarah Teacher"],
            "MATH-101"
        ),
        
        # Undergraduate  
        create_test_metadata(
            "organic_chem.pdf",
            "Organic Chemistry: Molecular Structure and Reactions",
            "University of California",
            ["Dr. Chemistry Prof"],
            "CHEM-301"
        ),
        
        # Graduate
        create_test_metadata(
            "research_methods.pdf", 
            "Advanced Research Methodology in Cognitive Psychology",
            "Harvard University",
            ["Prof. PhD Researcher", "Dr. Senior Scientist"],
            "PSYC-701"
        )
    ]
    
    content_texts = {
        "algebra_basics.pdf": """
        This is a basic introduction to algebra for 9th grade students.
        We will learn about solving simple equations like x + 5 = 10.
        Homework assignments will help you practice these fundamental skills.
        Quiz next Friday on chapters 1 and 2.
        """,
        "organic_chem.pdf": """
        This undergraduate course covers advanced organic chemistry concepts.
        Students will analyze complex molecular structures and reaction mechanisms.
        Prerequisites include general chemistry and calculus.
        Laboratory experiments require sophisticated analytical techniques.
        """,
        "research_methods.pdf": """
        This graduate seminar examines sophisticated methodological approaches
        in contemporary cognitive psychology research. Students will critically
        evaluate empirical studies, design original experiments, and conduct
        statistical analyses using advanced computational methods.
        Dissertation proposals are due at semester end.
        """
    }
    
    filter_system = IntelligentContentFilter()
    
    for metadata in test_cases:
        content_text = content_texts.get(metadata.filename, "")
        result = filter_system.filter_pdf(metadata, content_text)
        
        print(f"\nFile: {metadata.filename}")
        print(f"  Predicted Level: {result.predicted_level.value}")
        print(f"  Confidence: {result.classification_confidence:.2f}")
        print(f"  Complexity Score: {result.complexity_score:.2f}")
        print(f"  Readability Score: {result.pdf_metadata.readability_score:.2f}")
        print(f"  Classification Reasons: {', '.join(result.classification_reasons[:2])}")


if __name__ == "__main__":
    try:
        test_educational_relevance_filter()
        test_academic_level_classification()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()