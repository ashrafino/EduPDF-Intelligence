#!/usr/bin/env python3
"""
Simple demonstration of the integrated advanced organization system.
Shows how the organization system works with the existing scraper infrastructure.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from data.models import PDFMetadata, AcademicLevel
from utils.advanced_organization import AdvancedOrganizationSystem
from utils.logging_setup import setup_logging
from config.settings import config_manager


def create_sample_collection() -> list[PDFMetadata]:
    """Create a sample collection that simulates scraped PDFs."""
    return [
        PDFMetadata(
            filename="intro_programming.pdf",
            file_path="/scraped/intro_programming.pdf",
            file_size=2500000,
            title="Introduction to Programming with Python",
            authors=["Prof. John Smith"],
            institution="Tech University",
            subject_area="computer_science",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["programming", "python", "introduction", "coding"],
            page_count=150,
            text_ratio=0.85,
            quality_score=0.9,
            description="Comprehensive introduction to programming concepts"
        ),
        
        PDFMetadata(
            filename="calculus_textbook.pdf",
            file_path="/scraped/calculus_textbook.pdf",
            file_size=8000000,
            title="Calculus: Theory and Applications",
            authors=["Dr. Maria Garcia"],
            institution="Mathematics Institute",
            subject_area="mathematics",
            academic_level=AcademicLevel.UNDERGRADUATE,
            keywords=["calculus", "derivatives", "integrals", "mathematics"],
            page_count=450,
            text_ratio=0.92,
            quality_score=0.95,
            description="Advanced calculus textbook with practical applications"
        ),
        
        PDFMetadata(
            filename="quantum_physics.pdf",
            file_path="/scraped/quantum_physics.pdf",
            file_size=6500000,
            title="Principles of Quantum Physics",
            authors=["Prof. David Einstein"],
            institution="Physics Research Center",
            subject_area="physics",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["quantum", "physics", "mechanics", "wave function"],
            page_count=320,
            text_ratio=0.88,
            quality_score=0.93,
            description="Graduate-level quantum physics theory and applications"
        ),
        
        PDFMetadata(
            filename="machine_learning_basics.pdf",
            file_path="/scraped/machine_learning_basics.pdf",
            file_size=4200000,
            title="Machine Learning Fundamentals",
            authors=["Dr. Alice Johnson"],
            institution="AI Research Lab",
            subject_area="computer_science",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["machine learning", "AI", "algorithms", "neural networks"],
            page_count=280,
            text_ratio=0.87,
            quality_score=0.91,
            description="Introduction to machine learning algorithms and techniques"
        ),
        
        PDFMetadata(
            filename="business_strategy.pdf",
            file_path="/scraped/business_strategy.pdf",
            file_size=3800000,
            title="Strategic Management in Modern Business",
            authors=["Prof. Amanda White"],
            institution="Business School",
            subject_area="business",
            academic_level=AcademicLevel.GRADUATE,
            keywords=["business", "strategy", "management", "leadership"],
            page_count=220,
            text_ratio=0.84,
            quality_score=0.89,
            description="Comprehensive guide to strategic business management"
        )
    ]


async def demonstrate_integration():
    """Demonstrate the integrated organization system."""
    print("🎓 Enhanced PDF Scraper with Advanced Organization")
    print("=" * 60)
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # Load configuration
    app_config = config_manager.load_app_config()
    sources = config_manager.load_sources()
    
    logger.info("Enhanced Educational PDF Scraper with Advanced Organization starting...")
    logger.info(f"Configuration loaded: {len(sources)} sources configured")
    logger.info(f"Output directory: {Path(app_config.base_output_dir).absolute()}")
    
    # Create sample collection (simulating scraped PDFs)
    print("\n📚 Simulating PDF scraping results...")
    sample_pdfs = create_sample_collection()
    print(f"   Simulated {len(sample_pdfs)} scraped PDFs")
    
    # Initialize organization system
    print("\n🔧 Initializing Advanced Organization System...")
    org_system = AdvancedOrganizationSystem(app_config.base_output_dir)
    print("   ✅ Hierarchical categorization system loaded")
    print("   ✅ Smart content grouping system loaded")
    print("   ✅ Integration system ready")
    
    # Organize the collection
    print("\n🔄 Organizing PDF collection...")
    organized_collection = org_system.organize_collection(sample_pdfs)
    
    # Get and display results
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
    
    # Show content groups
    print("\n📋 Content Groups Created:")
    for group in organized_collection.content_groups:
        print(f"   🔖 {group.title} ({group.group_type}) - {len(group.pdfs)} PDFs")
    
    # Demonstrate recommendation system
    print("\n🎯 Content Recommendation Demo:")
    sample_pdf = sample_pdfs[0]
    recommendations = org_system.grouper.generate_recommendations(
        sample_pdf, sample_pdfs, max_recommendations=3
    )
    
    print(f"   📄 For: {sample_pdf.title}")
    print("   🔍 Recommendations:")
    for i, (filename, similarity) in enumerate(recommendations, 1):
        print(f"      {i}. {filename} (similarity: {similarity:.3f})")
    
    # Save organization data
    org_data_path = Path(app_config.base_output_dir) / "organization_data.json"
    org_system.save_organization(organized_collection, str(org_data_path))
    print(f"\n💾 Organization data saved to: {org_data_path}")
    
    # Show folder structure
    print(f"\n📁 Organized folder structure created in: {summary['base_path']}")
    print("   Check the directory to see:")
    print("   • Hierarchical subject/level/topic folders")
    print("   • Smart content groups")
    print("   • Comprehensive indexes and documentation")
    print("   • Course material organization")
    print("   • Document series detection")
    
    print("\n🎉 Integration demonstration completed successfully!")
    print("The advanced organization system is fully integrated and ready to use!")
    
    return organized_collection


if __name__ == "__main__":
    asyncio.run(demonstrate_integration())