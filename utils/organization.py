"""
Advanced organization system for educational PDF collection.
Implements hierarchical categorization, automatic folder structure generation,
and subject taxonomy management.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from data.models import PDFMetadata, AcademicLevel


class SubjectArea(Enum):
    """Main subject areas for educational content."""
    MATHEMATICS = "mathematics"
    COMPUTER_SCIENCE = "computer_science"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGINEERING = "engineering"
    BUSINESS = "business"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    HISTORY = "history"
    LITERATURE = "literature"
    PHILOSOPHY = "philosophy"
    MEDICINE = "medicine"
    LAW = "law"
    EDUCATION = "education"
    ARTS = "arts"
    LANGUAGES = "languages"
    SOCIAL_SCIENCES = "social_sciences"
    ENVIRONMENTAL_SCIENCE = "environmental_science"
    STATISTICS = "statistics"
    OTHER = "other"


@dataclass
class TopicCategory:
    """Represents a topic category within a subject area."""
    name: str
    keywords: List[str] = field(default_factory=list)
    subcategories: List[str] = field(default_factory=list)
    academic_levels: List[AcademicLevel] = field(default_factory=list)
    description: str = ""
    
    def matches_content(self, metadata: PDFMetadata) -> float:
        """Calculate how well this category matches the given content."""
        score = 0.0
        total_keywords = len(self.keywords)
        
        if total_keywords == 0:
            return 0.0
        
        # Check title and keywords
        content_text = f"{metadata.title} {' '.join(metadata.keywords)} {metadata.description}".lower()
        
        matched_keywords = 0
        for keyword in self.keywords:
            if keyword.lower() in content_text:
                matched_keywords += 1
        
        score = matched_keywords / total_keywords
        
        # Boost score if academic level matches
        if metadata.academic_level in self.academic_levels:
            score *= 1.2
        
        return min(score, 1.0)


@dataclass
class SubjectTaxonomy:
    """Complete taxonomy for a subject area."""
    subject: SubjectArea
    topics: Dict[str, TopicCategory] = field(default_factory=dict)
    folder_structure: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_topic(self, topic: TopicCategory):
        """Add a topic category to this taxonomy."""
        self.topics[topic.name] = topic
    
    def classify_content(self, metadata: PDFMetadata) -> Tuple[str, float]:
        """Classify content into the best matching topic."""
        best_topic = "general"
        best_score = 0.0
        
        for topic_name, topic in self.topics.items():
            score = topic.matches_content(metadata)
            if score > best_score:
                best_score = score
                best_topic = topic_name
        
        return best_topic, best_score
    
    def get_folder_path(self, topic: str, academic_level: AcademicLevel) -> Path:
        """Generate folder path for given topic and academic level."""
        base_path = Path(self.subject.value)
        level_path = base_path / academic_level.value
        topic_path = level_path / topic
        return topic_path


class HierarchicalCategorizer:
    """
    Main categorization system that implements hierarchical organization
    with automatic folder structure generation and subject taxonomy.
    """
    
    def __init__(self, base_output_dir: str = "EducationalPDFs"):
        self.base_output_dir = Path(base_output_dir)
        self.taxonomies: Dict[SubjectArea, SubjectTaxonomy] = {}
        self.folder_stats: Dict[str, int] = {}
        
        # Initialize default taxonomies
        self._initialize_taxonomies()
    
    def _initialize_taxonomies(self):
        """Initialize default subject taxonomies with topics and keywords."""
        
        # Mathematics taxonomy
        math_taxonomy = SubjectTaxonomy(SubjectArea.MATHEMATICS)
        math_taxonomy.add_topic(TopicCategory(
            name="calculus",
            keywords=["calculus", "derivative", "integral", "limit", "differential", "multivariable"],
            academic_levels=[AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE]
        ))
        math_taxonomy.add_topic(TopicCategory(
            name="algebra",
            keywords=["algebra", "linear", "matrix", "vector", "polynomial", "equation"],
            academic_levels=[AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE]
        ))
        math_taxonomy.add_topic(TopicCategory(
            name="statistics",
            keywords=["statistics", "probability", "distribution", "hypothesis", "regression", "analysis"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        math_taxonomy.add_topic(TopicCategory(
            name="discrete_math",
            keywords=["discrete", "combinatorics", "graph theory", "logic", "proof", "set theory"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        self.taxonomies[SubjectArea.MATHEMATICS] = math_taxonomy
        
        # Computer Science taxonomy
        cs_taxonomy = SubjectTaxonomy(SubjectArea.COMPUTER_SCIENCE)
        cs_taxonomy.add_topic(TopicCategory(
            name="programming",
            keywords=["programming", "coding", "python", "java", "javascript", "algorithm", "data structure"],
            academic_levels=[AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE]
        ))
        cs_taxonomy.add_topic(TopicCategory(
            name="machine_learning",
            keywords=["machine learning", "neural network", "deep learning", "ai", "artificial intelligence"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        cs_taxonomy.add_topic(TopicCategory(
            name="databases",
            keywords=["database", "sql", "nosql", "data modeling", "query", "relational"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        cs_taxonomy.add_topic(TopicCategory(
            name="software_engineering",
            keywords=["software engineering", "design patterns", "architecture", "testing", "agile"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        cs_taxonomy.add_topic(TopicCategory(
            name="cybersecurity",
            keywords=["security", "cryptography", "encryption", "cybersecurity", "network security"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        self.taxonomies[SubjectArea.COMPUTER_SCIENCE] = cs_taxonomy
        
        # Physics taxonomy
        physics_taxonomy = SubjectTaxonomy(SubjectArea.PHYSICS)
        physics_taxonomy.add_topic(TopicCategory(
            name="mechanics",
            keywords=["mechanics", "motion", "force", "energy", "momentum", "kinematics"],
            academic_levels=[AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE]
        ))
        physics_taxonomy.add_topic(TopicCategory(
            name="electromagnetism",
            keywords=["electromagnetic", "electric", "magnetic", "field", "wave", "maxwell"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        physics_taxonomy.add_topic(TopicCategory(
            name="quantum_physics",
            keywords=["quantum", "particle", "wave function", "uncertainty", "schrodinger"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        physics_taxonomy.add_topic(TopicCategory(
            name="thermodynamics",
            keywords=["thermodynamics", "heat", "temperature", "entropy", "gas laws"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        self.taxonomies[SubjectArea.PHYSICS] = physics_taxonomy
        
        # Add more taxonomies for other subjects
        self._add_chemistry_taxonomy()
        self._add_biology_taxonomy()
        self._add_engineering_taxonomy()
        self._add_business_taxonomy()
    
    def _add_chemistry_taxonomy(self):
        """Add chemistry subject taxonomy."""
        chem_taxonomy = SubjectTaxonomy(SubjectArea.CHEMISTRY)
        chem_taxonomy.add_topic(TopicCategory(
            name="organic_chemistry",
            keywords=["organic", "carbon", "hydrocarbon", "functional group", "synthesis", "reaction mechanism"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        chem_taxonomy.add_topic(TopicCategory(
            name="inorganic_chemistry",
            keywords=["inorganic", "metal", "coordination", "crystal", "solid state"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        chem_taxonomy.add_topic(TopicCategory(
            name="physical_chemistry",
            keywords=["physical chemistry", "thermochemistry", "kinetics", "spectroscopy", "quantum chemistry"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        self.taxonomies[SubjectArea.CHEMISTRY] = chem_taxonomy
    
    def _add_biology_taxonomy(self):
        """Add biology subject taxonomy."""
        bio_taxonomy = SubjectTaxonomy(SubjectArea.BIOLOGY)
        bio_taxonomy.add_topic(TopicCategory(
            name="molecular_biology",
            keywords=["molecular", "dna", "rna", "protein", "gene", "genetics"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        bio_taxonomy.add_topic(TopicCategory(
            name="ecology",
            keywords=["ecology", "ecosystem", "environment", "biodiversity", "conservation"],
            academic_levels=[AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE]
        ))
        bio_taxonomy.add_topic(TopicCategory(
            name="cell_biology",
            keywords=["cell", "cellular", "membrane", "organelle", "mitosis", "meiosis"],
            academic_levels=[AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE]
        ))
        self.taxonomies[SubjectArea.BIOLOGY] = bio_taxonomy
    
    def _add_engineering_taxonomy(self):
        """Add engineering subject taxonomy."""
        eng_taxonomy = SubjectTaxonomy(SubjectArea.ENGINEERING)
        eng_taxonomy.add_topic(TopicCategory(
            name="mechanical_engineering",
            keywords=["mechanical", "thermodynamics", "fluid mechanics", "materials", "design"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        eng_taxonomy.add_topic(TopicCategory(
            name="electrical_engineering",
            keywords=["electrical", "circuit", "electronics", "signal processing", "control systems"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        eng_taxonomy.add_topic(TopicCategory(
            name="civil_engineering",
            keywords=["civil", "structural", "construction", "infrastructure", "concrete", "steel"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        self.taxonomies[SubjectArea.ENGINEERING] = eng_taxonomy
    
    def _add_business_taxonomy(self):
        """Add business subject taxonomy."""
        biz_taxonomy = SubjectTaxonomy(SubjectArea.BUSINESS)
        biz_taxonomy.add_topic(TopicCategory(
            name="management",
            keywords=["management", "leadership", "strategy", "organization", "planning"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        biz_taxonomy.add_topic(TopicCategory(
            name="finance",
            keywords=["finance", "investment", "accounting", "financial", "money", "capital"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        biz_taxonomy.add_topic(TopicCategory(
            name="marketing",
            keywords=["marketing", "advertising", "brand", "consumer", "market research"],
            academic_levels=[AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]
        ))
        self.taxonomies[SubjectArea.BUSINESS] = biz_taxonomy
    
    def classify_pdf(self, metadata: PDFMetadata) -> Tuple[SubjectArea, str, float]:
        """
        Classify a PDF into subject area and topic.
        
        Returns:
            Tuple of (subject_area, topic, confidence_score)
        """
        best_subject = SubjectArea.OTHER
        best_topic = "general"
        best_score = 0.0
        
        # First, determine subject area based on metadata
        subject_area = self._determine_subject_area(metadata)
        
        # Then classify within that subject area
        if subject_area in self.taxonomies:
            topic, score = self.taxonomies[subject_area].classify_content(metadata)
            if score > 0.3:  # Minimum confidence threshold
                return subject_area, topic, score
        
        # Fallback: try all taxonomies
        for subject, taxonomy in self.taxonomies.items():
            topic, score = taxonomy.classify_content(metadata)
            if score > best_score:
                best_score = score
                best_subject = subject
                best_topic = topic
        
        return best_subject, best_topic, best_score
    
    def _determine_subject_area(self, metadata: PDFMetadata) -> SubjectArea:
        """Determine the primary subject area from metadata."""
        subject_keywords = {
            SubjectArea.MATHEMATICS: ["math", "mathematics", "calculus", "algebra", "geometry", "statistics"],
            SubjectArea.COMPUTER_SCIENCE: ["computer", "programming", "software", "algorithm", "data structure"],
            SubjectArea.PHYSICS: ["physics", "quantum", "mechanics", "electromagnetic", "thermodynamics"],
            SubjectArea.CHEMISTRY: ["chemistry", "chemical", "organic", "inorganic", "molecular"],
            SubjectArea.BIOLOGY: ["biology", "biological", "genetics", "ecology", "molecular biology"],
            SubjectArea.ENGINEERING: ["engineering", "mechanical", "electrical", "civil", "design"],
            SubjectArea.BUSINESS: ["business", "management", "finance", "marketing", "economics"],
        }
        
        content_text = f"{metadata.title} {metadata.subject_area} {' '.join(metadata.keywords)}".lower()
        
        # Check explicit subject area first
        if metadata.subject_area:
            for subject, keywords in subject_keywords.items():
                if any(keyword in metadata.subject_area.lower() for keyword in keywords):
                    return subject
        
        # Check content for subject indicators
        best_subject = SubjectArea.OTHER
        best_count = 0
        
        for subject, keywords in subject_keywords.items():
            count = sum(1 for keyword in keywords if keyword in content_text)
            if count > best_count:
                best_count = count
                best_subject = subject
        
        return best_subject
    
    def generate_folder_structure(self, metadata: PDFMetadata) -> Path:
        """
        Generate hierarchical folder structure for a PDF.
        
        Structure: base_dir/subject/academic_level/topic/
        """
        subject, topic, confidence = self.classify_pdf(metadata)
        
        # Create hierarchical path
        folder_path = self.base_output_dir / subject.value / metadata.academic_level.value / topic
        
        # Update folder statistics
        folder_key = str(folder_path)
        self.folder_stats[folder_key] = self.folder_stats.get(folder_key, 0) + 1
        
        return folder_path
    
    def create_folder_structure(self, folder_path: Path) -> bool:
        """Create the folder structure if it doesn't exist."""
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating folder structure {folder_path}: {e}")
            return False
    
    def get_folder_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about folder usage and content distribution."""
        stats = {
            "by_subject": {},
            "by_level": {},
            "by_topic": {},
            "total_folders": len(self.folder_stats)
        }
        
        for folder_path, count in self.folder_stats.items():
            path_parts = Path(folder_path).parts
            
            if len(path_parts) >= 2:
                subject = path_parts[-3] if len(path_parts) >= 3 else "unknown"
                level = path_parts[-2] if len(path_parts) >= 2 else "unknown"
                topic = path_parts[-1] if len(path_parts) >= 1 else "unknown"
                
                stats["by_subject"][subject] = stats["by_subject"].get(subject, 0) + count
                stats["by_level"][level] = stats["by_level"].get(level, 0) + count
                stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + count
        
        return stats
    
    def add_custom_taxonomy(self, subject: SubjectArea, taxonomy: SubjectTaxonomy):
        """Add or update a custom taxonomy for a subject area."""
        self.taxonomies[subject] = taxonomy
    
    def add_custom_topic(self, subject: SubjectArea, topic: TopicCategory):
        """Add a custom topic to an existing taxonomy."""
        if subject in self.taxonomies:
            self.taxonomies[subject].add_topic(topic)
        else:
            # Create new taxonomy with this topic
            new_taxonomy = SubjectTaxonomy(subject)
            new_taxonomy.add_topic(topic)
            self.taxonomies[subject] = new_taxonomy
    
    def save_taxonomies(self, filepath: str = "config/taxonomies.json"):
        """Save current taxonomies to file for persistence."""
        taxonomies_data = {}
        
        for subject, taxonomy in self.taxonomies.items():
            topics_data = {}
            for topic_name, topic in taxonomy.topics.items():
                topics_data[topic_name] = {
                    "name": topic.name,
                    "keywords": topic.keywords,
                    "subcategories": topic.subcategories,
                    "academic_levels": [level.value for level in topic.academic_levels],
                    "description": topic.description
                }
            
            taxonomies_data[subject.value] = {
                "subject": subject.value,
                "topics": topics_data,
                "folder_structure": taxonomy.folder_structure
            }
        
        # Create directory only if filepath has a directory component
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(taxonomies_data, f, indent=2, ensure_ascii=False)
    
    def load_taxonomies(self, filepath: str = "config/taxonomies.json"):
        """Load taxonomies from file."""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                taxonomies_data = json.load(f)
            
            for subject_name, taxonomy_data in taxonomies_data.items():
                subject = SubjectArea(subject_name)
                taxonomy = SubjectTaxonomy(subject)
                
                for topic_name, topic_data in taxonomy_data["topics"].items():
                    topic = TopicCategory(
                        name=topic_data["name"],
                        keywords=topic_data["keywords"],
                        subcategories=topic_data["subcategories"],
                        academic_levels=[AcademicLevel(level) for level in topic_data["academic_levels"]],
                        description=topic_data["description"]
                    )
                    taxonomy.add_topic(topic)
                
                taxonomy.folder_structure = taxonomy_data.get("folder_structure", {})
                self.taxonomies[subject] = taxonomy
                
        except Exception as e:
            print(f"Error loading taxonomies from {filepath}: {e}")