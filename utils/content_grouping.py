"""
Smart content grouping system for educational PDF collection.
Implements related content detection, course material grouping,
series detection, and recommendation system.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import difflib

from data.models import PDFMetadata, AcademicLevel


@dataclass
class ContentGroup:
    """Represents a group of related educational content."""
    group_id: str
    group_type: str  # 'course', 'series', 'topic', 'author'
    title: str
    description: str = ""
    pdfs: List[str] = field(default_factory=list)  # List of PDF filenames
    metadata: Dict[str, any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_pdf(self, pdf_filename: str):
        """Add a PDF to this group."""
        if pdf_filename not in self.pdfs:
            self.pdfs.append(pdf_filename)
            self.last_updated = datetime.now()
    
    def remove_pdf(self, pdf_filename: str):
        """Remove a PDF from this group."""
        if pdf_filename in self.pdfs:
            self.pdfs.remove(pdf_filename)
            self.last_updated = datetime.now()
    
    def get_size(self) -> int:
        """Get the number of PDFs in this group."""
        return len(self.pdfs)


@dataclass
class CourseStructure:
    """Represents the structure of a course with its materials."""
    course_code: str
    course_name: str
    institution: str
    instructor: str = ""
    academic_level: AcademicLevel = AcademicLevel.UNKNOWN
    
    # Course materials organized by type
    lectures: List[str] = field(default_factory=list)
    assignments: List[str] = field(default_factory=list)
    exams: List[str] = field(default_factory=list)
    textbooks: List[str] = field(default_factory=list)
    supplementary: List[str] = field(default_factory=list)
    
    def get_all_materials(self) -> List[str]:
        """Get all course materials as a single list."""
        return (self.lectures + self.assignments + self.exams + 
                self.textbooks + self.supplementary)
    
    def categorize_material(self, pdf_metadata: PDFMetadata) -> str:
        """Categorize a PDF into the appropriate course material type."""
        title_lower = pdf_metadata.title.lower()
        keywords_lower = [k.lower() for k in pdf_metadata.keywords]
        
        # Check for lecture materials
        lecture_indicators = ['lecture', 'slides', 'presentation', 'notes', 'chapter']
        if any(indicator in title_lower for indicator in lecture_indicators):
            return 'lectures'
        
        # Check for assignments
        assignment_indicators = ['assignment', 'homework', 'exercise', 'problem set', 'lab']
        if any(indicator in title_lower for indicator in assignment_indicators):
            return 'assignments'
        
        # Check for exams
        exam_indicators = ['exam', 'test', 'quiz', 'midterm', 'final', 'solution']
        if any(indicator in title_lower for indicator in exam_indicators):
            return 'exams'
        
        # Check for textbooks
        textbook_indicators = ['textbook', 'book', 'manual', 'guide', 'handbook']
        if (any(indicator in title_lower for indicator in textbook_indicators) or
            pdf_metadata.page_count > 200):
            return 'textbooks'
        
        # Default to supplementary
        return 'supplementary'


@dataclass
class SeriesInfo:
    """Information about a document series."""
    series_name: str
    total_parts: int
    identified_parts: List[int] = field(default_factory=list)
    part_filenames: Dict[int, str] = field(default_factory=dict)
    
    def add_part(self, part_number: int, filename: str):
        """Add a part to the series."""
        if part_number not in self.identified_parts:
            self.identified_parts.append(part_number)
            self.identified_parts.sort()
        self.part_filenames[part_number] = filename
    
    def is_complete(self) -> bool:
        """Check if all parts of the series are identified."""
        return len(self.identified_parts) == self.total_parts
    
    def get_missing_parts(self) -> List[int]:
        """Get list of missing part numbers."""
        all_parts = set(range(1, self.total_parts + 1))
        identified_parts = set(self.identified_parts)
        return sorted(list(all_parts - identified_parts))


class SmartContentGrouper:
    """
    Main content grouping system that implements intelligent grouping
    of educational materials based on various criteria.
    """
    
    def __init__(self):
        self.content_groups: Dict[str, ContentGroup] = {}
        self.course_structures: Dict[str, CourseStructure] = {}
        self.series_info: Dict[str, SeriesInfo] = {}
        self.similarity_threshold = 0.7
        self.course_code_patterns = [
            r'[A-Z]{2,4}\s*\d{3,4}[A-Z]?',  # CS101, MATH 2210A
            r'[A-Z]{2,4}-\d{3,4}',          # CS-101
            r'\d{1,2}\.\d{3}',              # 6.001 (MIT style)
        ]
    
    def group_related_content(self, pdf_list: List[PDFMetadata]) -> Dict[str, List[str]]:
        """
        Group PDFs by content similarity and relationships.
        
        Returns:
            Dictionary mapping group names to lists of PDF filenames
        """
        # Group by various criteria
        author_groups = self._group_by_author(pdf_list)
        topic_groups = self._group_by_topic_similarity(pdf_list)
        institution_groups = self._group_by_institution(pdf_list)
        
        # Merge and optimize groups
        merged_groups = self._merge_overlapping_groups([
            author_groups, topic_groups, institution_groups
        ])
        
        return merged_groups
    
    def _group_by_author(self, pdf_list: List[PDFMetadata]) -> Dict[str, List[str]]:
        """Group PDFs by author."""
        author_groups = defaultdict(list)
        
        for pdf in pdf_list:
            if pdf.authors:
                # Use primary author (first in list)
                primary_author = pdf.authors[0]
                author_groups[f"author_{primary_author}"].append(pdf.filename)
        
        # Filter out single-item groups
        return {k: v for k, v in author_groups.items() if len(v) > 1}
    
    def _group_by_topic_similarity(self, pdf_list: List[PDFMetadata]) -> Dict[str, List[str]]:
        """Group PDFs by topic similarity using keywords and titles."""
        topic_groups = defaultdict(list)
        
        # Create keyword frequency map
        all_keywords = []
        for pdf in pdf_list:
            all_keywords.extend(pdf.keywords)
        
        keyword_freq = Counter(all_keywords)
        common_keywords = [k for k, v in keyword_freq.most_common(20) if v > 1]
        
        # Group by common keywords
        for keyword in common_keywords:
            group_pdfs = []
            for pdf in pdf_list:
                if keyword in pdf.keywords:
                    group_pdfs.append(pdf.filename)
            
            if len(group_pdfs) > 1:
                topic_groups[f"topic_{keyword}"] = group_pdfs
        
        return dict(topic_groups)
    
    def _group_by_institution(self, pdf_list: List[PDFMetadata]) -> Dict[str, List[str]]:
        """Group PDFs by institution."""
        institution_groups = defaultdict(list)
        
        for pdf in pdf_list:
            if pdf.institution:
                institution_groups[f"institution_{pdf.institution}"].append(pdf.filename)
        
        # Filter out single-item groups
        return {k: v for k, v in institution_groups.items() if len(v) > 1}
    
    def _merge_overlapping_groups(self, group_lists: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """Merge overlapping groups from different grouping methods."""
        all_groups = {}
        
        for group_dict in group_lists:
            for group_name, pdf_list in group_dict.items():
                # Check for overlap with existing groups
                merged = False
                for existing_name, existing_list in all_groups.items():
                    overlap = set(pdf_list) & set(existing_list)
                    if len(overlap) > 0:
                        # Merge groups if there's significant overlap
                        overlap_ratio = len(overlap) / min(len(pdf_list), len(existing_list))
                        if overlap_ratio > 0.5:
                            all_groups[existing_name] = list(set(existing_list + pdf_list))
                            merged = True
                            break
                
                if not merged:
                    all_groups[group_name] = pdf_list
        
        return all_groups
    
    def detect_course_materials(self, pdf_list: List[PDFMetadata]) -> Dict[str, CourseStructure]:
        """
        Detect and group course materials by analyzing course codes and patterns.
        """
        course_materials = defaultdict(list)
        
        # Extract course codes from titles and metadata
        for pdf in pdf_list:
            course_codes = self._extract_course_codes(pdf)
            
            for course_code in course_codes:
                course_materials[course_code].append(pdf)
        
        # Build course structures
        courses = {}
        for course_code, pdfs in course_materials.items():
            if len(pdfs) > 1:  # Only create course if multiple materials found
                course = self._build_course_structure(course_code, pdfs)
                courses[course_code] = course
        
        return courses
    
    def _extract_course_codes(self, pdf: PDFMetadata) -> List[str]:
        """Extract course codes from PDF metadata."""
        course_codes = []
        
        # Check explicit course_code field
        if pdf.course_code:
            course_codes.append(pdf.course_code)
        
        # Search in title and description
        text_to_search = f"{pdf.title} {pdf.description}"
        
        for pattern in self.course_code_patterns:
            matches = re.findall(pattern, text_to_search, re.IGNORECASE)
            course_codes.extend(matches)
        
        # Clean and normalize course codes
        normalized_codes = []
        for code in course_codes:
            normalized = re.sub(r'\s+', '', code.upper())
            if normalized not in normalized_codes:
                normalized_codes.append(normalized)
        
        return normalized_codes
    
    def _build_course_structure(self, course_code: str, pdfs: List[PDFMetadata]) -> CourseStructure:
        """Build a course structure from a list of PDFs."""
        # Determine course name and institution from most common values
        institutions = [pdf.institution for pdf in pdfs if pdf.institution]
        institution = max(set(institutions), key=institutions.count) if institutions else ""
        
        # Extract course name from titles
        course_name = self._extract_course_name(pdfs)
        
        # Determine academic level
        levels = [pdf.academic_level for pdf in pdfs if pdf.academic_level != AcademicLevel.UNKNOWN]
        academic_level = max(set(levels), key=levels.count) if levels else AcademicLevel.UNKNOWN
        
        # Create course structure
        course = CourseStructure(
            course_code=course_code,
            course_name=course_name,
            institution=institution,
            academic_level=academic_level
        )
        
        # Categorize materials
        for pdf in pdfs:
            material_type = course.categorize_material(pdf)
            getattr(course, material_type).append(pdf.filename)
        
        return course
    
    def _extract_course_name(self, pdfs: List[PDFMetadata]) -> str:
        """Extract course name from PDF titles."""
        titles = [pdf.title for pdf in pdfs]
        
        # Find common words in titles
        if not titles:
            return "Unknown Course"
        
        # Split titles into words and find common patterns
        word_sets = [set(title.lower().split()) for title in titles]
        common_words = set.intersection(*word_sets) if word_sets else set()
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = common_words - stop_words
        
        if meaningful_words:
            return ' '.join(sorted(meaningful_words)).title()
        else:
            # Fallback: use first title
            return titles[0]
    
    def detect_document_series(self, pdf_list: List[PDFMetadata]) -> Dict[str, SeriesInfo]:
        """
        Detect document series (multi-part materials) using pattern matching.
        """
        series_candidates = defaultdict(list)
        
        # Patterns for series detection
        series_patterns = [
            r'(.+?)\s+(?:part|volume|chapter|section)\s+(\d+)',
            r'(.+?)\s+(\d+)\s*(?:of\s*(\d+))?',
            r'(.+?)\s*-\s*(\d+)',
            r'(.+?)\s*\(\s*(\d+)\s*\)',
        ]
        
        for pdf in pdf_list:
            title = pdf.title.lower()
            
            for pattern in series_patterns:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    series_name = match.group(1).strip()
                    part_number = int(match.group(2))
                    
                    # Normalize series name
                    series_name = re.sub(r'\s+', ' ', series_name).strip()
                    
                    series_candidates[series_name].append((part_number, pdf.filename, pdf))
        
        # Build series info
        series_info = {}
        for series_name, parts in series_candidates.items():
            if len(parts) > 1:  # Only consider if multiple parts found
                part_numbers = [part[0] for part in parts]
                max_part = max(part_numbers)
                
                series = SeriesInfo(
                    series_name=series_name,
                    total_parts=max_part  # Estimate based on highest number found
                )
                
                for part_number, filename, pdf in parts:
                    series.add_part(part_number, filename)
                
                series_info[series_name] = series
        
        return series_info
    
    def generate_recommendations(self, target_pdf: PDFMetadata, 
                               pdf_collection: List[PDFMetadata], 
                               max_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Generate content recommendations based on similarity to target PDF.
        
        Returns:
            List of tuples (filename, similarity_score)
        """
        recommendations = []
        
        for pdf in pdf_collection:
            if pdf.filename == target_pdf.filename:
                continue
            
            similarity = self._calculate_content_similarity(target_pdf, pdf)
            if similarity > 0.1:  # Minimum similarity threshold
                recommendations.append((pdf.filename, similarity))
        
        # Sort by similarity score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:max_recommendations]
    
    def _calculate_content_similarity(self, pdf1: PDFMetadata, pdf2: PDFMetadata) -> float:
        """Calculate similarity score between two PDFs."""
        similarity_score = 0.0
        
        # Title similarity (30% weight)
        title_similarity = difflib.SequenceMatcher(None, 
                                                 pdf1.title.lower(), 
                                                 pdf2.title.lower()).ratio()
        similarity_score += title_similarity * 0.3
        
        # Keyword overlap (25% weight)
        keywords1 = set(k.lower() for k in pdf1.keywords)
        keywords2 = set(k.lower() for k in pdf2.keywords)
        if keywords1 and keywords2:
            keyword_overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            similarity_score += keyword_overlap * 0.25
        
        # Author similarity (20% weight)
        authors1 = set(a.lower() for a in pdf1.authors)
        authors2 = set(a.lower() for a in pdf2.authors)
        if authors1 and authors2:
            author_overlap = len(authors1 & authors2) / len(authors1 | authors2)
            similarity_score += author_overlap * 0.2
        
        # Subject area match (15% weight)
        if pdf1.subject_area and pdf2.subject_area:
            if pdf1.subject_area.lower() == pdf2.subject_area.lower():
                similarity_score += 0.15
        
        # Academic level match (10% weight)
        if pdf1.academic_level == pdf2.academic_level:
            similarity_score += 0.1
        
        return min(similarity_score, 1.0)
    
    def create_content_groups(self, pdf_list: List[PDFMetadata]) -> List[ContentGroup]:
        """
        Create content groups from PDF collection using all grouping methods.
        """
        groups = []
        
        # Related content groups
        related_groups = self.group_related_content(pdf_list)
        for group_name, pdf_filenames in related_groups.items():
            group = ContentGroup(
                group_id=f"related_{len(groups)}",
                group_type="related",
                title=group_name.replace('_', ' ').title(),
                pdfs=pdf_filenames
            )
            groups.append(group)
        
        # Course material groups
        course_structures = self.detect_course_materials(pdf_list)
        for course_code, course_structure in course_structures.items():
            group = ContentGroup(
                group_id=f"course_{course_code}",
                group_type="course",
                title=f"{course_code}: {course_structure.course_name}",
                pdfs=course_structure.get_all_materials(),
                metadata={
                    "course_code": course_code,
                    "institution": course_structure.institution,
                    "academic_level": course_structure.academic_level.value,
                    "instructor": course_structure.instructor
                }
            )
            groups.append(group)
        
        # Series groups
        series_info = self.detect_document_series(pdf_list)
        for series_name, series in series_info.items():
            group = ContentGroup(
                group_id=f"series_{len(groups)}",
                group_type="series",
                title=f"Series: {series_name}",
                pdfs=list(series.part_filenames.values()),
                metadata={
                    "total_parts": series.total_parts,
                    "identified_parts": series.identified_parts,
                    "is_complete": series.is_complete()
                }
            )
            groups.append(group)
        
        return groups
    
    def save_groups(self, groups: List[ContentGroup], filepath: str = "content_groups.json"):
        """Save content groups to file."""
        groups_data = []
        
        for group in groups:
            group_data = {
                "group_id": group.group_id,
                "group_type": group.group_type,
                "title": group.title,
                "description": group.description,
                "pdfs": group.pdfs,
                "metadata": group.metadata,
                "created_date": group.created_date.isoformat(),
                "last_updated": group.last_updated.isoformat()
            }
            groups_data.append(group_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(groups_data, f, indent=2, ensure_ascii=False)
    
    def load_groups(self, filepath: str = "content_groups.json") -> List[ContentGroup]:
        """Load content groups from file."""
        if not Path(filepath).exists():
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                groups_data = json.load(f)
            
            groups = []
            for group_data in groups_data:
                group = ContentGroup(
                    group_id=group_data["group_id"],
                    group_type=group_data["group_type"],
                    title=group_data["title"],
                    description=group_data["description"],
                    pdfs=group_data["pdfs"],
                    metadata=group_data["metadata"],
                    created_date=datetime.fromisoformat(group_data["created_date"]),
                    last_updated=datetime.fromisoformat(group_data["last_updated"])
                )
                groups.append(group)
            
            return groups
            
        except Exception as e:
            print(f"Error loading groups from {filepath}: {e}")
            return []
    
    def get_group_statistics(self, groups: List[ContentGroup]) -> Dict[str, any]:
        """Get statistics about content groups."""
        stats = {
            "total_groups": len(groups),
            "by_type": defaultdict(int),
            "by_size": defaultdict(int),
            "total_pdfs_grouped": 0,
            "average_group_size": 0
        }
        
        total_pdfs = 0
        for group in groups:
            stats["by_type"][group.group_type] += 1
            group_size = group.get_size()
            stats["by_size"][f"{group_size}_pdfs"] += 1
            total_pdfs += group_size
        
        stats["total_pdfs_grouped"] = total_pdfs
        if groups:
            stats["average_group_size"] = total_pdfs / len(groups)
        
        return dict(stats)