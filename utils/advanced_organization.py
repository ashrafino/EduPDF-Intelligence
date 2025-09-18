"""
Advanced organization system integration module.
Combines hierarchical categorization and smart content grouping
for comprehensive educational PDF organization.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from data.models import PDFMetadata, AcademicLevel
from utils.organization import HierarchicalCategorizer, SubjectArea
from utils.content_grouping import SmartContentGrouper, ContentGroup


@dataclass
class OrganizedCollection:
    """Represents a fully organized PDF collection."""
    base_path: Path
    hierarchical_structure: Dict[str, List[str]] = field(default_factory=dict)
    content_groups: List[ContentGroup] = field(default_factory=list)
    folder_statistics: Dict[str, any] = field(default_factory=dict)
    group_statistics: Dict[str, any] = field(default_factory=dict)
    organization_metadata: Dict[str, any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedOrganizationSystem:
    """
    Comprehensive organization system that combines hierarchical categorization
    with smart content grouping for optimal educational PDF organization.
    """
    
    def __init__(self, base_output_dir: str = "EducationalPDFs"):
        self.base_output_dir = Path(base_output_dir)
        self.categorizer = HierarchicalCategorizer(str(self.base_output_dir))
        self.grouper = SmartContentGrouper()
        
        # Organization settings
        self.create_group_folders = True
        self.create_symbolic_links = True
        self.generate_index_files = True
    
    def organize_collection(self, pdf_list: List[PDFMetadata]) -> OrganizedCollection:
        """
        Perform complete organization of PDF collection using both
        hierarchical categorization and smart content grouping.
        """
        print(f"Organizing collection of {len(pdf_list)} PDFs...")
        
        # Step 1: Hierarchical categorization
        print("Step 1: Applying hierarchical categorization...")
        hierarchical_structure = self._apply_hierarchical_organization(pdf_list)
        
        # Step 2: Smart content grouping
        print("Step 2: Creating smart content groups...")
        content_groups = self.grouper.create_content_groups(pdf_list)
        
        # Step 3: Create integrated folder structure
        print("Step 3: Creating integrated folder structure...")
        self._create_integrated_structure(pdf_list, content_groups)
        
        # Step 4: Generate organization metadata
        print("Step 4: Generating organization metadata...")
        folder_stats = self.categorizer.get_folder_statistics()
        group_stats = self.grouper.get_group_statistics(content_groups)
        
        # Step 5: Create index files and documentation
        if self.generate_index_files:
            print("Step 5: Generating index files...")
            self._generate_index_files(pdf_list, content_groups)
        
        # Create organized collection object
        organized_collection = OrganizedCollection(
            base_path=self.base_output_dir,
            hierarchical_structure=hierarchical_structure,
            content_groups=content_groups,
            folder_statistics=folder_stats,
            group_statistics=group_stats,
            organization_metadata={
                "total_pdfs": len(pdf_list),
                "organization_method": "hierarchical_and_grouped",
                "created_folders": len(hierarchical_structure),
                "created_groups": len(content_groups),
                "settings": {
                    "create_group_folders": self.create_group_folders,
                    "create_symbolic_links": self.create_symbolic_links,
                    "generate_index_files": self.generate_index_files
                }
            }
        )
        
        print("Organization complete!")
        return organized_collection
    
    def _apply_hierarchical_organization(self, pdf_list: List[PDFMetadata]) -> Dict[str, List[str]]:
        """Apply hierarchical categorization to all PDFs."""
        hierarchical_structure = {}
        
        for pdf in pdf_list:
            # Generate folder path
            folder_path = self.categorizer.generate_folder_structure(pdf)
            
            # Create folder structure
            self.categorizer.create_folder_structure(folder_path)
            
            # Track structure
            folder_key = str(folder_path)
            if folder_key not in hierarchical_structure:
                hierarchical_structure[folder_key] = []
            hierarchical_structure[folder_key].append(pdf.filename)
        
        return hierarchical_structure
    
    def _create_integrated_structure(self, pdf_list: List[PDFMetadata], 
                                   content_groups: List[ContentGroup]):
        """Create integrated folder structure combining hierarchy and groups."""
        
        if not self.create_group_folders:
            return
        
        # Ensure base directory exists first
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create groups folder
        groups_base = self.base_output_dir / "Groups"
        groups_base.mkdir(exist_ok=True)
        
        # Create folders for each content group
        for group in content_groups:
            group_folder = groups_base / group.group_type / self._sanitize_filename(group.title)
            group_folder.mkdir(parents=True, exist_ok=True)
            
            # Create group metadata file
            self._create_group_metadata_file(group, group_folder)
            
            # Create symbolic links or copies if enabled
            if self.create_symbolic_links:
                self._create_group_links(group, group_folder, pdf_list)
    
    def _create_group_links(self, group: ContentGroup, group_folder: Path, 
                          pdf_list: List[PDFMetadata]):
        """Create symbolic links to PDFs in group folders."""
        pdf_dict = {pdf.filename: pdf for pdf in pdf_list}
        
        for pdf_filename in group.pdfs:
            if pdf_filename in pdf_dict:
                pdf = pdf_dict[pdf_filename]
                
                # Find the hierarchical location
                hierarchical_path = self.categorizer.generate_folder_structure(pdf)
                source_file = hierarchical_path / pdf_filename
                
                # Create symbolic link in group folder
                link_file = group_folder / pdf_filename
                
                try:
                    if source_file.exists() and not link_file.exists():
                        link_file.symlink_to(source_file.resolve())
                except (OSError, NotImplementedError):
                    # Fallback: create a reference file instead of symlink
                    with open(group_folder / f"{pdf_filename}.ref", 'w') as f:
                        f.write(f"Reference to: {source_file}\n")
                        f.write(f"Original location: {hierarchical_path}\n")
    
    def _create_group_metadata_file(self, group: ContentGroup, group_folder: Path):
        """Create metadata file for content group."""
        metadata = {
            "group_id": group.group_id,
            "group_type": group.group_type,
            "title": group.title,
            "description": group.description,
            "created_date": group.created_date.isoformat(),
            "last_updated": group.last_updated.isoformat(),
            "pdf_count": len(group.pdfs),
            "pdfs": group.pdfs,
            "metadata": group.metadata
        }
        
        metadata_file = group_folder / "group_info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _generate_index_files(self, pdf_list: List[PDFMetadata], 
                            content_groups: List[ContentGroup]):
        """Generate comprehensive index files for the organized collection."""
        
        # Ensure base directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main collection index
        self._create_main_index(pdf_list, content_groups)
        
        # Subject area indexes
        self._create_subject_indexes(pdf_list)
        
        # Group indexes
        self._create_group_indexes(content_groups)
        
        # Course indexes
        self._create_course_indexes(content_groups)
    
    def _create_main_index(self, pdf_list: List[PDFMetadata], 
                          content_groups: List[ContentGroup]):
        """Create main collection index file."""
        index_content = []
        index_content.append("# Educational PDF Collection Index")
        index_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        index_content.append(f"Total PDFs: {len(pdf_list)}")
        index_content.append(f"Total Groups: {len(content_groups)}")
        index_content.append("")
        
        # Statistics
        folder_stats = self.categorizer.get_folder_statistics()
        group_stats = self.grouper.get_group_statistics(content_groups)
        
        index_content.append("## Collection Statistics")
        index_content.append("")
        index_content.append("### By Subject Area")
        for subject, count in folder_stats.get('by_subject', {}).items():
            index_content.append(f"- {subject.replace('_', ' ').title()}: {count} PDFs")
        
        index_content.append("")
        index_content.append("### By Academic Level")
        for level, count in folder_stats.get('by_level', {}).items():
            index_content.append(f"- {level.replace('_', ' ').title()}: {count} PDFs")
        
        index_content.append("")
        index_content.append("### By Content Groups")
        for group_type, count in group_stats.get('by_type', {}).items():
            index_content.append(f"- {group_type.title()}: {count} groups")
        
        # Directory structure
        index_content.append("")
        index_content.append("## Directory Structure")
        index_content.append("")
        index_content.append("```")
        index_content.append("EducationalPDFs/")
        index_content.append("├── Groups/")
        index_content.append("│   ├── course/")
        index_content.append("│   ├── related/")
        index_content.append("│   └── series/")
        
        # Add subject directories
        for subject in folder_stats.get('by_subject', {}).keys():
            index_content.append(f"├── {subject}/")
            index_content.append(f"│   ├── high_school/")
            index_content.append(f"│   ├── undergraduate/")
            index_content.append(f"│   └── graduate/")
        
        index_content.append("└── indexes/")
        index_content.append("```")
        
        # Write index file
        index_file = self.base_output_dir / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(index_content))
    
    def _create_subject_indexes(self, pdf_list: List[PDFMetadata]):
        """Create index files for each subject area."""
        indexes_dir = self.base_output_dir / "indexes"
        indexes_dir.mkdir(exist_ok=True)
        
        # Group PDFs by subject
        subject_pdfs = {}
        for pdf in pdf_list:
            subject, _, _ = self.categorizer.classify_pdf(pdf)
            if subject not in subject_pdfs:
                subject_pdfs[subject] = []
            subject_pdfs[subject].append(pdf)
        
        # Create index for each subject
        for subject, pdfs in subject_pdfs.items():
            index_content = []
            index_content.append(f"# {subject.value.replace('_', ' ').title()} Collection")
            index_content.append(f"Total PDFs: {len(pdfs)}")
            index_content.append("")
            
            # Group by academic level
            level_groups = {}
            for pdf in pdfs:
                level = pdf.academic_level
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(pdf)
            
            for level, level_pdfs in level_groups.items():
                index_content.append(f"## {level.value.replace('_', ' ').title()} Level")
                index_content.append("")
                
                for pdf in sorted(level_pdfs, key=lambda x: x.title):
                    index_content.append(f"- **{pdf.title}**")
                    if pdf.authors:
                        index_content.append(f"  - Authors: {', '.join(pdf.authors)}")
                    if pdf.institution:
                        index_content.append(f"  - Institution: {pdf.institution}")
                    index_content.append(f"  - File: `{pdf.filename}`")
                    index_content.append("")
            
            # Write subject index
            subject_file = indexes_dir / f"{subject.value}_index.md"
            with open(subject_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(index_content))
    
    def _create_group_indexes(self, content_groups: List[ContentGroup]):
        """Create index files for content groups."""
        indexes_dir = self.base_output_dir / "indexes"
        indexes_dir.mkdir(exist_ok=True)
        
        # Group by type
        type_groups = {}
        for group in content_groups:
            if group.group_type not in type_groups:
                type_groups[group.group_type] = []
            type_groups[group.group_type].append(group)
        
        # Create index for each group type
        for group_type, groups in type_groups.items():
            index_content = []
            index_content.append(f"# {group_type.title()} Groups Index")
            index_content.append(f"Total Groups: {len(groups)}")
            index_content.append("")
            
            for group in sorted(groups, key=lambda x: x.title):
                index_content.append(f"## {group.title}")
                index_content.append(f"- Group ID: {group.group_id}")
                index_content.append(f"- PDFs: {len(group.pdfs)}")
                index_content.append(f"- Created: {group.created_date.strftime('%Y-%m-%d')}")
                
                if group.description:
                    index_content.append(f"- Description: {group.description}")
                
                if group.metadata:
                    index_content.append("- Metadata:")
                    for key, value in group.metadata.items():
                        index_content.append(f"  - {key}: {value}")
                
                index_content.append("- Files:")
                for pdf_filename in group.pdfs:
                    index_content.append(f"  - `{pdf_filename}`")
                
                index_content.append("")
            
            # Write group type index
            group_file = indexes_dir / f"{group_type}_groups_index.md"
            with open(group_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(index_content))
    
    def _create_course_indexes(self, content_groups: List[ContentGroup]):
        """Create special index for course materials."""
        course_groups = [g for g in content_groups if g.group_type == 'course']
        
        if not course_groups:
            return
        
        indexes_dir = self.base_output_dir / "indexes"
        indexes_dir.mkdir(exist_ok=True)
        
        index_content = []
        index_content.append("# Course Materials Index")
        index_content.append(f"Total Courses: {len(course_groups)}")
        index_content.append("")
        
        for course in sorted(course_groups, key=lambda x: x.metadata.get('course_code', x.title)):
            course_code = course.metadata.get('course_code', 'Unknown')
            institution = course.metadata.get('institution', 'Unknown')
            level = course.metadata.get('academic_level', 'unknown')
            
            index_content.append(f"## {course_code}: {course.title}")
            index_content.append(f"- Institution: {institution}")
            index_content.append(f"- Level: {level.replace('_', ' ').title()}")
            index_content.append(f"- Materials: {len(course.pdfs)} files")
            index_content.append("")
            
            # List materials
            for pdf_filename in course.pdfs:
                index_content.append(f"  - `{pdf_filename}`")
            
            index_content.append("")
        
        # Write course index
        course_file = indexes_dir / "courses_index.md"
        with open(course_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(index_content))
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in folder names."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove multiple underscores and trim
        filename = '_'.join(filename.split())
        return filename[:100]  # Limit length
    
    def save_organization(self, organized_collection: OrganizedCollection, 
                         filepath: str = "organization_data.json"):
        """Save organization data to file."""
        # Convert to serializable format
        data = {
            "base_path": str(organized_collection.base_path),
            "hierarchical_structure": organized_collection.hierarchical_structure,
            "content_groups": [],
            "folder_statistics": organized_collection.folder_statistics,
            "group_statistics": organized_collection.group_statistics,
            "organization_metadata": organized_collection.organization_metadata,
            "created_date": organized_collection.created_date.isoformat(),
            "last_updated": organized_collection.last_updated.isoformat()
        }
        
        # Convert content groups
        for group in organized_collection.content_groups:
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
            data["content_groups"].append(group_data)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_organization_summary(self, organized_collection: OrganizedCollection) -> Dict[str, any]:
        """Get a summary of the organization results."""
        return {
            "total_pdfs": organized_collection.organization_metadata.get("total_pdfs", 0),
            "hierarchical_folders": len(organized_collection.hierarchical_structure),
            "content_groups": len(organized_collection.content_groups),
            "group_types": organized_collection.group_statistics.get("by_type", {}),
            "subject_distribution": organized_collection.folder_statistics.get("by_subject", {}),
            "level_distribution": organized_collection.folder_statistics.get("by_level", {}),
            "organization_date": organized_collection.created_date.strftime("%Y-%m-%d %H:%M:%S"),
            "base_path": str(organized_collection.base_path)
        }