"""
Simplified scraper controller that integrates scraping, downloading, and organization.
Focuses on core functionality with the advanced organization system.
"""

import asyncio
import logging
import requests
import time
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from config.settings import config_manager
from data.models import PDFMetadata, AcademicLevel
from utils.advanced_organization import AdvancedOrganizationSystem
from processors.metadata_extractor import PDFMetadataExtractor


class SimplifiedScraperController:
    """
    Simplified scraper controller that scrapes, downloads, and organizes PDFs automatically.
    Integrates all functionality in a streamlined workflow.
    """
    
    def __init__(self):
        """Initialize the simplified scraper controller."""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.app_config = config_manager.load_app_config()
        self.sources = config_manager.load_sources()
        
        # Initialize components
        self.metadata_extractor = PDFMetadataExtractor()
        self.organization_system = AdvancedOrganizationSystem(
            base_output_dir=self.app_config.base_output_dir
        )
        
        # Setup session for downloads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational PDF Collector for Research)'
        })
        
        # Processing state
        self.processed_pdfs: List[PDFMetadata] = []
        self.downloaded_files: List[str] = []
        
        self.logger.info("Simplified scraper controller initialized")
    
    async def scrape_and_organize(self, max_pdfs_per_source: int = 20) -> Dict[str, any]:
        """
        Complete workflow: scrape internet, download PDFs, and organize automatically.
        
        Args:
            max_pdfs_per_source: Maximum PDFs to download per source
            
        Returns:
            Dictionary with complete results
        """
        self.logger.info("Starting complete scrape and organize workflow...")
        
        results = {
            "workflow_started": datetime.now(),
            "sources_processed": 0,
            "pdfs_discovered": 0,
            "pdfs_downloaded": 0,
            "pdfs_processed": 0,
            "organization_results": None,
            "errors": []
        }
        
        try:
            # Step 1: Scrape and download from all sources
            self.logger.info("Step 1: Scraping and downloading PDFs from internet sources...")
            await self._scrape_all_sources(max_pdfs_per_source, results)
            
            # Step 2: Process downloaded PDFs
            self.logger.info("Step 2: Processing downloaded PDFs...")
            await self._process_all_downloaded_pdfs(results)
            
            # Step 3: Organize everything with advanced organization system
            self.logger.info("Step 3: Organizing collection with advanced organization system...")
            await self._organize_complete_collection(results)
            
            # Step 4: Generate summary report
            self.logger.info("Step 4: Generating final reports...")
            await self._generate_final_report(results)
            
            results["workflow_completed"] = datetime.now()
            duration = (results["workflow_completed"] - results["workflow_started"]).total_seconds()
            
            self.logger.info(f"Complete workflow finished in {duration:.1f} seconds!")
            self.logger.info(f"Downloaded: {results['pdfs_downloaded']} PDFs")
            self.logger.info(f"Processed: {results['pdfs_processed']} PDFs")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in scrape and organize workflow: {e}")
            results["errors"].append(str(e))
            raise
    
    async def _scrape_all_sources(self, max_per_source: int, results: Dict):
        """Scrape PDFs from all configured sources."""
        
        # Define comprehensive list of educational sources to scrape
        educational_sources = [
            # MIT OpenCourseWare
            {
                'name': 'MIT OCW Computer Science',
                'url': 'https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/',
                'subject': 'computer_science'
            },
            {
                'name': 'MIT OCW Mathematics', 
                'url': 'https://ocw.mit.edu/courses/mathematics/',
                'subject': 'mathematics'
            },
            {
                'name': 'MIT OCW Physics',
                'url': 'https://ocw.mit.edu/courses/physics/',
                'subject': 'physics'
            },
            {
                'name': 'MIT OCW Economics',
                'url': 'https://ocw.mit.edu/courses/economics/',
                'subject': 'economics'
            },
            
            # Stanford University
            {
                'name': 'Stanford CS Course Materials',
                'url': 'https://cs.stanford.edu/courses',
                'subject': 'computer_science'
            },
            {
                'name': 'Stanford Engineering',
                'url': 'https://engineering.stanford.edu/students-academics/courses',
                'subject': 'engineering'
            },
            
            # UC Berkeley
            {
                'name': 'UC Berkeley EECS',
                'url': 'https://eecs.berkeley.edu/resources/students/class-materials',
                'subject': 'computer_science'
            },
            {
                'name': 'UC Berkeley Mathematics',
                'url': 'https://math.berkeley.edu/courses',
                'subject': 'mathematics'
            },
            
            # Carnegie Mellon University
            {
                'name': 'CMU Computer Science',
                'url': 'https://www.cs.cmu.edu/courses',
                'subject': 'computer_science'
            },
            {
                'name': 'CMU Machine Learning',
                'url': 'https://www.ml.cmu.edu/academics/courses.html',
                'subject': 'computer_science'
            },
            
            # Harvard University
            {
                'name': 'Harvard CS50',
                'url': 'https://cs50.harvard.edu/college/',
                'subject': 'computer_science'
            },
            {
                'name': 'Harvard Mathematics',
                'url': 'https://www.math.harvard.edu/teaching/',
                'subject': 'mathematics'
            },
            
            # Princeton University
            {
                'name': 'Princeton Computer Science',
                'url': 'https://www.cs.princeton.edu/courses',
                'subject': 'computer_science'
            },
            {
                'name': 'Princeton Mathematics',
                'url': 'https://www.math.princeton.edu/undergraduate/courses',
                'subject': 'mathematics'
            },
            
            # University of Washington
            {
                'name': 'UW Computer Science',
                'url': 'https://www.cs.washington.edu/education/courses',
                'subject': 'computer_science'
            },
            
            # Georgia Tech
            {
                'name': 'Georgia Tech CS',
                'url': 'https://www.cc.gatech.edu/academics/courses',
                'subject': 'computer_science'
            },
            
            # arXiv repositories
            {
                'name': 'arXiv Computer Science',
                'url': 'https://arxiv.org/list/cs/recent',
                'subject': 'computer_science'
            },
            {
                'name': 'arXiv Mathematics',
                'url': 'https://arxiv.org/list/math/recent', 
                'subject': 'mathematics'
            },
            {
                'name': 'arXiv Physics',
                'url': 'https://arxiv.org/list/physics/recent',
                'subject': 'physics'
            },
            {
                'name': 'arXiv Statistics',
                'url': 'https://arxiv.org/list/stat/recent',
                'subject': 'statistics'
            },
            
            # International Universities
            {
                'name': 'Oxford Mathematics',
                'url': 'https://www.maths.ox.ac.uk/study-here/undergraduate-study/lecture-notes',
                'subject': 'mathematics'
            },
            {
                'name': 'Cambridge Computer Science',
                'url': 'https://www.cl.cam.ac.uk/teaching/current/',
                'subject': 'computer_science'
            },
            {
                'name': 'ETH Zurich Mathematics',
                'url': 'https://math.ethz.ch/education/bachelor/lectures.html',
                'subject': 'mathematics'
            },
            {
                'name': 'University of Toronto CS',
                'url': 'https://web.cs.toronto.edu/undergraduate/courses',
                'subject': 'computer_science'
            },
            
            # Specialized Educational Resources
            {
                'name': 'Khan Academy',
                'url': 'https://www.khanacademy.org/',
                'subject': 'general'
            },
            {
                'name': 'Coursera Course Materials',
                'url': 'https://www.coursera.org/browse/computer-science',
                'subject': 'computer_science'
            },
            {
                'name': 'edX Course Materials',
                'url': 'https://www.edx.org/learn/computer-science',
                'subject': 'computer_science'
            },
            
            # Government and Research Institutions
            {
                'name': 'NIST Publications',
                'url': 'https://www.nist.gov/publications',
                'subject': 'engineering'
            },
            {
                'name': 'NASA Technical Reports',
                'url': 'https://ntrs.nasa.gov/',
                'subject': 'engineering'
            },
            {
                'name': 'NIH Publications',
                'url': 'https://www.ncbi.nlm.nih.gov/pmc/',
                'subject': 'biology'
            },
            
            # Academic Publishers (Open Access)
            {
                'name': 'PLOS ONE',
                'url': 'https://journals.plos.org/plosone/',
                'subject': 'science'
            },
            {
                'name': 'Nature Open Access',
                'url': 'https://www.nature.com/nature/articles',
                'subject': 'science'
            },
            {
                'name': 'IEEE Xplore Open Access',
                'url': 'https://ieeexplore.ieee.org/browse/periodicals/title',
                'subject': 'engineering'
            },
            
            # Educational Repositories
            {
                'name': 'ERIC Education Database',
                'url': 'https://eric.ed.gov/',
                'subject': 'education'
            },
            {
                'name': 'Directory of Open Access Journals',
                'url': 'https://doaj.org/',
                'subject': 'general'
            },
            {
                'name': 'OpenStax Textbooks',
                'url': 'https://openstax.org/subjects',
                'subject': 'general'
            },
            {
                'name': 'MIT Press Open Access',
                'url': 'https://mitpress.mit.edu/books/open-access',
                'subject': 'general'
            },
            
            # Specific Subject Resources
            {
                'name': 'Mathematical Association of America',
                'url': 'https://www.maa.org/press/ebooks',
                'subject': 'mathematics'
            },
            {
                'name': 'American Physical Society',
                'url': 'https://journals.aps.org/',
                'subject': 'physics'
            },
            {
                'name': 'ACM Digital Library',
                'url': 'https://dl.acm.org/',
                'subject': 'computer_science'
            },
            {
                'name': 'Springer Open Access',
                'url': 'https://www.springeropen.com/',
                'subject': 'general'
            },
            
            # International Educational Resources
            {
                'name': 'CERN Document Server',
                'url': 'https://cds.cern.ch/',
                'subject': 'physics'
            },
            {
                'name': 'Max Planck Institute',
                'url': 'https://www.mpg.de/publications',
                'subject': 'science'
            },
            {
                'name': 'French National Research',
                'url': 'https://hal.archives-ouvertes.fr/',
                'subject': 'general'
            },
            
            # Business and Economics
            {
                'name': 'Federal Reserve Publications',
                'url': 'https://www.federalreserve.gov/publications.htm',
                'subject': 'economics'
            },
            {
                'name': 'World Bank Open Knowledge',
                'url': 'https://openknowledge.worldbank.org/',
                'subject': 'economics'
            },
            {
                'name': 'IMF Publications',
                'url': 'https://www.imf.org/en/Publications',
                'subject': 'economics'
            },
            
            # Medical and Health Sciences
            {
                'name': 'PubMed Central',
                'url': 'https://www.ncbi.nlm.nih.gov/pmc/',
                'subject': 'medicine'
            },
            {
                'name': 'WHO Publications',
                'url': 'https://www.who.int/publications',
                'subject': 'medicine'
            },
            {
                'name': 'CDC Publications',
                'url': 'https://www.cdc.gov/publications/',
                'subject': 'medicine'
            },
            
            # Environmental and Earth Sciences
            {
                'name': 'USGS Publications',
                'url': 'https://www.usgs.gov/publications',
                'subject': 'earth_science'
            },
            {
                'name': 'NOAA Publications',
                'url': 'https://www.noaa.gov/education/resource-collections',
                'subject': 'earth_science'
            },
            {
                'name': 'EPA Publications',
                'url': 'https://www.epa.gov/research',
                'subject': 'environmental_science'
            }
        ]
        
        total_downloaded = 0
        
        for source in educational_sources:
            if total_downloaded >= max_per_source * len(educational_sources):
                break
                
            try:
                self.logger.info(f"Scraping source: {source['name']}")
                
                # Discover PDFs from this source
                pdf_urls = await self._discover_pdfs_from_source(source)
                results["pdfs_discovered"] += len(pdf_urls)
                
                # Download PDFs from this source
                downloaded_count = await self._download_pdfs_from_source(
                    source, pdf_urls[:max_per_source]
                )
                
                total_downloaded += downloaded_count
                results["pdfs_downloaded"] += downloaded_count
                results["sources_processed"] += 1
                
                self.logger.info(f"Downloaded {downloaded_count} PDFs from {source['name']}")
                
                # Small delay between sources
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.warning(f"Error scraping {source['name']}: {e}")
                results["errors"].append(f"Source {source['name']}: {str(e)}")
                continue
    
    async def _discover_pdfs_from_source(self, source: Dict) -> List[str]:
        """Discover PDF URLs from a source using multiple strategies."""
        pdf_urls = []
        
        try:
            # Get the page content with better headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = self.session.get(source['url'], timeout=30, headers=headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Strategy 1: Direct PDF links in anchor tags
            for link in soup.find_all('a', href=True):
                href = link['href']
                if self._looks_like_pdf_link(href):
                    full_url = urljoin(source['url'], href)
                    if self._is_valid_pdf_url(full_url) and full_url not in pdf_urls:
                        pdf_urls.append(full_url)
            
            # Strategy 2: Search for PDF patterns in all text
            pdf_patterns = [
                r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
                r'href=["\']([^"\']*filetype[=:]pdf[^"\']*)["\']',
                r'href=["\']([^"\']*\.pdf\?[^"\']*)["\']',
                r'src=["\']([^"\']*\.pdf[^"\']*)["\']',
                r'(https?://[^\s<>"\']*\.pdf[^\s<>"\']*)',
                r'(https?://[^\s<>"\']*filetype[=:]pdf[^\s<>"\']*)'
            ]
            
            html_content = str(soup)
            for pattern in pdf_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    full_url = urljoin(source['url'], match)
                    if self._is_valid_pdf_url(full_url) and full_url not in pdf_urls:
                        pdf_urls.append(full_url)
            
            # Strategy 3: Look for common academic PDF indicators
            academic_indicators = [
                'lecture', 'notes', 'slides', 'handout', 'assignment', 'homework',
                'exam', 'quiz', 'solution', 'textbook', 'chapter', 'paper',
                'publication', 'research', 'thesis', 'dissertation', 'report'
            ]
            
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().lower()
                href = link['href']
                
                # Check if link text suggests it might lead to PDFs
                if any(indicator in link_text for indicator in academic_indicators):
                    if 'pdf' in link_text or href.lower().endswith('.pdf'):
                        full_url = urljoin(source['url'], href)
                        if self._is_valid_pdf_url(full_url) and full_url not in pdf_urls:
                            pdf_urls.append(full_url)
            
            # Strategy 4: Look for download links and document repositories
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text().lower()
                
                if ('download' in link_text or 'document' in link_text or 
                    'material' in link_text or 'resource' in link_text):
                    # Follow these links to look for PDFs (one level deep)
                    try:
                        if not href.startswith('http'):
                            href = urljoin(source['url'], href)
                        
                        # Quick check if this might be a PDF directory
                        if self._might_contain_pdfs(href):
                            sub_pdfs = await self._quick_pdf_scan(href)
                            pdf_urls.extend(sub_pdfs)
                    except:
                        continue
            
            # Remove duplicates while preserving order
            unique_pdfs = []
            seen = set()
            for url in pdf_urls:
                if url not in seen:
                    unique_pdfs.append(url)
                    seen.add(url)
            
            self.logger.info(f"Discovered {len(unique_pdfs)} PDF URLs from {source['name']}")
            return unique_pdfs[:30]  # Reasonable limit per source
            
        except Exception as e:
            self.logger.warning(f"Error discovering PDFs from {source['name']}: {e}")
            return []  # Return empty list on error
    
    def _looks_like_pdf_link(self, href: str) -> bool:
        """Check if a link looks like it might be a PDF."""
        href_lower = href.lower()
        return (href_lower.endswith('.pdf') or 
                'filetype=pdf' in href_lower or 
                'type=pdf' in href_lower or
                'format=pdf' in href_lower or
                '.pdf?' in href_lower or
                '.pdf#' in href_lower)
    
    def _might_contain_pdfs(self, url: str) -> bool:
        """Check if a URL might contain PDFs."""
        url_lower = url.lower()
        pdf_indicators = [
            'document', 'material', 'resource', 'download', 'file',
            'publication', 'paper', 'report', 'lecture', 'course'
        ]
        return any(indicator in url_lower for indicator in pdf_indicators)
    
    async def _quick_pdf_scan(self, url: str) -> List[str]:
        """Quickly scan a page for PDF links (one level deep)."""
        pdf_urls = []
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if self._looks_like_pdf_link(href):
                    full_url = urljoin(url, href)
                    if self._is_valid_pdf_url(full_url):
                        pdf_urls.append(full_url)
            
            return pdf_urls[:10]  # Limit from sub-pages
            
        except:
            return []
    
    def _is_valid_pdf_url(self, url: str) -> bool:
        """Check if URL is a valid PDF URL."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Skip obviously non-PDF URLs
            if any(skip in url.lower() for skip in ['javascript:', 'mailto:', '#', 'tel:']):
                return False
            
            # Check for PDF extension or PDF-related parameters
            path_lower = parsed.path.lower()
            query_lower = parsed.query.lower()
            url_lower = url.lower()
            
            pdf_indicators = [
                path_lower.endswith('.pdf'),
                'filetype=pdf' in query_lower,
                'type=pdf' in query_lower,
                'format=pdf' in query_lower,
                '.pdf?' in url_lower,
                '.pdf#' in url_lower,
                '/pdf/' in url_lower,
                'download.pdf' in url_lower,
                'arxiv.org/pdf/' in url_lower,
                'arxiv.org/abs/' in url_lower  # arXiv abstracts can lead to PDFs
            ]
            
            return any(pdf_indicators)
        except:
            return False
    
    async def _download_pdfs_from_source(self, source: Dict, pdf_urls: List[str]) -> int:
        """Download PDFs from a source."""
        downloaded_count = 0
        
        # Create source-specific directory
        source_dir = Path(self.app_config.base_output_dir) / "downloads" / source['subject']
        source_dir.mkdir(parents=True, exist_ok=True)
        
        for url in pdf_urls:
            try:
                # Generate filename
                filename = self._generate_filename(url, source)
                filepath = source_dir / filename
                
                # Skip if already exists
                if filepath.exists():
                    self.logger.debug(f"File already exists: {filename}")
                    continue
                
                # Download the PDF
                self.logger.debug(f"Downloading: {url}")
                
                # Handle special cases like arXiv
                download_url = self._get_actual_pdf_url(url)
                
                response = self.session.get(download_url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                
                # Save the file and verify it's a PDF
                with open(filepath, 'wb') as f:
                    first_chunk_written = False
                    for chunk in response.iter_content(chunk_size=8192):
                        if not first_chunk_written:
                            # Check first chunk to verify it's a PDF
                            if not (chunk.startswith(b'%PDF') or 'pdf' in content_type):
                                self.logger.debug(f"Not a PDF: {url}")
                                break
                            first_chunk_written = True
                        f.write(chunk)
                    else:
                        # File was written successfully
                        pass
                    
                # If we broke out of the loop, remove the file
                if not first_chunk_written or not filepath.exists():
                    if filepath.exists():
                        filepath.unlink()
                    continue
                
                # Verify it's a valid PDF file
                if filepath.stat().st_size < 1024:  # Too small
                    filepath.unlink()
                    continue
                
                self.downloaded_files.append(str(filepath))
                downloaded_count += 1
                
                self.logger.info(f"Downloaded: {filename} ({filepath.stat().st_size // 1024} KB)")
                
                # Small delay between downloads
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Failed to download {url}: {e}")
                continue
        
        return downloaded_count
    
    def _generate_filename(self, url: str, source: Dict) -> str:
        """Generate a safe filename for the PDF."""
        # Extract filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or not filename.lower().endswith('.pdf'):
            # Generate filename from URL components
            filename = re.sub(r'[^\w\-_.]', '_', parsed.path.replace('/', '_'))
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
        
        # Add source prefix and clean up
        clean_filename = re.sub(r'[^\w\-_.]', '_', filename)
        source_prefix = re.sub(r'[^\w]', '_', source['name'])[:20]
        
        return f"{source_prefix}_{clean_filename}"[:100]  # Limit length
    
    def _get_actual_pdf_url(self, url: str) -> str:
        """Convert URLs to actual PDF download URLs."""
        # Handle arXiv URLs
        if 'arxiv.org/abs/' in url:
            # Convert abstract URL to PDF URL
            paper_id = url.split('/abs/')[-1]
            return f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        # Handle other special cases
        if 'github.com' in url and '/blob/' in url:
            # Convert GitHub blob URLs to raw URLs
            return url.replace('/blob/', '/raw/')
        
        # Handle Google Drive sharing links
        if 'drive.google.com' in url and '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return url
    
    async def _process_all_downloaded_pdfs(self, results: Dict):
        """Process all downloaded PDFs to extract metadata."""
        processed_count = 0
        
        for filepath in self.downloaded_files:
            try:
                self.logger.debug(f"Processing: {Path(filepath).name}")
                
                # Extract metadata
                metadata = self.metadata_extractor.extract_pdf_metadata(filepath)
                
                # Enhance metadata with source information
                self._enhance_metadata(metadata, filepath)
                
                self.processed_pdfs.append(metadata)
                processed_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process {filepath}: {e}")
                continue
        
        results["pdfs_processed"] = processed_count
        self.logger.info(f"Successfully processed {processed_count} PDFs")
    
    def _enhance_metadata(self, metadata: PDFMetadata, filepath: str):
        """Enhance metadata with additional information."""
        path = Path(filepath)
        
        # Determine subject from path
        if 'computer' in str(path).lower() or 'cs' in str(path).lower():
            metadata.subject_area = 'computer_science'
        elif 'math' in str(path).lower():
            metadata.subject_area = 'mathematics'
        elif 'physics' in str(path).lower():
            metadata.subject_area = 'physics'
        
        # Set download source
        metadata.download_source = "internet_scraping"
        metadata.download_date = datetime.now()
        
        # Enhance keywords based on filename and path
        filename_lower = path.name.lower()
        path_keywords = []
        
        if 'lecture' in filename_lower:
            path_keywords.extend(['lecture', 'course material'])
        if 'tutorial' in filename_lower:
            path_keywords.extend(['tutorial', 'learning'])
        if 'assignment' in filename_lower:
            path_keywords.extend(['assignment', 'homework'])
        if 'exam' in filename_lower:
            path_keywords.extend(['exam', 'test'])
        
        metadata.keywords.extend(path_keywords)
        metadata.keywords = list(set(metadata.keywords))  # Remove duplicates
    
    async def _organize_complete_collection(self, results: Dict):
        """Organize the complete collection using the advanced organization system."""
        if not self.processed_pdfs:
            self.logger.warning("No PDFs to organize")
            return
        
        try:
            # Organize the collection
            organized_collection = self.organization_system.organize_collection(self.processed_pdfs)
            
            # Get organization summary
            summary = self.organization_system.get_organization_summary(organized_collection)
            
            # Save organization data
            org_data_path = Path(self.app_config.base_output_dir) / "organization_data.json"
            self.organization_system.save_organization(organized_collection, str(org_data_path))
            
            results["organization_results"] = {
                "organized_collection": organized_collection,
                "summary": summary,
                "organization_file": str(org_data_path)
            }
            
            self.logger.info(f"Organization complete: {summary['total_pdfs']} PDFs organized into {summary['content_groups']} groups")
            
        except Exception as e:
            self.logger.error(f"Error during organization: {e}")
            results["errors"].append(f"Organization error: {str(e)}")
    
    async def _generate_final_report(self, results: Dict):
        """Generate final workflow report."""
        try:
            reports_dir = Path(self.app_config.base_output_dir) / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Create comprehensive report
            report = {
                "workflow_summary": {
                    "started": results.get("workflow_started"),
                    "completed": results.get("workflow_completed"),
                    "duration_minutes": (
                        (results.get("workflow_completed", datetime.now()) - 
                         results.get("workflow_started", datetime.now())).total_seconds() / 60
                    ) if results.get("workflow_started") else 0
                },
                "scraping_results": {
                    "sources_processed": results.get("sources_processed", 0),
                    "pdfs_discovered": results.get("pdfs_discovered", 0),
                    "pdfs_downloaded": results.get("pdfs_downloaded", 0),
                    "pdfs_processed": results.get("pdfs_processed", 0),
                    "success_rate": (
                        results.get("pdfs_processed", 0) / max(results.get("pdfs_discovered", 1), 1) * 100
                    )
                },
                "organization_results": results.get("organization_results", {}).get("summary", {}),
                "errors": results.get("errors", [])
            }
            
            # Save report
            report_path = reports_dir / f"complete_workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Final report saved to: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Error generating final report: {e}")
    
    def get_results_summary(self, results: Dict) -> str:
        """Get a formatted summary of results."""
        summary_lines = [
            "🎉 Complete Scrape and Organize Workflow Results:",
            "=" * 50,
            f"📊 Sources Processed: {results.get('sources_processed', 0)}",
            f"🔍 PDFs Discovered: {results.get('pdfs_discovered', 0)}",
            f"⬇️  PDFs Downloaded: {results.get('pdfs_downloaded', 0)}",
            f"⚙️  PDFs Processed: {results.get('pdfs_processed', 0)}",
        ]
        
        if results.get("organization_results"):
            org_summary = results["organization_results"]["summary"]
            summary_lines.extend([
                "",
                "📁 Organization Results:",
                f"   Total PDFs Organized: {org_summary.get('total_pdfs', 0)}",
                f"   Hierarchical Folders: {org_summary.get('hierarchical_folders', 0)}",
                f"   Content Groups: {org_summary.get('content_groups', 0)}",
                f"   Output Directory: {org_summary.get('base_path', 'N/A')}"
            ])
        
        if results.get("errors"):
            summary_lines.extend([
                "",
                f"⚠️  Errors Encountered: {len(results['errors'])}",
                "   (Check logs for details)"
            ])
        
        return "\n".join(summary_lines)