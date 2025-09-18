import os
import requests
import time
import csv
import re
import hashlib
from urllib.parse import urljoin, urlparse
from pathlib import Path
from dataclasses import dataclass
import logging

# -- Logging setup --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PDFSource:
    name: str
    url: str
    lang: str
    field: str

class MultilingualPDFScraper:
    def __init__(self, base_dir: str = str(Path.home() / "EducationalPDFs")):
        self.base_dir = Path(base_dir)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Educational PDF Collector for Students)'})
        self.metadata_path = self.base_dir / "metadata.csv"
        self.languages = ["Arabic", "French", "English"]
        self.fields = {
            'Arabic': ['Computer_Science_AR', 'Mathematics_AR'],
            'French': ['Informatique', 'Mathematiques'],
            'English': ['Computer_Science', 'Mathematics'],
        }
        self.setup_dirs()
        self.init_metadata()

    def setup_dirs(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for lang in self.languages:
            (self.base_dir / lang).mkdir(exist_ok=True)
            for field in self.fields[lang]:
                (self.base_dir / lang / field).mkdir(exist_ok=True)

    def init_metadata(self):
        if not self.metadata_path.exists():
            with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'language', 'field', 'url', 'source', 'hash', 'filesize', 'title', 'text_excerpt'])

    def get_sources(self):
        return {
            'English': {
                'Computer_Science': [
                    {'name': 'MIT OCW EECS', 'url': 'https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/'},
                    {'name': 'Stanford CS', 'url': 'https://cs.stanford.edu/'},
                    {'name': 'Berkeley EECS', 'url': 'https://eecs.berkeley.edu/resources/students/class-materials'},
                    {'name': 'CMU CS', 'url': 'https://www.cs.cmu.edu/courses'},
                    {'name': 'Princeton CS', 'url': 'https://www.cs.princeton.edu/courses/archive/'},
                    {'name': 'UCLA CS', 'url': 'https://www.cs.ucla.edu/academics/courses'},
                ],
                'Mathematics': [
                    {'name': 'MIT OCW Math', 'url': 'https://ocw.mit.edu/courses/mathematics/'},
                    {'name': 'Harvard Math', 'url': 'https://www.math.harvard.edu/teaching/archives/'},
                    {'name': 'UCLA Math', 'url': 'https://www.math.ucla.edu/courses'},
                    {'name': 'Oxford Math', 'url': 'https://www.maths.ox.ac.uk/study-here/undergraduate-study/lecture-notes'},
                    {'name': 'ETH Zurich Math', 'url': 'https://math.ethz.ch/students/lecture-notes.html'},
                ],
            },
            'French': {
                'Informatique': [
                    {'name': 'INRIA', 'url': 'https://hal.inria.fr/'},
                    {'name': 'Université Lyon CS', 'url': 'https://perso.limos.fr/~kevin.jean/wikindx/'},
                    {'name': 'Télécom ParisTech', 'url': 'https://www.telecom-paris.fr/ressources-pedagogiques/'},
                ],
                'Mathematiques': [
                    {'name': 'CNRS Math', 'url': 'https://math.cnrs.fr/'},
                    {'name': 'UPMC Math', 'url': 'https://www.ljll.math.upmc.fr/cours/'},
                ],
            },
            'Arabic': {
                'Computer_Science_AR': [
                    {'name': 'KSU CS', 'url': 'https://cs.ksu.edu.sa/arabic/cs-department'},
                    {'name': 'MIT OCW Arabic', 'url': 'https://ocw.mit.edu/courses/translated-courses/ar/'},
                ],
                'Mathematics_AR': [
                    {'name': 'Arab Math Portal', 'url': 'https://www.arabmath.org/'},
                ],
            }
        }

    def get_hash(self, filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, 'rb') as file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def filename(self, url: str, language: str, field: str):
        basename = os.path.basename(urlparse(url).path)
        if not basename or not basename.endswith('.pdf'):
            basename = re.sub(r'\W+', '_', basename) + '.pdf'
        return f"{language}_{field}_{basename[:64]}"

    def save_metadata(self, filepath, language, field, url, source, title="", text_excerpt=""):
        file_hash = self.get_hash(filepath)
        file_size = filepath.stat().st_size
        with open(self.metadata_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                filepath.name, language, field, url, source, file_hash, file_size, title, text_excerpt[:500]
            ])

    def download_pdf(self, url, language, field, source="unknown"):
        filename = self.filename(url, language, field)
        filepath = self.base_dir / language / field / filename
        if filepath.exists():
            logger.info(f"File exists: {filepath.name}")
            return False
        try:
            response = self.session.get(url, stream=True, timeout=15)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(10240):
                    f.write(chunk)
            # Try to extract simple text excerpt and metadata if possible
            try:
                import PyPDF2
                with open(filepath, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    title = pdf_reader.metadata.title if pdf_reader.metadata else ""
                    text_excerpt = ""
                    for page in pdf_reader.pages[:1]:
                        text_excerpt += page.extract_text() or ""
            except Exception:
                title, text_excerpt = "", ""
            self.save_metadata(filepath, language, field, url, source, title, text_excerpt)
            logger.info(f"Saved: {filepath}, size: {filepath.stat().st_size/1024:.1f} KB")
            return True
        except Exception as e:
            logger.warning(f"Download failed {url}: {e}")
            return False

    def scrape_educational_site(self, url: str, language: str, field: str, source_name: str, max_per_field=5):
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            pdf_patterns = [
                r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
                r'href=["\']([^"\']*filetype[=:]pdf[^"\']*)["\']'
            ]
            pdf_links = []
            for pattern in pdf_patterns:
                links = re.findall(pattern, response.text, re.IGNORECASE)
                pdf_links.extend(links)
            pdf_links = [link for link in pdf_links if link.endswith('.pdf')]
            downloaded = 0
            for link in pdf_links:
                if downloaded >= max_per_field:
                    break
                full_url = urljoin(url, link)
                if self.download_pdf(full_url, language, field, source_name):
                    downloaded += 1
                    time.sleep(1)
        except Exception as e:
            logger.warning(f"Scraping failed for {url}: {e}")

    def scrape_all_sources(self, max_per_field=5, delay=1):
        sources = self.get_sources()
        for lang, fields in sources.items():
            for field, sites in fields.items():
                for site in sites:
                    logger.info(f"Scraping {site['name']}: {site['url']}")
                    try:
                        self.scrape_educational_site(site['url'], lang, field, site['name'], max_per_field)
                    except Exception as e:
                        logger.warning(f"Failed scraping {site['name']} ({site['url']}): {e}")
                    time.sleep(delay)

if __name__ == "__main__":
    print("Starting Multilingual Educational PDF Scraper!")
    scraper = MultilingualPDFScraper()
    scraper.scrape_all_sources(max_per_field=5)
    print("DONE! Check your EducationalPDFs folder and metadata.csv.")
