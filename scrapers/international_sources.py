"""
International source support for educational PDF scraping.
Handles non-Latin character encodings, language-specific strategies, and cultural context awareness.
"""

import asyncio
import logging
import re
import json
from typing import List, Dict, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse, quote
import aiohttp
from bs4 import BeautifulSoup
import chardet

from data.models import SourceConfig, PDFMetadata, AcademicLevel, ScrapingStrategy


class InternationalSourceManager:
    """
    Manager for international educational sources with multi-language support.
    Handles encoding detection, language-specific scraping, and cultural context.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize international source manager.
        
        Args:
            config: Configuration dictionary for international sources
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Language-specific configurations
        self.language_configs = self._load_language_configs()
        
        # Regional academic databases
        self.regional_databases = self._load_regional_databases()
        
        # Character encoding mappings
        self.encoding_mappings = self._load_encoding_mappings()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize HTTP session with international support."""
        timeout = aiohttp.ClientTimeout(total=30)
        
        # Headers that work well internationally
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,ja;q=0.6,ko;q=0.5,es;q=0.4,fr;q=0.3,de;q=0.2',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Charset': 'utf-8, iso-8859-1;q=0.5, *;q=0.1',
        }
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    def _load_language_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load language-specific configurations."""
        return {
            'zh': {  # Chinese
                'name': 'Chinese',
                'encodings': ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5'],
                'pdf_patterns': [r'\.pdf$', r'文档\.pdf', r'论文\.pdf', r'资料\.pdf'],
                'academic_keywords': ['大学', '学院', '研究', '论文', '学术', '教育', '课程'],
                'exclude_patterns': [r'广告', r'商业', r'购买'],
                'date_formats': ['%Y年%m月%d日', '%Y-%m-%d', '%m/%d/%Y'],
                'number_system': 'chinese_traditional'
            },
            'ja': {  # Japanese
                'name': 'Japanese',
                'encodings': ['utf-8', 'shift_jis', 'euc-jp', 'iso-2022-jp'],
                'pdf_patterns': [r'\.pdf$', r'資料\.pdf', r'論文\.pdf', r'講義\.pdf'],
                'academic_keywords': ['大学', '学校', '研究', '論文', '学術', '教育', '講義'],
                'exclude_patterns': [r'広告', r'商業', r'購入'],
                'date_formats': ['%Y年%m月%d日', '%Y-%m-%d', '%m/%d/%Y'],
                'number_system': 'japanese'
            },
            'ko': {  # Korean
                'name': 'Korean',
                'encodings': ['utf-8', 'euc-kr', 'cp949'],
                'pdf_patterns': [r'\.pdf$', r'자료\.pdf', r'논문\.pdf', r'강의\.pdf'],
                'academic_keywords': ['대학교', '대학', '연구', '논문', '학술', '교육', '강의'],
                'exclude_patterns': [r'광고', r'상업', r'구매'],
                'date_formats': ['%Y년 %m월 %d일', '%Y-%m-%d', '%m/%d/%Y'],
                'number_system': 'korean'
            },
            'ar': {  # Arabic
                'name': 'Arabic',
                'encodings': ['utf-8', 'iso-8859-6', 'windows-1256'],
                'pdf_patterns': [r'\.pdf$', r'وثيقة\.pdf', r'بحث\.pdf', r'محاضرة\.pdf'],
                'academic_keywords': ['جامعة', 'كلية', 'بحث', 'دراسة', 'أكاديمي', 'تعليم', 'محاضرة'],
                'exclude_patterns': [r'إعلان', r'تجاري', r'شراء'],
                'date_formats': ['%d/%m/%Y', '%Y-%m-%d'],
                'number_system': 'arabic',
                'rtl': True  # Right-to-left text
            },
            'ru': {  # Russian
                'name': 'Russian',
                'encodings': ['utf-8', 'windows-1251', 'koi8-r', 'iso-8859-5'],
                'pdf_patterns': [r'\.pdf$', r'документ\.pdf', r'статья\.pdf', r'лекция\.pdf'],
                'academic_keywords': ['университет', 'институт', 'исследование', 'статья', 'академический', 'образование', 'лекция'],
                'exclude_patterns': [r'реклама', r'коммерческий', r'покупка'],
                'date_formats': ['%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y'],
                'number_system': 'cyrillic'
            },
            'es': {  # Spanish
                'name': 'Spanish',
                'encodings': ['utf-8', 'iso-8859-1', 'windows-1252'],
                'pdf_patterns': [r'\.pdf$', r'documento\.pdf', r'artículo\.pdf', r'conferencia\.pdf'],
                'academic_keywords': ['universidad', 'instituto', 'investigación', 'artículo', 'académico', 'educación', 'conferencia'],
                'exclude_patterns': [r'publicidad', r'comercial', r'comprar'],
                'date_formats': ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'],
                'number_system': 'latin'
            },
            'fr': {  # French
                'name': 'French',
                'encodings': ['utf-8', 'iso-8859-1', 'windows-1252'],
                'pdf_patterns': [r'\.pdf$', r'document\.pdf', r'article\.pdf', r'conférence\.pdf'],
                'academic_keywords': ['université', 'institut', 'recherche', 'article', 'académique', 'éducation', 'conférence'],
                'exclude_patterns': [r'publicité', r'commercial', r'acheter'],
                'date_formats': ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'],
                'number_system': 'latin'
            },
            'de': {  # German
                'name': 'German',
                'encodings': ['utf-8', 'iso-8859-1', 'windows-1252'],
                'pdf_patterns': [r'\.pdf$', r'dokument\.pdf', r'artikel\.pdf', r'vorlesung\.pdf'],
                'academic_keywords': ['universität', 'institut', 'forschung', 'artikel', 'akademisch', 'bildung', 'vorlesung'],
                'exclude_patterns': [r'werbung', r'kommerziell', r'kaufen'],
                'date_formats': ['%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y'],
                'number_system': 'latin'
            }
        }
    
    def _load_regional_databases(self) -> Dict[str, Dict[str, Any]]:
        """Load regional academic database configurations."""
        return {
            'china': {
                'name': 'Chinese Academic Databases',
                'databases': [
                    {
                        'name': 'CNKI (China National Knowledge Infrastructure)',
                        'url': 'https://www.cnki.net/',
                        'language': 'zh',
                        'subjects': ['engineering', 'science', 'medicine', 'humanities'],
                        'api_available': False
                    },
                    {
                        'name': 'Wanfang Data',
                        'url': 'https://www.wanfangdata.com.cn/',
                        'language': 'zh',
                        'subjects': ['science', 'technology', 'medicine'],
                        'api_available': False
                    }
                ]
            },
            'japan': {
                'name': 'Japanese Academic Databases',
                'databases': [
                    {
                        'name': 'CiNii (NII Academic Content Portal)',
                        'url': 'https://ci.nii.ac.jp/',
                        'language': 'ja',
                        'subjects': ['all'],
                        'api_available': True
                    },
                    {
                        'name': 'J-STAGE',
                        'url': 'https://www.jstage.jst.go.jp/',
                        'language': 'ja',
                        'subjects': ['science', 'technology', 'medicine'],
                        'api_available': True
                    }
                ]
            },
            'korea': {
                'name': 'Korean Academic Databases',
                'databases': [
                    {
                        'name': 'KISS (Korean studies Information Service System)',
                        'url': 'https://kiss.kstudy.com/',
                        'language': 'ko',
                        'subjects': ['humanities', 'social_sciences'],
                        'api_available': False
                    },
                    {
                        'name': 'RISS (Research Information Sharing Service)',
                        'url': 'http://www.riss.kr/',
                        'language': 'ko',
                        'subjects': ['all'],
                        'api_available': False
                    }
                ]
            },
            'europe': {
                'name': 'European Academic Databases',
                'databases': [
                    {
                        'name': 'HAL (Hyper Articles en Ligne)',
                        'url': 'https://hal.archives-ouvertes.fr/',
                        'language': 'fr',
                        'subjects': ['all'],
                        'api_available': True
                    },
                    {
                        'name': 'SSOAR (Social Science Open Access Repository)',
                        'url': 'https://www.ssoar.info/',
                        'language': 'de',
                        'subjects': ['social_sciences'],
                        'api_available': True
                    }
                ]
            },
            'latin_america': {
                'name': 'Latin American Academic Databases',
                'databases': [
                    {
                        'name': 'SciELO',
                        'url': 'https://scielo.org/',
                        'language': 'es',
                        'subjects': ['science', 'medicine', 'humanities'],
                        'api_available': True
                    },
                    {
                        'name': 'Redalyc',
                        'url': 'https://www.redalyc.org/',
                        'language': 'es',
                        'subjects': ['all'],
                        'api_available': False
                    }
                ]
            }
        }
    
    def _load_encoding_mappings(self) -> Dict[str, List[str]]:
        """Load character encoding mappings by region."""
        return {
            'east_asia': ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5', 'shift_jis', 'euc-jp', 'euc-kr', 'cp949'],
            'middle_east': ['utf-8', 'iso-8859-6', 'windows-1256', 'iso-8859-8', 'windows-1255'],
            'eastern_europe': ['utf-8', 'windows-1251', 'koi8-r', 'iso-8859-5', 'windows-1250', 'iso-8859-2'],
            'western_europe': ['utf-8', 'iso-8859-1', 'windows-1252', 'iso-8859-15'],
            'latin_america': ['utf-8', 'iso-8859-1', 'windows-1252']
        }
    
    async def detect_encoding(self, content_bytes: bytes) -> str:
        """
        Detect character encoding of content.
        
        Args:
            content_bytes: Raw content bytes
            
        Returns:
            Detected encoding name
        """
        try:
            # Use chardet for automatic detection
            detection = chardet.detect(content_bytes)
            encoding = detection.get('encoding', 'utf-8')
            confidence = detection.get('confidence', 0.0)
            
            self.logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                self.logger.warning(f"Low confidence encoding detection, falling back to utf-8")
                return 'utf-8'
            
            return encoding.lower()
            
        except Exception as e:
            self.logger.error(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    async def fetch_with_encoding_detection(self, url: str) -> Tuple[str, str]:
        """
        Fetch content with automatic encoding detection.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (content_text, detected_encoding)
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return "", "utf-8"
                
                content_bytes = await response.read()
                
                # Try to get encoding from headers first
                content_type = response.headers.get('content-type', '')
                charset_match = re.search(r'charset=([^;]+)', content_type)
                
                if charset_match:
                    declared_encoding = charset_match.group(1).strip().lower()
                    try:
                        content_text = content_bytes.decode(declared_encoding)
                        return content_text, declared_encoding
                    except UnicodeDecodeError:
                        self.logger.warning(f"Declared encoding {declared_encoding} failed, detecting...")
                
                # Detect encoding automatically
                detected_encoding = await self.detect_encoding(content_bytes)
                
                try:
                    content_text = content_bytes.decode(detected_encoding)
                    return content_text, detected_encoding
                except UnicodeDecodeError:
                    # Final fallback with error handling
                    content_text = content_bytes.decode('utf-8', errors='replace')
                    return content_text, 'utf-8'
                    
        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {e}")
            return "", "utf-8"
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Try to import langdetect
            from langdetect import detect, detect_langs
            
            # Detect language with confidence
            detections = detect_langs(text)
            
            if detections:
                best_detection = detections[0]
                return best_detection.lang, best_detection.prob
            else:
                return 'en', 0.0
                
        except ImportError:
            self.logger.warning("langdetect not available, using keyword-based detection")
            return self._detect_language_by_keywords(text)
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return 'en', 0.0
    
    def _detect_language_by_keywords(self, text: str) -> Tuple[str, float]:
        """Fallback language detection using keywords."""
        text_lower = text.lower()
        
        # Count language-specific keywords
        language_scores = {}
        
        for lang_code, config in self.language_configs.items():
            score = 0
            keywords = config.get('academic_keywords', [])
            
            for keyword in keywords:
                score += text_lower.count(keyword.lower())
            
            if score > 0:
                language_scores[lang_code] = score
        
        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            max_score = language_scores[best_lang]
            confidence = min(max_score / 10.0, 1.0)  # Normalize to 0-1
            return best_lang, confidence
        
        return 'en', 0.0
    
    def get_language_specific_patterns(self, language: str) -> Dict[str, Any]:
        """
        Get language-specific scraping patterns.
        
        Args:
            language: Language code (e.g., 'zh', 'ja', 'ko')
            
        Returns:
            Dictionary with language-specific patterns
        """
        return self.language_configs.get(language, self.language_configs.get('en', {}))
    
    async def scrape_international_source(self, source_config: SourceConfig) -> List[str]:
        """
        Scrape an international source with language-aware processing.
        
        Args:
            source_config: Source configuration
            
        Returns:
            List of discovered PDF URLs
        """
        pdf_urls = []
        
        try:
            # Fetch content with encoding detection
            content, encoding = await self.fetch_with_encoding_detection(source_config.base_url)
            
            if not content:
                return pdf_urls
            
            # Detect language
            language, confidence = self.detect_language(content)
            self.logger.info(f"Detected language: {language} (confidence: {confidence:.2f})")
            
            # Get language-specific patterns
            lang_patterns = self.get_language_specific_patterns(language)
            
            # Parse content with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find PDF links using language-specific patterns
            pdf_patterns = lang_patterns.get('pdf_patterns', [r'\.pdf$'])
            exclude_patterns = lang_patterns.get('exclude_patterns', [])
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(source_config.base_url, href)
                
                # Check if matches PDF patterns
                matches_pdf = any(re.search(pattern, href, re.IGNORECASE) for pattern in pdf_patterns)
                
                # Check if should be excluded
                should_exclude = any(re.search(pattern, href, re.IGNORECASE) for pattern in exclude_patterns)
                
                if matches_pdf and not should_exclude:
                    # Additional validation using link text
                    link_text = link.get_text().strip()
                    if self._is_academic_link(link_text, language):
                        pdf_urls.append(absolute_url)
            
            # Look for additional patterns in page content
            additional_urls = self._extract_embedded_pdf_urls(content, source_config.base_url, language)
            pdf_urls.extend(additional_urls)
            
        except Exception as e:
            self.logger.error(f"Error scraping international source {source_config.base_url}: {e}")
        
        return list(set(pdf_urls))  # Remove duplicates
    
    def _is_academic_link(self, link_text: str, language: str) -> bool:
        """
        Check if link text indicates academic content.
        
        Args:
            link_text: Text content of the link
            language: Detected language
            
        Returns:
            True if appears to be academic content
        """
        if not link_text:
            return True  # Assume academic if no text
        
        lang_config = self.language_configs.get(language, {})
        academic_keywords = lang_config.get('academic_keywords', [])
        exclude_keywords = lang_config.get('exclude_patterns', [])
        
        link_text_lower = link_text.lower()
        
        # Check for academic keywords
        has_academic = any(keyword.lower() in link_text_lower for keyword in academic_keywords)
        
        # Check for exclude patterns
        has_exclude = any(re.search(pattern, link_text_lower) for pattern in exclude_keywords)
        
        # Default to academic if no specific indicators
        return has_academic or not has_exclude
    
    def _extract_embedded_pdf_urls(self, content: str, base_url: str, language: str) -> List[str]:
        """
        Extract PDF URLs from JavaScript or embedded content.
        
        Args:
            content: Page content
            base_url: Base URL for relative links
            language: Detected language
            
        Returns:
            List of additional PDF URLs
        """
        urls = []
        
        try:
            # Look for PDF URLs in JavaScript
            js_pdf_pattern = r'["\']([^"\']*\.pdf[^"\']*)["\']'
            js_matches = re.findall(js_pdf_pattern, content, re.IGNORECASE)
            
            for match in js_matches:
                absolute_url = urljoin(base_url, match)
                urls.append(absolute_url)
            
            # Look for data attributes
            data_pdf_pattern = r'data-[^=]*=["\']([^"\']*\.pdf[^"\']*)["\']'
            data_matches = re.findall(data_pdf_pattern, content, re.IGNORECASE)
            
            for match in data_matches:
                absolute_url = urljoin(base_url, match)
                urls.append(absolute_url)
            
        except Exception as e:
            self.logger.error(f"Error extracting embedded PDF URLs: {e}")
        
        return urls
    
    def classify_cultural_context(self, metadata: Dict[str, Any], language: str) -> Dict[str, Any]:
        """
        Add cultural context awareness to content classification.
        
        Args:
            metadata: Existing metadata
            language: Detected language
            
        Returns:
            Enhanced metadata with cultural context
        """
        cultural_context = {}
        
        # Get language configuration
        lang_config = self.language_configs.get(language, {})
        
        # Add language information
        cultural_context['language'] = language
        cultural_context['language_name'] = lang_config.get('name', language)
        cultural_context['writing_system'] = self._get_writing_system(language)
        cultural_context['text_direction'] = 'rtl' if lang_config.get('rtl', False) else 'ltr'
        
        # Academic level classification adjustments
        if language in ['zh', 'ja', 'ko']:
            # East Asian academic systems
            cultural_context['academic_system'] = 'east_asian'
            cultural_context['grade_levels'] = self._map_east_asian_levels(metadata.get('title', ''))
        elif language == 'ar':
            # Arabic academic systems
            cultural_context['academic_system'] = 'arabic'
            cultural_context['grade_levels'] = self._map_arabic_levels(metadata.get('title', ''))
        else:
            # Western academic systems
            cultural_context['academic_system'] = 'western'
            cultural_context['grade_levels'] = self._map_western_levels(metadata.get('title', ''))
        
        # Subject area adjustments
        cultural_context['subject_classification'] = self._classify_subject_culturally(
            metadata.get('title', ''), 
            metadata.get('abstract', ''), 
            language
        )
        
        return cultural_context
    
    def _get_writing_system(self, language: str) -> str:
        """Get writing system for language."""
        writing_systems = {
            'zh': 'chinese_characters',
            'ja': 'japanese_mixed',  # Hiragana, Katakana, Kanji
            'ko': 'hangul',
            'ar': 'arabic_script',
            'ru': 'cyrillic',
            'th': 'thai_script',
            'hi': 'devanagari'
        }
        return writing_systems.get(language, 'latin')
    
    def _map_east_asian_levels(self, title: str) -> List[str]:
        """Map East Asian academic level indicators."""
        levels = []
        title_lower = title.lower()
        
        # Chinese/Japanese/Korean level indicators
        level_indicators = {
            'elementary': ['小学', '초등', '小學'],
            'middle_school': ['中学', '중학', '中學'],
            'high_school': ['高中', '고등학교', '高校'],
            'undergraduate': ['本科', '학부', '大学'],
            'graduate': ['研究生', '대학원', '修士'],
            'doctoral': ['博士', '박사', '博士']
        }
        
        for level, indicators in level_indicators.items():
            if any(indicator in title for indicator in indicators):
                levels.append(level)
        
        return levels if levels else ['unknown']
    
    def _map_arabic_levels(self, title: str) -> List[str]:
        """Map Arabic academic level indicators."""
        levels = []
        
        # Arabic level indicators
        level_indicators = {
            'elementary': ['ابتدائي', 'الابتدائية'],
            'middle_school': ['متوسط', 'المتوسطة'],
            'high_school': ['ثانوي', 'الثانوية'],
            'undergraduate': ['بكالوريوس', 'الجامعة'],
            'graduate': ['ماجستير', 'الدراسات العليا'],
            'doctoral': ['دكتوراه', 'الدكتوراه']
        }
        
        for level, indicators in level_indicators.items():
            if any(indicator in title for indicator in indicators):
                levels.append(level)
        
        return levels if levels else ['unknown']
    
    def _map_western_levels(self, title: str) -> List[str]:
        """Map Western academic level indicators."""
        levels = []
        title_lower = title.lower()
        
        # Western level indicators
        level_indicators = {
            'elementary': ['elementary', 'primary', 'grade 1', 'grade 2', 'grade 3', 'grade 4', 'grade 5'],
            'middle_school': ['middle school', 'grade 6', 'grade 7', 'grade 8'],
            'high_school': ['high school', 'secondary', 'grade 9', 'grade 10', 'grade 11', 'grade 12'],
            'undergraduate': ['undergraduate', 'bachelor', 'college', 'university'],
            'graduate': ['graduate', 'master', 'msc', 'ma'],
            'doctoral': ['doctoral', 'phd', 'doctorate']
        }
        
        for level, indicators in level_indicators.items():
            if any(indicator in title_lower for indicator in indicators):
                levels.append(level)
        
        return levels if levels else ['unknown']
    
    def _classify_subject_culturally(self, title: str, abstract: str, language: str) -> Dict[str, Any]:
        """Classify subject area with cultural context."""
        content = f"{title} {abstract}".lower()
        
        # Base subject classification
        subjects = []
        
        # Universal subjects (translated keywords)
        universal_subjects = {
            'mathematics': {
                'en': ['math', 'mathematics', 'algebra', 'calculus', 'geometry'],
                'zh': ['数学', '代数', '几何', '微积分'],
                'ja': ['数学', '代数', '幾何', '微積分'],
                'ko': ['수학', '대수', '기하', '미적분'],
                'ar': ['رياضيات', 'جبر', 'هندسة', 'تفاضل'],
                'ru': ['математика', 'алгебра', 'геометрия', 'исчисление'],
                'es': ['matemáticas', 'álgebra', 'geometría', 'cálculo'],
                'fr': ['mathématiques', 'algèbre', 'géométrie', 'calcul'],
                'de': ['mathematik', 'algebra', 'geometrie', 'kalkül']
            },
            'computer_science': {
                'en': ['computer', 'programming', 'algorithm', 'software'],
                'zh': ['计算机', '编程', '算法', '软件'],
                'ja': ['コンピュータ', 'プログラミング', 'アルゴリズム', 'ソフトウェア'],
                'ko': ['컴퓨터', '프로그래밍', '알고리즘', '소프트웨어'],
                'ar': ['حاسوب', 'برمجة', 'خوارزمية', 'برمجيات'],
                'ru': ['компьютер', 'программирование', 'алгоритм', 'программное обеспечение'],
                'es': ['computadora', 'programación', 'algoritmo', 'software'],
                'fr': ['ordinateur', 'programmation', 'algorithme', 'logiciel'],
                'de': ['computer', 'programmierung', 'algorithmus', 'software']
            }
        }
        
        # Check for subject keywords in detected language
        for subject, translations in universal_subjects.items():
            keywords = translations.get(language, translations.get('en', []))
            if any(keyword in content for keyword in keywords):
                subjects.append(subject)
        
        # Cultural-specific subjects
        cultural_subjects = self._get_cultural_subjects(content, language)
        subjects.extend(cultural_subjects)
        
        return {
            'subjects': list(set(subjects)) if subjects else ['general'],
            'cultural_context': language,
            'classification_method': 'cultural_aware'
        }
    
    def _get_cultural_subjects(self, content: str, language: str) -> List[str]:
        """Get culture-specific academic subjects."""
        cultural_subjects = []
        
        if language in ['zh', 'ja', 'ko']:
            # East Asian specific subjects
            east_asian_subjects = {
                'traditional_medicine': ['中医', '漢方', '한의학'],
                'calligraphy': ['书法', '書道', '서예'],
                'martial_arts': ['武术', '武道', '무술'],
                'confucian_studies': ['儒学', '儒教', '유교']
            }
            
            for subject, keywords in east_asian_subjects.items():
                if any(keyword in content for keyword in keywords):
                    cultural_subjects.append(subject)
        
        elif language == 'ar':
            # Arabic/Islamic specific subjects
            arabic_subjects = {
                'islamic_studies': ['الدراسات الإسلامية', 'الفقه', 'التفسير'],
                'arabic_literature': ['الأدب العربي', 'الشعر العربي'],
                'islamic_law': ['الشريعة', 'القانون الإسلامي']
            }
            
            for subject, keywords in arabic_subjects.items():
                if any(keyword in content for keyword in keywords):
                    cultural_subjects.append(subject)
        
        return cultural_subjects
    
    async def get_regional_databases(self, region: str) -> List[Dict[str, Any]]:
        """
        Get academic databases for a specific region.
        
        Args:
            region: Region identifier (e.g., 'china', 'japan', 'europe')
            
        Returns:
            List of database configurations
        """
        return self.regional_databases.get(region, {}).get('databases', [])
    
    async def search_regional_database(self, database_config: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Search a regional academic database.
        
        Args:
            database_config: Database configuration
            query: Search query
            
        Returns:
            List of search results
        """
        results = []
        
        try:
            if database_config.get('api_available', False):
                results = await self._search_database_api(database_config, query)
            else:
                results = await self._search_database_web(database_config, query)
                
        except Exception as e:
            self.logger.error(f"Error searching database {database_config.get('name', 'unknown')}: {e}")
        
        return results
    
    async def _search_database_api(self, database_config: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Search database using API."""
        # Implementation would depend on specific database APIs
        # This is a placeholder for API-based searches
        self.logger.info(f"API search not implemented for {database_config.get('name')}")
        return []
    
    async def _search_database_web(self, database_config: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Search database using web scraping."""
        results = []
        
        try:
            base_url = database_config.get('url', '')
            language = database_config.get('language', 'en')
            
            # Create search URL (this would need to be customized per database)
            search_url = f"{base_url}/search?q={quote(query)}"
            
            # Fetch with encoding detection
            content, encoding = await self.fetch_with_encoding_detection(search_url)
            
            if content:
                # Parse results (implementation would be database-specific)
                soup = BeautifulSoup(content, 'html.parser')
                
                # Generic result parsing
                result_items = soup.find_all(['div', 'li'], class_=re.compile(r'result|item|paper'))
                
                for item in result_items[:10]:  # Limit results
                    result = self._parse_database_result(item, base_url, language)
                    if result:
                        results.append(result)
                        
        except Exception as e:
            self.logger.error(f"Error in web search: {e}")
        
        return results
    
    def _parse_database_result(self, item_soup: BeautifulSoup, base_url: str, language: str) -> Optional[Dict[str, Any]]:
        """Parse a single database search result."""
        try:
            result = {
                'source': 'regional_database',
                'language': language
            }
            
            # Title
            title_elem = item_soup.find(['h1', 'h2', 'h3', 'h4', 'a'])
            if title_elem:
                result['title'] = title_elem.get_text().strip()
            
            # PDF link
            pdf_link = item_soup.find('a', href=re.compile(r'\.pdf'))
            if pdf_link:
                result['pdf_url'] = urljoin(base_url, pdf_link['href'])
            
            # Authors
            author_elem = item_soup.find(['div', 'span'], class_=re.compile(r'author'))
            if author_elem:
                result['authors'] = [author.strip() for author in author_elem.get_text().split(',')]
            
            return result if result.get('title') else None
            
        except Exception as e:
            self.logger.error(f"Error parsing database result: {e}")
            return None