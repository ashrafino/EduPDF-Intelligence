"""
Content analysis system for educational PDFs.
Implements keyword extraction, topic classification, language detection, and subject area classification.
"""

import logging
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import langdetect
from langdetect import detect, detect_langs

# Data models
from data.models import PDFMetadata


class ContentAnalyzer:
    """
    Comprehensive content analysis system for educational PDFs.
    Implements requirements 3.2 and 3.4 for content analysis and classification.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        self._download_nltk_data()
        
        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Subject area keywords for classification
        self.subject_keywords = {
            'computer_science': [
                'algorithm', 'programming', 'software', 'computer', 'data structure',
                'machine learning', 'artificial intelligence', 'database', 'network',
                'cybersecurity', 'web development', 'python', 'java', 'javascript'
            ],
            'mathematics': [
                'calculus', 'algebra', 'geometry', 'statistics', 'probability',
                'differential', 'integral', 'matrix', 'theorem', 'proof',
                'equation', 'function', 'derivative', 'limit'
            ],
            'physics': [
                'mechanics', 'thermodynamics', 'electromagnetism', 'quantum',
                'relativity', 'wave', 'particle', 'energy', 'force', 'motion',
                'field', 'radiation', 'optics', 'nuclear'
            ],
            'chemistry': [
                'molecule', 'atom', 'reaction', 'compound', 'element',
                'organic', 'inorganic', 'biochemistry', 'catalyst', 'bond',
                'periodic table', 'solution', 'acid', 'base'
            ],
            'biology': [
                'cell', 'genetics', 'evolution', 'ecology', 'anatomy',
                'physiology', 'molecular biology', 'organism', 'species',
                'dna', 'rna', 'protein', 'enzyme', 'metabolism'
            ],
            'engineering': [
                'design', 'system', 'mechanical', 'electrical', 'civil',
                'chemical engineering', 'materials', 'manufacturing', 'control',
                'optimization', 'simulation', 'modeling', 'analysis'
            ],
            'business': [
                'management', 'marketing', 'finance', 'economics', 'accounting',
                'strategy', 'leadership', 'organization', 'entrepreneurship',
                'investment', 'market', 'business model', 'revenue'
            ],
            'psychology': [
                'behavior', 'cognitive', 'social psychology', 'development',
                'personality', 'learning', 'memory', 'perception', 'emotion',
                'therapy', 'mental health', 'research methods'
            ],
            'history': [
                'historical', 'ancient', 'medieval', 'modern', 'civilization',
                'culture', 'society', 'political', 'economic history',
                'war', 'revolution', 'empire', 'dynasty'
            ],
            'literature': [
                'novel', 'poetry', 'drama', 'literary analysis', 'author',
                'narrative', 'character', 'theme', 'symbolism', 'criticism',
                'genre', 'style', 'rhetoric', 'composition'
            ]
        }
        
        # Initialize ML models
        self.topic_classifier = None
        self.tfidf_vectorizer = None
        self._initialize_models()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def _initialize_models(self):
        """Initialize machine learning models for topic classification."""
        # Create a simple TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Initialize Naive Bayes classifier
        self.topic_classifier = Pipeline([
            ('tfidf', self.tfidf_vectorizer),
            ('classifier', MultinomialNB())
        ])
        
        # Train with sample data (in a real implementation, this would use a larger dataset)
        self._train_sample_classifier()
    
    def _train_sample_classifier(self):
        """Train classifier with sample educational content."""
        # Sample training data for demonstration
        sample_texts = []
        sample_labels = []
        
        # Generate sample texts for each subject
        for subject, keywords in self.subject_keywords.items():
            for i in range(5):  # 5 samples per subject
                # Create synthetic text using keywords
                text = f"This document discusses {' and '.join(keywords[:3])}. "
                text += f"The main topics include {', '.join(keywords[3:6])}. "
                text += f"Students will learn about {' '.join(keywords[6:9])}."
                
                sample_texts.append(text)
                sample_labels.append(subject)
        
        try:
            # Train the classifier
            self.topic_classifier.fit(sample_texts, sample_labels)
            self.logger.info("Topic classifier trained successfully")
        except Exception as e:
            self.logger.warning(f"Could not train topic classifier: {e}")
    
    def analyze_content(self, text_content: str, metadata: PDFMetadata) -> PDFMetadata:
        """
        Perform comprehensive content analysis on PDF text.
        
        Args:
            text_content: Extracted text from PDF
            metadata: PDFMetadata object to update
            
        Returns:
            Updated PDFMetadata with analysis results
        """
        self.logger.info(f"Analyzing content for: {metadata.filename}")
        
        try:
            # Extract keywords using TF-IDF
            metadata.keywords = self.extract_keywords(text_content)
            
            # Detect language with confidence
            lang_result = self.detect_language_with_confidence(text_content)
            metadata.language = lang_result[0]
            metadata.language_confidence = lang_result[1]
            
            # Classify subject area
            metadata.subject_area = self.classify_subject_area(text_content)
            
            # Perform topic classification using ML
            topic_scores = self.classify_topics_ml(text_content)
            if topic_scores:
                # Add top topics as tags
                metadata.tags = [topic for topic, score in topic_scores[:5]]
            
            self.logger.info(f"Content analysis completed for: {metadata.filename}")
            
        except Exception as e:
            error_msg = f"Error analyzing content for {metadata.filename}: {str(e)}"
            self.logger.error(error_msg)
            metadata.processing_errors.append(error_msg)
        
        return metadata
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords using TF-IDF and frequency analysis.
        
        Args:
            text: Input text content
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            # Clean and tokenize text
            cleaned_text = self._clean_text(text)
            tokens = word_tokenize(cleaned_text.lower())
            
            # Filter tokens
            filtered_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if (token.isalpha() and 
                    len(token) > 2 and 
                    token not in self.stop_words)
            ]
            
            # Count frequency
            token_freq = Counter(filtered_tokens)
            
            # Get most common tokens
            common_tokens = token_freq.most_common(max_keywords * 2)
            
            # Use TF-IDF for better keyword extraction
            try:
                # Create a simple TF-IDF analysis
                sentences = sent_tokenize(text)
                if len(sentences) > 1:
                    vectorizer = TfidfVectorizer(
                        max_features=max_keywords,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                    tfidf_matrix = vectorizer.fit_transform(sentences)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get average TF-IDF scores
                    mean_scores = tfidf_matrix.mean(axis=0).A1
                    keyword_scores = list(zip(feature_names, mean_scores))
                    keyword_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    return [kw for kw, score in keyword_scores[:max_keywords]]
            
            except Exception as e:
                self.logger.warning(f"TF-IDF extraction failed, using frequency: {e}")
            
            # Fallback to frequency-based extraction
            return [token for token, freq in common_tokens[:max_keywords]]
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def detect_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detect language with confidence scoring using langdetect.
        
        Args:
            text: Input text content
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 50:
            return ("en", 0.0)
        
        try:
            # Use langdetect for language detection
            lang_probs = detect_langs(text)
            
            if lang_probs:
                best_lang = lang_probs[0]
                return (best_lang.lang, best_lang.prob)
            else:
                return ("en", 0.0)
                
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return ("en", 0.0)
    
    def classify_subject_area(self, text: str) -> str:
        """
        Classify subject area based on keyword analysis.
        
        Args:
            text: Input text content
            
        Returns:
            Classified subject area
        """
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        subject_scores = {}
        
        # Score each subject based on keyword matches
        for subject, keywords in self.subject_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = text_lower.count(keyword.lower())
                score += count
                
                # Bonus for exact phrase matches
                if keyword.lower() in text_lower:
                    score += 2
            
            subject_scores[subject] = score
        
        # Return subject with highest score
        if subject_scores:
            best_subject = max(subject_scores.items(), key=lambda x: x[1])
            if best_subject[1] > 0:
                return best_subject[0]
        
        return "unknown"
    
    def classify_topics_ml(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify topics using machine learning models.
        
        Args:
            text: Input text content
            
        Returns:
            List of (topic, confidence_score) tuples
        """
        if not text or not self.topic_classifier:
            return []
        
        try:
            # Predict probabilities for each class
            probabilities = self.topic_classifier.predict_proba([text])[0]
            classes = self.topic_classifier.classes_
            
            # Create topic-score pairs
            topic_scores = list(zip(classes, probabilities))
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top topics with confidence > 0.1
            return [(topic, float(score)) for topic, score in topic_scores if score > 0.1]
            
        except Exception as e:
            self.logger.warning(f"ML topic classification failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for analysis by removing special characters and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def generate_content_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a brief summary of the content.
        
        Args:
            text: Input text content
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Content summary
        """
        if not text or len(text.strip()) < 100:
            return ""
        
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return text[:500] + "..." if len(text) > 500 else text
            
            # Simple extractive summarization - take first few sentences
            # In a more advanced implementation, this could use TF-IDF or other methods
            summary_sentences = sentences[:max_sentences]
            summary = " ".join(summary_sentences)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return text[:500] + "..." if len(text) > 500 else text