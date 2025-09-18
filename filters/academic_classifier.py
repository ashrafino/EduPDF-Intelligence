"""
Academic level classifier for educational PDF content.
Implements text complexity analysis, keyword-based classification, and ML models.
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from data.models import PDFMetadata, AcademicLevel


@dataclass
class ClassificationResult:
    """Result of academic level classification."""
    predicted_level: AcademicLevel
    confidence: float
    complexity_score: float
    readability_score: float
    features: Dict[str, float]
    reasoning: List[str]


class ReadabilityAnalyzer:
    """Analyzes text complexity using various readability metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score.
        Higher scores indicate easier readability.
        """
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score))
    
    def flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.
        Higher scores indicate higher grade level required.
        """
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        grade = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        return max(0.0, grade)
    
    def automated_readability_index(self, text: str) -> float:
        """
        Calculate Automated Readability Index (ARI).
        Estimates grade level needed to comprehend text.
        """
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        characters = len(re.sub(r'\s+', '', text))
        
        if sentences == 0 or words == 0:
            return 0.0
        
        ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43
        return max(0.0, ari)
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentence_endings = re.findall(r'[.!?]+', text)
        return max(1, len(sentence_endings))
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count in text."""
        words = re.findall(r'\b\w+\b', text.lower())
        total_syllables = 0
        
        for word in words:
            syllables = self._syllables_in_word(word)
            total_syllables += syllables
        
        return max(1, total_syllables)
    
    def _syllables_in_word(self, word: str) -> int:
        """Estimate syllables in a single word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)


class AcademicLevelClassifier:
    """
    Classifier for determining academic level of educational content.
    Uses readability metrics, keyword analysis, and machine learning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.readability_analyzer = ReadabilityAnalyzer()
        
        # Academic level keywords and indicators
        self.level_keywords = {
            AcademicLevel.HIGH_SCHOOL: {
                'course_codes': [r'\b(9|10|11|12)th\b', r'\bgrade\s*(9|10|11|12)\b', 
                               r'\bhigh\s*school\b', r'\bsecondary\b'],
                'subjects': ['algebra', 'geometry', 'biology', 'chemistry', 'physics',
                           'world history', 'us history', 'english', 'literature'],
                'complexity_terms': ['basic', 'introduction', 'fundamentals', 'overview',
                                   'beginner', 'elementary', 'simple'],
                'indicators': ['homework', 'quiz', 'test prep', 'study guide', 'review']
            },
            AcademicLevel.UNDERGRADUATE: {
                'course_codes': [r'\b[A-Z]{2,4}\s*[1-4]\d{2}\b', r'\bundergrad\b',
                               r'\bbachelor\b', r'\bb\.?s\.?\b', r'\bb\.?a\.?\b'],
                'subjects': ['calculus', 'statistics', 'organic chemistry', 'psychology',
                           'economics', 'computer science', 'engineering', 'sociology'],
                'complexity_terms': ['intermediate', 'analysis', 'theory', 'principles',
                                   'concepts', 'methods', 'applications'],
                'indicators': ['midterm', 'final exam', 'project', 'lab report', 'essay']
            },
            AcademicLevel.GRADUATE: {
                'course_codes': [r'\b[A-Z]{2,4}\s*[5-9]\d{2}\b', r'\bgrad\b', r'\bmaster\b',
                               r'\bm\.?s\.?\b', r'\bm\.?a\.?\b', r'\bphd\b', r'\bdoctoral\b'],
                'subjects': ['advanced', 'research methods', 'thesis', 'dissertation',
                           'seminar', 'independent study', 'practicum'],
                'complexity_terms': ['advanced', 'sophisticated', 'complex', 'theoretical',
                                   'empirical', 'methodology', 'framework', 'paradigm'],
                'indicators': ['research', 'publication', 'conference', 'peer review',
                             'citation', 'bibliography']
            }
        }
        
        # Readability thresholds for academic levels
        self.readability_thresholds = {
            AcademicLevel.HIGH_SCHOOL: {
                'flesch_ease': (30, 70),      # Moderate difficulty
                'grade_level': (8, 12),       # 8th-12th grade
                'ari': (8, 12)
            },
            AcademicLevel.UNDERGRADUATE: {
                'flesch_ease': (10, 50),      # Difficult
                'grade_level': (12, 16),      # 12th-16th grade
                'ari': (12, 16)
            },
            AcademicLevel.GRADUATE: {
                'flesch_ease': (0, 30),       # Very difficult
                'grade_level': (16, 25),      # 16th+ grade
                'ari': (16, 25)
            }
        }
        
        # ML model for classification (will be trained)
        self.ml_model = None
        self.vectorizer = None
        self._initialize_ml_model()
    
    def classify_academic_level(self, metadata: PDFMetadata, 
                              content_text: Optional[str] = None) -> ClassificationResult:
        """
        Classify the academic level of educational content.
        
        Args:
            metadata: PDF metadata object
            content_text: Optional extracted text content
            
        Returns:
            ClassificationResult with predicted level and analysis
        """
        features = {}
        reasoning = []
        
        # Collect text for analysis
        text_sources = []
        if metadata.title:
            text_sources.append(metadata.title)
        if metadata.description:
            text_sources.append(metadata.description)
        if metadata.course_code:
            text_sources.append(metadata.course_code)
        if content_text:
            # Use first 2000 characters for analysis
            text_sources.append(content_text[:2000])
        
        combined_text = ' '.join(text_sources)
        
        if not combined_text.strip():
            return ClassificationResult(
                predicted_level=AcademicLevel.UNKNOWN,
                confidence=0.0,
                complexity_score=0.0,
                readability_score=0.0,
                features={},
                reasoning=["No text content available for analysis"]
            )
        
        # 1. Readability analysis
        readability_scores = self._analyze_readability(combined_text)
        features.update(readability_scores)
        
        # 2. Keyword-based classification
        keyword_scores = self._analyze_keywords(combined_text, metadata)
        features.update(keyword_scores)
        
        # 3. Course code pattern recognition
        course_level = self._analyze_course_codes(combined_text, metadata)
        features['course_level_score'] = course_level
        
        # 4. ML-based classification (if model is available)
        ml_prediction = self._ml_classify(combined_text)
        if ml_prediction:
            features['ml_confidence'] = ml_prediction[1]
        
        # Combine all evidence to make final prediction
        predicted_level, confidence = self._combine_predictions(
            readability_scores, keyword_scores, course_level, ml_prediction
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, predicted_level)
        
        # Calculate overall complexity score
        complexity_score = self._calculate_complexity_score(features)
        
        return ClassificationResult(
            predicted_level=predicted_level,
            confidence=confidence,
            complexity_score=complexity_score,
            readability_score=features.get('flesch_ease', 0.0),
            features=features,
            reasoning=reasoning
        )
    
    def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze text readability using multiple metrics."""
        flesch_ease = self.readability_analyzer.flesch_reading_ease(text)
        grade_level = self.readability_analyzer.flesch_kincaid_grade(text)
        ari = self.readability_analyzer.automated_readability_index(text)
        
        return {
            'flesch_ease': flesch_ease,
            'grade_level': grade_level,
            'ari': ari
        }
    
    def _analyze_keywords(self, text: str, metadata: PDFMetadata) -> Dict[str, float]:
        """Analyze keywords for academic level indicators."""
        text_lower = text.lower()
        scores = {}
        
        for level, keywords_dict in self.level_keywords.items():
            level_score = 0.0
            
            # Check course codes
            for pattern in keywords_dict['course_codes']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                level_score += matches * 0.3
            
            # Check subject keywords
            for keyword in keywords_dict['subjects']:
                if keyword.lower() in text_lower:
                    level_score += 0.2
            
            # Check complexity terms
            for term in keywords_dict['complexity_terms']:
                if term.lower() in text_lower:
                    level_score += 0.1
            
            # Check indicators
            for indicator in keywords_dict['indicators']:
                if indicator.lower() in text_lower:
                    level_score += 0.15
            
            scores[f'{level.value}_keyword_score'] = min(level_score, 1.0)
        
        return scores
    
    def _analyze_course_codes(self, text: str, metadata: PDFMetadata) -> float:
        """Analyze course codes to determine academic level."""
        course_text = text
        if metadata.course_code:
            course_text += " " + metadata.course_code
        
        # Look for numbered course patterns
        course_numbers = re.findall(r'\b[A-Z]{2,4}\s*(\d{3,4})\b', course_text, re.IGNORECASE)
        
        if not course_numbers:
            return 0.0
        
        # Analyze course number ranges
        total_score = 0.0
        for number_str in course_numbers:
            number = int(number_str)
            
            if 100 <= number <= 299:  # Typically undergraduate lower division
                total_score += 0.6
            elif 300 <= number <= 499:  # Typically undergraduate upper division
                total_score += 0.8
            elif 500 <= number <= 699:  # Typically graduate level
                total_score += 1.0
            elif 700 <= number <= 999:  # Advanced graduate
                total_score += 1.0
            elif number < 100:  # High school or remedial
                total_score += 0.3
        
        return min(total_score / len(course_numbers), 1.0)
    
    def _ml_classify(self, text: str) -> Optional[Tuple[AcademicLevel, float]]:
        """Use ML model for classification if available."""
        if not self.ml_model or not self.vectorizer:
            return None
        
        try:
            # Transform text using trained vectorizer
            text_vector = self.vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.ml_model.predict(text_vector)[0]
            probabilities = self.ml_model.predict_proba(text_vector)[0]
            confidence = max(probabilities)
            
            # Map prediction to AcademicLevel
            level_mapping = {
                0: AcademicLevel.HIGH_SCHOOL,
                1: AcademicLevel.UNDERGRADUATE,
                2: AcademicLevel.GRADUATE
            }
            
            predicted_level = level_mapping.get(prediction, AcademicLevel.UNKNOWN)
            return predicted_level, confidence
            
        except Exception as e:
            self.logger.warning(f"ML classification failed: {e}")
            return None
    
    def _combine_predictions(self, readability_scores: Dict[str, float],
                           keyword_scores: Dict[str, float],
                           course_level: float,
                           ml_prediction: Optional[Tuple[AcademicLevel, float]]) -> Tuple[AcademicLevel, float]:
        """Combine all prediction methods to make final classification."""
        
        # Score each academic level based on all evidence
        level_scores = {level: 0.0 for level in AcademicLevel if level != AcademicLevel.UNKNOWN}
        
        # 1. Readability-based scoring
        flesch_ease = readability_scores.get('flesch_ease', 50)
        grade_level = readability_scores.get('grade_level', 10)
        
        for level, thresholds in self.readability_thresholds.items():
            readability_score = 0.0
            
            # Flesch Reading Ease
            if thresholds['flesch_ease'][0] <= flesch_ease <= thresholds['flesch_ease'][1]:
                readability_score += 0.4
            
            # Grade Level
            if thresholds['grade_level'][0] <= grade_level <= thresholds['grade_level'][1]:
                readability_score += 0.4
            
            level_scores[level] += readability_score * 0.3  # 30% weight
        
        # 2. Keyword-based scoring
        for level in level_scores.keys():
            keyword_key = f'{level.value}_keyword_score'
            if keyword_key in keyword_scores:
                level_scores[level] += keyword_scores[keyword_key] * 0.4  # 40% weight
        
        # 3. Course code scoring
        if course_level > 0.5:
            level_scores[AcademicLevel.UNDERGRADUATE] += course_level * 0.2
            level_scores[AcademicLevel.GRADUATE] += course_level * 0.2
        else:
            level_scores[AcademicLevel.HIGH_SCHOOL] += (1 - course_level) * 0.2
        
        # 4. ML prediction (if available)
        if ml_prediction:
            predicted_level, ml_confidence = ml_prediction
            if predicted_level in level_scores:
                level_scores[predicted_level] += ml_confidence * 0.1  # 10% weight
        
        # Find the level with highest score
        best_level = max(level_scores.keys(), key=lambda k: level_scores[k])
        confidence = level_scores[best_level]
        
        # If confidence is too low, return UNKNOWN
        if confidence < 0.3:
            return AcademicLevel.UNKNOWN, confidence
        
        return best_level, confidence
    
    def _generate_reasoning(self, features: Dict[str, float], 
                          predicted_level: AcademicLevel) -> List[str]:
        """Generate human-readable reasoning for the classification."""
        reasoning = []
        
        # Readability reasoning
        flesch_ease = features.get('flesch_ease', 0)
        grade_level = features.get('grade_level', 0)
        
        if flesch_ease < 30:
            reasoning.append("Very difficult text (low readability score)")
        elif flesch_ease > 70:
            reasoning.append("Easy to read text (high readability score)")
        
        if grade_level > 16:
            reasoning.append("Graduate-level reading complexity")
        elif grade_level > 12:
            reasoning.append("College-level reading complexity")
        elif grade_level > 8:
            reasoning.append("High school reading complexity")
        
        # Keyword reasoning
        for level in [AcademicLevel.HIGH_SCHOOL, AcademicLevel.UNDERGRADUATE, AcademicLevel.GRADUATE]:
            keyword_key = f'{level.value}_keyword_score'
            if features.get(keyword_key, 0) > 0.3:
                reasoning.append(f"Contains {level.value.replace('_', ' ')} keywords")
        
        # Course code reasoning
        if features.get('course_level_score', 0) > 0.5:
            reasoning.append("Course numbering suggests advanced level")
        
        return reasoning
    
    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """Calculate overall complexity score (0-1)."""
        flesch_ease = features.get('flesch_ease', 50)
        grade_level = features.get('grade_level', 10)
        
        # Normalize scores
        complexity_from_flesch = max(0, (100 - flesch_ease) / 100)
        complexity_from_grade = min(1, grade_level / 20)
        
        return (complexity_from_flesch + complexity_from_grade) / 2
    
    def _initialize_ml_model(self):
        """Initialize and train a basic ML model for classification."""
        # This is a placeholder for ML model initialization
        # In a real implementation, you would train on labeled data
        try:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.ml_model = MultinomialNB()
            
            # For now, create a dummy trained model
            # In practice, you would load pre-trained model or train on real data
            dummy_texts = [
                "high school algebra basic math",
                "undergraduate calculus advanced mathematics",
                "graduate research methodology dissertation"
            ]
            dummy_labels = [0, 1, 2]  # 0=HS, 1=UG, 2=Grad
            
            X = self.vectorizer.fit_transform(dummy_texts)
            self.ml_model.fit(X, dummy_labels)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize ML model: {e}")
            self.ml_model = None
            self.vectorizer = None
    
    def train_model(self, training_data: List[Tuple[str, AcademicLevel]]):
        """
        Train the ML model with labeled training data.
        
        Args:
            training_data: List of (text, academic_level) tuples
        """
        if not training_data:
            self.logger.warning("No training data provided")
            return
        
        texts = [item[0] for item in training_data]
        labels = [self._level_to_int(item[1]) for item in training_data]
        
        try:
            # Initialize vectorizer and model
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.ml_model = MultinomialNB()
            
            # Transform texts and train model
            X = self.vectorizer.fit_transform(texts)
            self.ml_model.fit(X, labels)
            
            self.logger.info(f"ML model trained on {len(training_data)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to train ML model: {e}")
            self.ml_model = None
            self.vectorizer = None
    
    def _level_to_int(self, level: AcademicLevel) -> int:
        """Convert AcademicLevel to integer for ML model."""
        mapping = {
            AcademicLevel.HIGH_SCHOOL: 0,
            AcademicLevel.UNDERGRADUATE: 1,
            AcademicLevel.GRADUATE: 2,
            AcademicLevel.POSTGRADUATE: 2,  # Treat same as graduate
            AcademicLevel.UNKNOWN: 1  # Default to undergraduate
        }
        return mapping.get(level, 1)