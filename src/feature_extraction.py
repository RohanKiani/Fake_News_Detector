"""
Feature Extraction Module for Fake News Detector App
Handles TF-IDF vectorization, feature engineering, and text analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data (with error handling)
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")

# ==================== TF-IDF CONFIGURATION ====================
DEFAULT_TFIDF_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'stop_words': 'english',
    'lowercase': True,
    'strip_accents': 'ascii',
    'token_pattern': r'\b[a-zA-Z][a-zA-Z0-9]*\b',
    'sublinear_tf': True,
    'use_idf': True,
    'smooth_idf': True,
    'norm': 'l2'
}

# ==================== FEATURE EXTRACTION CLASSES ====================
class FakeNewsFeatureExtractor:
    """
    Advanced feature extractor for fake news detection
    Combines TF-IDF with linguistic and stylistic features
    """
    
    def __init__(self, tfidf_config: Dict = None):
        """
        Initialize the feature extractor
        
        Args:
            tfidf_config (dict): TF-IDF configuration parameters
        """
        self.tfidf_config = tfidf_config or DEFAULT_TFIDF_CONFIG
        self.tfidf_vectorizer = None
        self.sentiment_analyzer = None
        self.is_fitted = False
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Sentiment analyzer initialization warning: {e}")
            self.sentiment_analyzer = None
        
        # Common fake news indicators
        self.fake_indicators = {
            'sensational_words': [
                'shocking', 'amazing', 'unbelievable', 'incredible', 'outrageous',
                'scandal', 'exposed', 'revealed', 'secret', 'hidden', 'bombshell',
                'explosive', 'devastating', 'urgent', 'breaking', 'exclusive'
            ],
            'emotional_words': [
                'hate', 'love', 'fear', 'anger', 'rage', 'fury', 'terrified',
                'disgusted', 'outraged', 'shocked', 'appalled', 'horrified'
            ],
            'uncertainty_words': [
                'allegedly', 'reportedly', 'supposedly', 'apparently', 'presumably',
                'rumored', 'claimed', 'suggested', 'speculated', 'believed'
            ],
            'absolute_words': [
                'never', 'always', 'all', 'every', 'none', 'everyone', 'nobody',
                'everything', 'nothing', 'completely', 'totally', 'absolutely'
            ]
        }
    
    def fit(self, texts: List[str]) -> 'FakeNewsFeatureExtractor':
        """
        Fit the feature extractor on training texts
        
        Args:
            texts (List[str]): Training texts
            
        Returns:
            FakeNewsFeatureExtractor: Self for chaining
        """
        try:
            logger.info("Fitting TF-IDF vectorizer...")
            
            # Initialize and fit TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_config)
            self.tfidf_vectorizer.fit(texts)
            
            self.is_fitted = True
            logger.info(f"Feature extractor fitted on {len(texts)} texts")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting feature extractor: {e}")
            raise
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors
        
        Args:
            texts (List[str]): Texts to transform
            
        Returns:
            np.ndarray: Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        try:
            # Extract TF-IDF features
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            
            # Extract additional features
            additional_features = []
            for text in texts:
                features = self.extract_linguistic_features(text)
                additional_features.append(list(features.values()))
            
            # Combine features
            additional_features = np.array(additional_features)
            
            # Convert sparse to dense and combine
            tfidf_dense = tfidf_features.toarray()
            combined_features = np.hstack([tfidf_dense, additional_features])
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error transforming texts: {e}")
            raise
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            texts (List[str]): Texts to fit and transform
            
        Returns:
            np.ndarray: Feature matrix
        """
        return self.fit(texts).transform(texts)

# ==================== LINGUISTIC FEATURE EXTRACTION ====================
def extract_linguistic_features(text: str) -> Dict[str, float]:
    """
    Extract comprehensive linguistic features from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of linguistic features
    """
    if not text or not text.strip():
        return {feature: 0.0 for feature in get_feature_names()}
    
    try:
        # Basic text statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Character and word counts
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Basic ratios
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Punctuation analysis
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0
        
        # Case analysis
        uppercase_count = sum(1 for char in text if char.isupper())
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        
        # Digit analysis
        digit_count = sum(1 for char in text if char.isdigit())
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Readability scores
        readability_features = extract_readability_features(text)
        
        # Sentiment features
        sentiment_features = extract_sentiment_features(text)
        
        # Stylistic features
        stylistic_features = extract_stylistic_features(text)
        
        # Fake news indicator features
        indicator_features = extract_fake_news_indicators(text)
        
        # Combine all features
        features = {
            # Basic statistics
            'word_count': word_count,
            'sentence_count': sentence_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            
            # Ratios
            'punctuation_ratio': punctuation_ratio,
            'uppercase_ratio': uppercase_ratio,
            'digit_ratio': digit_ratio,
            'lexical_diversity': lexical_diversity,
            
            **readability_features,
            **sentiment_features,
            **stylistic_features,
            **indicator_features
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting linguistic features: {e}")
        return {feature: 0.0 for feature in get_feature_names()}

def extract_readability_features(text: str) -> Dict[str, float]:
    """
    Extract readability and complexity features
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Readability features
    """
    try:
        features = {
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text),
        }
        
        # Additional complexity measures
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Complex word ratio (words with 3+ syllables)
        complex_words = sum(1 for word in words if count_syllables(word) >= 3)
        features['complex_word_ratio'] = complex_words / len(words) if words else 0
        
        # Average words per sentence
        features['avg_words_per_sentence'] = len(words) / len(sentences) if sentences else 0
        
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting readability features: {e}")
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'complex_word_ratio': 0.0,
            'avg_words_per_sentence': 0.0
        }

def extract_sentiment_features(text: str) -> Dict[str, float]:
    """
    Extract sentiment-based features
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment features
    """
    try:
        # Initialize sentiment analyzer if not available
        try:
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
        except:
            # Fallback: simple sentiment analysis
            scores = {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0
            }
        
        features = {
            'sentiment_compound': scores['compound'],
            'sentiment_positive': scores['pos'],
            'sentiment_neutral': scores['neu'],
            'sentiment_negative': scores['neg'],
        }
        
        # Additional sentiment measures
        features['sentiment_intensity'] = abs(scores['compound'])
        features['sentiment_polarity'] = 1 if scores['compound'] > 0 else (-1 if scores['compound'] < 0 else 0)
        
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting sentiment features: {e}")
        return {
            'sentiment_compound': 0.0,
            'sentiment_positive': 0.0,
            'sentiment_neutral': 1.0,
            'sentiment_negative': 0.0,
            'sentiment_intensity': 0.0,
            'sentiment_polarity': 0.0
        }

def extract_stylistic_features(text: str) -> Dict[str, float]:
    """
    Extract stylistic and structural features
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Stylistic features
    """
    try:
        # Question and exclamation ratios
        question_count = text.count('?')
        exclamation_count = text.count('!')
        sentence_count = len(re.split(r'[.!?]+', text))
        
        features = {
            'question_ratio': question_count / sentence_count if sentence_count > 0 else 0,
            'exclamation_ratio': exclamation_count / sentence_count if sentence_count > 0 else 0,
        }
        
        # Quotation usage
        quote_count = text.count('"') + text.count("'")
        features['quote_ratio'] = quote_count / len(text) if text else 0
        
        # Capitalization patterns
        words = text.split()
        if words:
            all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            features['all_caps_ratio'] = all_caps_words / len(words)
        else:
            features['all_caps_ratio'] = 0
        
        # URL and email patterns
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        features['url_count'] = len(re.findall(url_pattern, text))
        features['email_count'] = len(re.findall(email_pattern, text))
        
        # Numbers and dates
        number_pattern = r'\b\d+\b'
        features['number_count'] = len(re.findall(number_pattern, text))
        
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting stylistic features: {e}")
        return {
            'question_ratio': 0.0,
            'exclamation_ratio': 0.0,
            'quote_ratio': 0.0,
            'all_caps_ratio': 0.0,
            'url_count': 0.0,
            'email_count': 0.0,
            'number_count': 0.0
        }

def extract_fake_news_indicators(text: str) -> Dict[str, float]:
    """
    Extract features that are indicators of fake news
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Fake news indicator features
    """
    try:
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Define indicator word lists
        fake_indicators = {
            'sensational_words': [
                'shocking', 'amazing', 'unbelievable', 'incredible', 'outrageous',
                'scandal', 'exposed', 'revealed', 'secret', 'hidden', 'bombshell'
            ],
            'emotional_words': [
                'hate', 'love', 'fear', 'anger', 'rage', 'fury', 'terrified',
                'disgusted', 'outraged', 'shocked', 'appalled'
            ],
            'uncertainty_words': [
                'allegedly', 'reportedly', 'supposedly', 'apparently', 'presumably',
                'rumored', 'claimed', 'suggested'
            ],
            'absolute_words': [
                'never', 'always', 'all', 'every', 'none', 'everyone', 'nobody',
                'everything', 'nothing'
            ]
        }
        
        features = {}
        
        # Count occurrences of each indicator type
        for indicator_type, word_list in fake_indicators.items():
            count = sum(1 for word in words if word in word_list)
            features[f'{indicator_type}_count'] = count
            features[f'{indicator_type}_ratio'] = count / word_count if word_count > 0 else 0
        
        # Overall fake indicators score
        total_indicators = sum(features[key] for key in features if key.endswith('_count'))
        features['total_fake_indicators'] = total_indicators
        features['fake_indicator_density'] = total_indicators / word_count if word_count > 0 else 0
        
        # Clickbait patterns
        clickbait_patterns = [
            r"you won't believe",
            r"what happens next",
            r"will shock you",
            r"doctors hate",
            r"this one trick",
            r"click here",
            r"find out"
        ]
        
        clickbait_count = sum(1 for pattern in clickbait_patterns if re.search(pattern, text_lower))
        features['clickbait_patterns'] = clickbait_count
        
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting fake news indicators: {e}")
        return {
            'sensational_words_count': 0.0,
            'sensational_words_ratio': 0.0,
            'emotional_words_count': 0.0,
            'emotional_words_ratio': 0.0,
            'uncertainty_words_count': 0.0,
            'uncertainty_words_ratio': 0.0,
            'absolute_words_count': 0.0,
            'absolute_words_ratio': 0.0,
            'total_fake_indicators': 0.0,
            'fake_indicator_density': 0.0,
            'clickbait_patterns': 0.0
        }

# ==================== UTILITY FUNCTIONS ====================
def count_syllables(word: str) -> int:
    """
    Count syllables in a word (simplified approach)
    
    Args:
        word (str): Input word
        
    Returns:
        int: Number of syllables
    """
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count += 1
    return count

def get_feature_names() -> List[str]:
    """
    Get list of all feature names
    
    Returns:
        List[str]: Feature names
    """
    return [
        # Basic statistics
        'word_count', 'sentence_count', 'char_count', 'avg_word_length', 'avg_sentence_length',
        
        # Ratios
        'punctuation_ratio', 'uppercase_ratio', 'digit_ratio', 'lexical_diversity',
        
        # Readability
        'flesch_reading_ease', 'flesch_kincaid_grade', 'complex_word_ratio', 'avg_words_per_sentence',
        
        # Sentiment
        'sentiment_compound', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative',
        'sentiment_intensity', 'sentiment_polarity',
        
        # Stylistic
        'question_ratio', 'exclamation_ratio', 'quote_ratio', 'all_caps_ratio',
        'url_count', 'email_count', 'number_count',
        
        # Fake news indicators
        'sensational_words_count', 'sensational_words_ratio',
        'emotional_words_count', 'emotional_words_ratio',
        'uncertainty_words_count', 'uncertainty_words_ratio',
        'absolute_words_count', 'absolute_words_ratio',
        'total_fake_indicators', 'fake_indicator_density', 'clickbait_patterns'
    ]

@st.cache_data
def analyze_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using TF-IDF
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Cosine similarity score
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculating text similarity: {e}")
        return 0.0

def extract_key_phrases(text: str, n_phrases: int = 10) -> List[Tuple[str, float]]:
    """
    Extract key phrases using TF-IDF scores
    
    Args:
        text (str): Input text
        n_phrases (int): Number of phrases to extract
        
    Returns:
        List[Tuple[str, float]]: List of (phrase, score) tuples
    """
    try:
        # Use TF-IDF to find important n-grams
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            stop_words='english',
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get top phrases
        phrase_scores = list(zip(feature_names, tfidf_scores))
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        
        return phrase_scores[:n_phrases]
        
    except Exception as e:
        logger.error(f"Error extracting key phrases: {e}")
        return []

def create_feature_summary(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Create a summary of extracted features
    
    Args:
        features (dict): Extracted features
        
    Returns:
        dict: Feature summary
    """
    try:
        summary = {
            'total_features': len(features),
            'feature_categories': {
                'basic_stats': ['word_count', 'sentence_count', 'char_count'],
                'readability': ['flesch_reading_ease', 'flesch_kincaid_grade'],
                'sentiment': ['sentiment_compound', 'sentiment_intensity'],
                'stylistic': ['question_ratio', 'exclamation_ratio', 'all_caps_ratio'],
                'fake_indicators': ['fake_indicator_density', 'clickbait_patterns']
            },
            'key_insights': {},
            'risk_factors': []
        }
        
        # Generate insights
        if features.get('flesch_reading_ease', 0) < 30:
            summary['key_insights']['readability'] = 'Very difficult to read'
        elif features.get('flesch_reading_ease', 0) > 90:
            summary['key_insights']['readability'] = 'Very easy to read'
        
        if features.get('sentiment_intensity', 0) > 0.8:
            summary['key_insights']['emotion'] = 'Highly emotional content'
        
        if features.get('fake_indicator_density', 0) > 0.1:
            summary['risk_factors'].append('High density of sensational language')
        
        if features.get('all_caps_ratio', 0) > 0.1:
            summary['risk_factors'].append('Excessive use of capital letters')
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating feature summary: {e}")
        return {'error': str(e)}

# ==================== EXPORT ALL FUNCTIONS ====================
__all__ = [
    'FakeNewsFeatureExtractor',
    'extract_linguistic_features',
    'extract_readability_features',
    'extract_sentiment_features',
    'extract_stylistic_features',
    'extract_fake_news_indicators',
    'analyze_text_similarity',
    'extract_key_phrases',
    'create_feature_summary',
    'get_feature_names',
    'DEFAULT_TFIDF_CONFIG'
]