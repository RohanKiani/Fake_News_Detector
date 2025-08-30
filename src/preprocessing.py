"""
Text Preprocessing Module for Fake News Detector App
Comprehensive text cleaning, normalization, and preparation functions
"""

import streamlit as st
import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import contractions
import html
import ftfy
import pandas as pd
import numpy as np
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data with error handling
NLTK_DOWNLOADS = [
    'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
    'omw-1.4', 'vader_lexicon'
]

def download_nltk_resources():
    """Download required NLTK resources with error handling"""
    for resource in NLTK_DOWNLOADS:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {e}")

# Initialize NLTK resources
download_nltk_resources()

# Initialize stemmer and lemmatizer
try:
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"NLTK initialization warning: {e}")
    STEMMER = None
    LEMMATIZER = None
    STOP_WORDS = set()

# ==================== PREPROCESSING CONFIGURATION ====================
DEFAULT_PREPROCESSING_CONFIG = {
    'lowercase': True,
    'remove_html': True,
    'fix_encoding': True,
    'expand_contractions': True,
    'remove_urls': True,
    'remove_emails': True,
    'remove_phone_numbers': True,
    'remove_special_chars': True,
    'remove_extra_whitespace': True,
    'remove_stopwords': False,  # Keep False for news analysis
    'apply_stemming': False,    # Keep False to preserve meaning
    'apply_lemmatization': False, # Keep False for production model
    'min_word_length': 2,
    'max_word_length': 50,
    'preserve_sentence_structure': True,
    'remove_short_sentences': True,
    'min_sentence_length': 10
}

# ==================== MAIN PREPROCESSING CLASS ====================
class NewsTextPreprocessor:
    """
    Advanced text preprocessor specifically designed for news content analysis
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config (dict): Preprocessing configuration parameters
        """
        self.config = {**DEFAULT_PREPROCESSING_CONFIG, **(config or {})}
        self.stats = {
            'texts_processed': 0,
            'total_chars_removed': 0,
            'total_words_removed': 0,
            'processing_time': 0
        }
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("NewsTextPreprocessor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.patterns = {
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'html_tags': re.compile(r'<[^>]+>'),
            'special_chars': re.compile(r'[^a-zA-Z0-9\s.,!?;:\'\"-]'),
            'extra_whitespace': re.compile(r'\s+'),
            'multiple_punctuation': re.compile(r'([.!?]){2,}'),
            'quotes': re.compile(r'[""''`´]'),
            'dashes': re.compile(r'[—–−-]{2,}'),
            'numbers_only': re.compile(r'\b\d+\b'),
            'repeated_chars': re.compile(r'(.)\1{3,}')
        }
    
    def preprocess(self, text: str, custom_config: Dict = None) -> str:
        """
        Main preprocessing pipeline
        
        Args:
            text (str): Input text to preprocess
            custom_config (dict): Custom config for this specific text
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Use custom config if provided
        config = {**self.config, **(custom_config or {})}
        
        try:
            original_text = text
            
            # Step 1: Fix encoding issues
            if config['fix_encoding']:
                text = self._fix_encoding(text)
            
            # Step 2: Remove HTML
            if config['remove_html']:
                text = self._remove_html(text)
            
            # Step 3: Expand contractions
            if config['expand_contractions']:
                text = self._expand_contractions(text)
            
            # Step 4: Normalize text
            text = self._normalize_text(text, config)
            
            # Step 5: Remove unwanted elements
            text = self._remove_unwanted_elements(text, config)
            
            # Step 6: Clean punctuation and formatting
            text = self._clean_formatting(text, config)
            
            # Step 7: Filter words and sentences
            text = self._filter_content(text, config)
            
            # Step 8: Final cleanup
            text = self._final_cleanup(text, config)
            
            # Update statistics
            self._update_stats(original_text, text)
            
            logger.debug(f"Preprocessed text: {len(original_text)} -> {len(text)} chars")
            
            return text
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return text  # Return original text if preprocessing fails
    
    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues and normalize unicode"""
        try:
            # Fix encoding issues using ftfy
            text = ftfy.fix_text(text)
            
            # Normalize unicode
            text = unicodedata.normalize('NFKD', text)
            
            # Decode HTML entities
            text = html.unescape(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Encoding fix warning: {e}")
            return text
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and entities"""
        try:
            # Remove HTML tags
            text = self.patterns['html_tags'].sub(' ', text)
            
            # Remove common HTML entities not caught by html.unescape
            html_entities = {
                '&nbsp;': ' ',
                '&amp;': '&',
                '&lt;': '<',
                '&gt;': '>',
                '&quot;': '"',
                '&#39;': "'",
                '&hellip;': '...',
                '&mdash;': '—',
                '&ndash;': '–'
            }
            
            for entity, replacement in html_entities.items():
                text = text.replace(entity, replacement)
            
            return text
            
        except Exception as e:
            logger.warning(f"HTML removal warning: {e}")
            return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions using the contractions library"""
        try:
            # Use contractions library for comprehensive expansion
            text = contractions.fix(text)
            
            # Additional manual contractions for news-specific terms
            news_contractions = {
                "gov't": "government",
                "dep't": "department",
                "int'l": "international",
                "nat'l": "national",
                "admin.": "administration",
                "pres.": "president",
                "min.": "minister",
                "sec.": "secretary"
            }
            
            for contraction, expansion in news_contractions.items():
                text = re.sub(rf'\b{re.escape(contraction)}\b', expansion, text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            logger.warning(f"Contraction expansion warning: {e}")
            return text
    
    def _normalize_text(self, text: str, config: Dict) -> str:
        """Normalize text case and basic formatting"""
        try:
            # Convert to lowercase if specified
            if config['lowercase']:
                text = text.lower()
            
            # Normalize quotes
            text = self.patterns['quotes'].sub('"', text)
            
            # Normalize dashes
            text = self.patterns['dashes'].sub(' - ', text)
            
            # Fix multiple punctuation
            text = self.patterns['multiple_punctuation'].sub(r'\1', text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Text normalization warning: {e}")
            return text
    
    def _remove_unwanted_elements(self, text: str, config: Dict) -> str:
        """Remove unwanted elements like URLs, emails, etc."""
        try:
            # Remove URLs
            if config['remove_urls']:
                text = self.patterns['url'].sub(' [URL] ', text)  # Replace with placeholder
            
            # Remove email addresses
            if config['remove_emails']:
                text = self.patterns['email'].sub(' [EMAIL] ', text)  # Replace with placeholder
            
            # Remove phone numbers
            if config['remove_phone_numbers']:
                text = self.patterns['phone'].sub(' [PHONE] ', text)  # Replace with placeholder
            
            return text
            
        except Exception as e:
            logger.warning(f"Element removal warning: {e}")
            return text
    
    def _clean_formatting(self, text: str, config: Dict) -> str:
        """Clean formatting and special characters"""
        try:
            # Remove repeated characters (e.g., "sooooo" -> "so")
            text = self.patterns['repeated_chars'].sub(r'\1\1', text)
            
            # Remove special characters but preserve sentence structure
            if config['remove_special_chars']:
                # Keep basic punctuation for sentence structure
                allowed_chars = r'[^a-zA-Z0-9\s.,!?;:\'\"-]'
                text = re.sub(allowed_chars, ' ', text)
            
            # Clean extra whitespace
            if config['remove_extra_whitespace']:
                text = self.patterns['extra_whitespace'].sub(' ', text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Formatting cleanup warning: {e}")
            return text
    
    def _filter_content(self, text: str, config: Dict) -> str:
        """Filter words and sentences based on criteria"""
        try:
            if not config.get('preserve_sentence_structure', True):
                # Word-level filtering
                words = text.split()
                filtered_words = []
                
                for word in words:
                    # Filter by length
                    if len(word) < config['min_word_length'] or len(word) > config['max_word_length']:
                        continue
                    
                    # Remove stopwords if specified
                    if config['remove_stopwords'] and word.lower() in STOP_WORDS:
                        continue
                    
                    # Apply stemming
                    if config['apply_stemming'] and STEMMER:
                        word = STEMMER.stem(word)
                    
                    # Apply lemmatization
                    if config['apply_lemmatization'] and LEMMATIZER:
                        word = LEMMATIZER.lemmatize(word)
                    
                    filtered_words.append(word)
                
                text = ' '.join(filtered_words)
            
            else:
                # Sentence-level filtering (preserve structure)
                if config.get('remove_short_sentences', True):
                    sentences = sent_tokenize(text) if nltk else text.split('.')
                    filtered_sentences = []
                    
                    for sentence in sentences:
                        if len(sentence.strip()) >= config.get('min_sentence_length', 10):
                            filtered_sentences.append(sentence.strip())
                    
                    text = '. '.join(filtered_sentences)
                    if text and not text.endswith('.'):
                        text += '.'
            
            return text
            
        except Exception as e:
            logger.warning(f"Content filtering warning: {e}")
            return text
    
    def _final_cleanup(self, text: str, config: Dict) -> str:
        """Final cleanup and validation"""
        try:
            # Remove extra whitespace one more time
            text = self.patterns['extra_whitespace'].sub(' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            # Ensure proper sentence endings
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            # Fix spacing around punctuation
            text = re.sub(r'\s+([.!?])', r'\1', text)
            text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Final cleanup warning: {e}")
            return text
    
    def _update_stats(self, original: str, processed: str):
        """Update preprocessing statistics"""
        try:
            self.stats['texts_processed'] += 1
            self.stats['total_chars_removed'] += len(original) - len(processed)
            
            original_words = len(original.split())
            processed_words = len(processed.split())
            self.stats['total_words_removed'] += original_words - processed_words
            
        except Exception as e:
            logger.warning(f"Stats update warning: {e}")
    
    def get_stats(self) -> Dict:
        """Get preprocessing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset preprocessing statistics"""
        self.stats = {
            'texts_processed': 0,
            'total_chars_removed': 0,
            'total_words_removed': 0,
            'processing_time': 0
        }

# ==================== BATCH PREPROCESSING FUNCTIONS ====================
def preprocess_batch(texts: List[str], config: Dict = None, show_progress: bool = True) -> List[str]:
    """
    Preprocess a batch of texts with optional progress display
    
    Args:
        texts (List[str]): List of texts to preprocess
        config (dict): Preprocessing configuration
        show_progress (bool): Whether to show progress bar
        
    Returns:
        List[str]: List of preprocessed texts
    """
    try:
        preprocessor = NewsTextPreprocessor(config)
        processed_texts = []
        
        if show_progress and len(texts) > 10:
            # Use Streamlit progress bar for large batches
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text in enumerate(texts):
                processed_text = preprocessor.preprocess(text)
                processed_texts.append(processed_text)
                
                # Update progress
                progress = (i + 1) / len(texts)
                progress_bar.progress(progress)
                status_text.text(f'Processing text {i + 1} of {len(texts)}...')
            
            progress_bar.empty()
            status_text.empty()
        
        else:
            # Process without progress bar
            for text in texts:
                processed_text = preprocessor.preprocess(text)
                processed_texts.append(processed_text)
        
        logger.info(f"Batch preprocessing completed: {len(texts)} texts")
        return processed_texts
        
    except Exception as e:
        logger.error(f"Batch preprocessing error: {e}")
        return texts  # Return original texts if preprocessing fails

@st.cache_data
def preprocess_dataframe(df: pd.DataFrame, text_column: str, config: Dict = None) -> pd.DataFrame:
    """
    Preprocess text in a pandas DataFrame with caching
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        config (dict): Preprocessing configuration
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed text
    """
    try:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        preprocessor = NewsTextPreprocessor(config)
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Add preprocessing progress
        with st.spinner(f"Preprocessing {len(df)} texts..."):
            df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(
                preprocessor.preprocess
            )
        
        # Add preprocessing statistics as metadata
        stats = preprocessor.get_stats()
        st.success(f"✅ Preprocessed {stats['texts_processed']} texts")
        
        return df_processed
        
    except Exception as e:
        logger.error(f"DataFrame preprocessing error: {e}")
        st.error(f"❌ Preprocessing failed: {str(e)}")
        return df

# ==================== VALIDATION AND QUALITY FUNCTIONS ====================
def validate_preprocessed_text(text: str) -> Dict[str, Union[bool, str, int]]:
    """
    Validate the quality of preprocessed text
    
    Args:
        text (str): Preprocessed text to validate
        
    Returns:
        dict: Validation results
    """
    try:
        validation = {
            'is_valid': True,
            'issues': [],
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentence_count': len(sent_tokenize(text)) if nltk else len(text.split('.')),
            'quality_score': 0.0
        }
        
        # Check minimum length
        if len(text.strip()) < 10:
            validation['is_valid'] = False
            validation['issues'].append('Text too short')
        
        # Check for excessive repetition
        words = text.split()
        if words:
            word_freq = Counter(words)
            most_common_ratio = word_freq.most_common(1)[0][1] / len(words)
            if most_common_ratio > 0.3:
                validation['issues'].append('Excessive word repetition')
        
        # Check for proper sentence structure
        if not re.search(r'[.!?]', text):
            validation['issues'].append('No sentence endings found')
        
        # Check for reasonable word length distribution
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 2 or avg_word_length > 20:
                validation['issues'].append('Unusual word length distribution')
        
        # Calculate quality score
        base_score = 1.0
        penalty_per_issue = 0.2
        validation['quality_score'] = max(0.0, base_score - (len(validation['issues']) * penalty_per_issue))
        
        return validation
        
    except Exception as e:
        logger.error(f"Text validation error: {e}")
        return {
            'is_valid': False,
            'issues': [f'Validation error: {str(e)}'],
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'quality_score': 0.0
        }

def analyze_preprocessing_impact(original_text: str, processed_text: str) -> Dict[str, Any]:
    """
    Analyze the impact of preprocessing on text
    
    Args:
        original_text (str): Original text
        processed_text (str): Preprocessed text
        
    Returns:
        dict: Analysis of preprocessing impact
    """
    try:
        original_words = original_text.split()
        processed_words = processed_text.split()
        
        analysis = {
            'character_reduction': {
                'original': len(original_text),
                'processed': len(processed_text),
                'reduction_ratio': 1 - (len(processed_text) / len(original_text)) if original_text else 0
            },
            'word_reduction': {
                'original': len(original_words),
                'processed': len(processed_words),
                'reduction_ratio': 1 - (len(processed_words) / len(original_words)) if original_words else 0
            },
            'changes_made': [],
            'quality_improvement': 0.0
        }
        
        # Identify specific changes
        if len(original_text) != len(processed_text):
            analysis['changes_made'].append('Length normalized')
        
        if original_text.lower() != original_text and processed_text == processed_text.lower():
            analysis['changes_made'].append('Converted to lowercase')
        
        if re.search(r'http[s]?://', original_text) and not re.search(r'http[s]?://', processed_text):
            analysis['changes_made'].append('URLs removed')
        
        if re.search(r'<[^>]+>', original_text) and not re.search(r'<[^>]+>', processed_text):
            analysis['changes_made'].append('HTML tags removed')
        
        # Estimate quality improvement (simplified)
        original_quality = validate_preprocessed_text(original_text)['quality_score']
        processed_quality = validate_preprocessed_text(processed_text)['quality_score']
        analysis['quality_improvement'] = processed_quality - original_quality
        
        return analysis
        
    except Exception as e:
        logger.error(f"Preprocessing impact analysis error: {e}")
        return {'error': str(e)}

# ==================== UTILITY FUNCTIONS ====================
def create_preprocessing_report(texts: List[str], preprocessed_texts: List[str]) -> Dict[str, Any]:
    """
    Create a comprehensive preprocessing report
    
    Args:
        texts (List[str]): Original texts
        preprocessed_texts (List[str]): Preprocessed texts
        
    Returns:
        dict: Comprehensive preprocessing report
    """
    try:
        if len(texts) != len(preprocessed_texts):
            raise ValueError("Texts and preprocessed_texts must have same length")
        
        report = {
            'summary': {
                'total_texts': len(texts),
                'avg_original_length': np.mean([len(text) for text in texts]),
                'avg_processed_length': np.mean([len(text) for text in preprocessed_texts]),
                'total_chars_removed': sum(len(orig) - len(proc) for orig, proc in zip(texts, preprocessed_texts)),
                'processing_efficiency': 0.0
            },
            'quality_metrics': {
                'texts_improved': 0,
                'texts_degraded': 0,
                'avg_quality_change': 0.0
            },
            'common_changes': Counter(),
            'validation_results': {
                'valid_texts': 0,
                'invalid_texts': 0,
                'common_issues': Counter()
            }
        }
        
        quality_changes = []
        
        for original, processed in zip(texts, preprocessed_texts):
            # Analyze impact
            impact = analyze_preprocessing_impact(original, processed)
            if 'error' not in impact:
                report['common_changes'].update(impact['changes_made'])
                quality_changes.append(impact['quality_improvement'])
                
                if impact['quality_improvement'] > 0:
                    report['quality_metrics']['texts_improved'] += 1
                elif impact['quality_improvement'] < 0:
                    report['quality_metrics']['texts_degraded'] += 1
            
            # Validate processed text
            validation = validate_preprocessed_text(processed)
            if validation['is_valid']:
                report['validation_results']['valid_texts'] += 1
            else:
                report['validation_results']['invalid_texts'] += 1
                report['validation_results']['common_issues'].update(validation['issues'])
        
        # Calculate averages
        if quality_changes:
            report['quality_metrics']['avg_quality_change'] = np.mean(quality_changes)
        
        report['summary']['processing_efficiency'] = (
            report['validation_results']['valid_texts'] / len(texts) * 100
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Preprocessing report error: {e}")
        return {'error': str(e)}

def get_preprocessing_recommendations(text: str) -> List[str]:
    """
    Get preprocessing recommendations for specific text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        List[str]: List of preprocessing recommendations
    """
    try:
        recommendations = []
        
        # Check for HTML
        if re.search(r'<[^>]+>', text):
            recommendations.append("Remove HTML tags for cleaner analysis")
        
        # Check for URLs
        if re.search(r'http[s]?://', text):
            recommendations.append("Consider removing or replacing URLs with placeholders")
        
        # Check case consistency
        if text != text.lower() and text != text.upper():
            recommendations.append("Normalize text case (lowercase recommended for news analysis)")
        
        # Check for excessive punctuation
        if re.search(r'[.!?]{3,}', text):
            recommendations.append("Normalize excessive punctuation")
        
        # Check for contractions
        if re.search(r"\w+'\w+", text):
            recommendations.append("Expand contractions for better tokenization")
        
        # Check text length
        if len(text) < 100:
            recommendations.append("Text might be too short for reliable analysis")
        elif len(text) > 10000:
            recommendations.append("Consider truncating very long texts")
        
        # Check for encoding issues
        if re.search(r'[^\x00-\x7F]', text):
            recommendations.append("Fix potential encoding issues")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return ["Error analyzing text for recommendations"]

# ==================== STREAMLIT INTEGRATION ====================
def create_preprocessing_interface():
    """Create Streamlit interface for preprocessing configuration"""
    st.markdown("### 🔧 Preprocessing Configuration")
    
    config = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Settings**")
        config['lowercase'] = st.checkbox("Convert to lowercase", value=True)
        config['remove_html'] = st.checkbox("Remove HTML tags", value=True)
        config['expand_contractions'] = st.checkbox("Expand contractions", value=True)
        config['remove_urls'] = st.checkbox("Remove URLs", value=True)
        config['remove_emails'] = st.checkbox("Remove email addresses", value=True)
    
    with col2:
        st.markdown("**Advanced Settings**")
        config['remove_stopwords'] = st.checkbox("Remove stop words", value=False)
        config['apply_stemming'] = st.checkbox("Apply stemming", value=False)
        config['apply_lemmatization'] = st.checkbox("Apply lemmatization", value=False)
        config['min_word_length'] = st.slider("Minimum word length", 1, 5, 2)
        config['min_sentence_length'] = st.slider("Minimum sentence length", 5, 50, 10)
    
    return config

# ==================== EXPORT ALL FUNCTIONS ====================
__all__ = [
    'NewsTextPreprocessor',
    'preprocess_batch',
    'preprocess_dataframe',
    'validate_preprocessed_text',
    'analyze_preprocessing_impact',
    'create_preprocessing_report',
    'get_preprocessing_recommendations',
    'create_preprocessing_interface',
    'DEFAULT_PREPROCESSING_CONFIG'
]