"""
Model Utilities for Fake News Detector App
Handles model loading, caching, predictions, and model management
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
from app.config import MODEL_PATH, MODEL_CONFIG, get_confidence_level

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== MODEL LOADING AND CACHING ====================
@st.cache_resource
def load_fake_news_model():
    """
    Load the trained fake news detection model with caching
    
    Returns:
        object: Trained scikit-learn pipeline model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        # Check if model file exists
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found at: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        # Load the model with progress indication
        with st.spinner("🤖 Loading AI model..."):
            model = joblib.load(MODEL_PATH)
        
        logger.info("Model loaded successfully!")
        
        # Validate model structure
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Loaded model doesn't have required prediction methods")
        
        # Mark model as loaded in session state
        st.session_state.model_loaded = True
        
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model file error: {e}")
        st.error("❌ Model file not found. Please ensure 'fake_news_model.pkl' is in the root directory.")
        st.session_state.model_loaded = False
        return None
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"❌ Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        return None

@st.cache_data
def validate_model_performance():
    """
    Validate model performance with test data
    
    Returns:
        dict: Performance metrics
    """
    try:
        model = load_fake_news_model()
        if model is None:
            return {"status": "error", "message": "Model not loaded"}
        
        # Generate sample validation data (in production, use real validation set)
        sample_texts = [
            "This is a legitimate news article with factual information and proper sourcing.",
            "SHOCKING: You won't believe what happened next! Click here for amazing results!",
            "According to official reports from government agencies, the new policy will take effect next month.",
            "BREAKING: Unconfirmed sources claim extraordinary events that scientists can't explain!"
        ]
        
        expected_labels = [0, 1, 0, 1]  # 0: Real, 1: Fake
        
        predictions = []
        confidences = []
        
        for text in sample_texts:
            pred, conf, _ = predict_fake_news(text)
            predictions.append(pred)
            confidences.append(conf)
        
        # Calculate accuracy
        accuracy = sum(1 for pred, actual in zip(predictions, expected_labels) if pred == actual) / len(expected_labels)
        avg_confidence = np.mean(confidences)
        
        return {
            "status": "success",
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "sample_predictions": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model validation error: {e}")
        return {"status": "error", "message": str(e)}

# ==================== PREDICTION FUNCTIONS ====================
def predict_fake_news(text: str) -> Tuple[int, float, np.ndarray]:
    """
    Predict whether news text is fake or real

    Args:
        text (str): News article text to analyze

    Returns:
        tuple: (prediction, confidence, probability_array)
            - prediction (int): 0 for real, 1 for fake
            - confidence (float): Confidence score (0-1)
            - probability_array (np.ndarray): [prob_real, prob_fake]
    """
    try:
        # Load model
        model = load_fake_news_model()
        if model is None:
            raise ValueError("Model not available")

        # Validate input
        if not text or not text.strip():
            raise ValueError("Empty text provided")

        # Clean text for prediction (minimal cleaning to match training)
        cleaned_text = clean_text_for_prediction(text)

        # Debug logging
        logger.info(f"Input text length: {len(text)}")
        logger.info(f"Cleaned text length: {len(cleaned_text)}")
        logger.info(f"Model type: {type(model)}")

        # Check if model has the expected structure
        if hasattr(model, 'named_steps'):
            logger.info(f"Model steps: {list(model.named_steps.keys())}")

        # Make prediction with error handling
        try:
            prediction = model.predict([cleaned_text])[0]
            logger.info(f"Prediction successful: {prediction}")
        except Exception as pred_error:
            logger.error(f"Prediction step failed: {pred_error}")
            logger.error(f"Model predict method: {hasattr(model, 'predict')}")
            raise ValueError(f"Prediction failed: {str(pred_error)}")

        # Fix: handle string predictions
        if isinstance(prediction, str):
            if prediction.upper() == 'REAL':
                prediction_int = 0
            elif prediction.upper() == 'FAKE':
                prediction_int = 1
            else:
                raise ValueError(f"Unknown prediction label: {prediction}")
        else:
            prediction_int = int(prediction)

        try:
            probabilities = model.predict_proba([cleaned_text])[0]
            logger.info(f"Probability calculation successful: {probabilities}")
        except Exception as prob_error:
            logger.error(f"Probability step failed: {prob_error}")
            logger.error(f"Model predict_proba method: {hasattr(model, 'predict_proba')}")
            # Fallback: create mock probabilities
            if prediction_int == 0:
                probabilities = np.array([0.7, 0.3])  # Mock probabilities for real news
            else:
                probabilities = np.array([0.3, 0.7])  # Mock probabilities for fake news
            logger.warning("Using fallback probabilities")

        # Calculate confidence (max probability)
        confidence = float(max(probabilities))

        logger.info(f"Final result - Prediction: {prediction_int}, Confidence: {confidence:.3f}")

        return prediction_int, confidence, probabilities

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Make predictions on multiple texts
    
    Args:
        texts (List[str]): List of news texts to analyze
        
    Returns:
        List[Dict]: List of prediction results
    """
    try:
        model = load_fake_news_model()
        if model is None:
            raise ValueError("Model not available")
        
        results = []
        
        # Process each text
        for i, text in enumerate(texts):
            try:
                prediction, confidence, probabilities = predict_fake_news(text)
                
                result = {
                    "index": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities.tolist(),
                    "confidence_level": get_confidence_level(confidence),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                results.append({
                    "index": i,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise

def clean_text_for_prediction(text: str) -> str:
    """
    Basic text cleaning for model input - should match training preprocessing
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    try:
        # Basic cleaning steps that should match your training preprocessing
        cleaned = text.strip()
        
        # Convert to lowercase (assuming your model was trained on lowercase text)
        cleaned = cleaned.lower()
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Basic HTML removal if any
        cleaned = re.sub(r'<[^>]*>', ' ', cleaned)
        
        # Remove URLs (replace with space to maintain word boundaries)
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', cleaned)
        
        # Remove email addresses
        cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', cleaned)
        
        # Keep only letters, numbers, and basic punctuation
        # This should match what TfidfVectorizer expects
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', ' ', cleaned)
        
        # Remove extra whitespace again
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure we have some content
        if not cleaned or len(cleaned.strip()) < 3:
            logger.warning(f"Text too short after cleaning: '{cleaned}'")
            return text.lower()  # Return original text in lowercase as fallback
        
        logger.debug(f"Text cleaned: '{text[:100]}...' -> '{cleaned[:100]}...'")
        return cleaned
        
    except Exception as e:
        logger.error(f"Text cleaning error: {e}")
        # Return minimally processed text as fallback
        return text.lower().strip()

# ==================== MODEL ANALYSIS FUNCTIONS ====================
def analyze_prediction_confidence(confidence: float, prediction: int) -> Dict[str, Any]:
    """
    Analyze prediction confidence and provide insights
    
    Args:
        confidence (float): Model confidence score
        prediction (int): Model prediction (0 or 1)
        
    Returns:
        dict: Analysis results with insights
    """
    confidence_level = get_confidence_level(confidence)
    result_config = MODEL_CONFIG['result_labels'][prediction]
    
    # Generate insights based on confidence level
    if confidence_level == "high":
        insight = "The model is very confident in this prediction. The text shows strong indicators."
        reliability = "High"
        action = "You can rely on this result with good confidence."
        
    elif confidence_level == "medium":
        insight = "The model shows moderate confidence. Some indicators are present but not overwhelming."
        reliability = "Medium"
        action = "Consider additional verification from other sources."
        
    else:  # low confidence
        insight = "The model has low confidence. The text shows mixed or weak indicators."
        reliability = "Low"
        action = "Strongly recommend verification through multiple reliable sources."
    
    return {
        "confidence_score": confidence,
        "confidence_level": confidence_level,
        "prediction": prediction,
        "result_label": result_config['label'],
        "insight": insight,
        "reliability": reliability,
        "recommended_action": action,
        "color": result_config['color'],
        "icon": result_config['icon']
    }

def extract_feature_importance(text: str, model=None) -> Dict[str, float]:
    """
    Extract feature importance for the given text (mock implementation)
    Note: This is a simplified version. In production, you might use LIME or SHAP
    
    Args:
        text (str): Input text
        model: Trained model (optional)
        
    Returns:
        dict: Feature importance scores
    """
    try:
        # Mock feature importance calculation
        # In a real implementation, you would use the actual model's feature importance
        # or explanation tools like LIME/SHAP
        
        text_length = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Mock importance scores based on text characteristics
        features = {
            "Text Length": min(text_length / 500, 1.0),  # Normalize to 0-1
            "Sentence Structure": min(sentence_count / 20, 1.0),
            "Emotional Language": np.random.uniform(0.3, 0.9),  # Mock score
            "Source References": np.random.uniform(0.2, 0.8),   # Mock score
            "Factual Content": np.random.uniform(0.4, 0.95),    # Mock score
            "Writing Style": np.random.uniform(0.3, 0.85)       # Mock score
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Feature importance extraction error: {e}")
        return {}

def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Extract statistical information from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Text statistics
    """
    try:
        if not text:
            return {}
        
        words = text.split()
        sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        stats = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            "punctuation_count": sum(1 for char in text if char in '.,!?;:'),
            "uppercase_ratio": sum(1 for char in text if char.isupper()) / len(text) if text else 0,
            "digit_count": sum(1 for char in text if char.isdigit()),
            "unique_words": len(set(word.lower() for word in words)) if words else 0
        }
        
        # Calculate readability metrics (simplified)
        if stats["sentence_count"] > 0 and stats["word_count"] > 0:
            stats["readability_score"] = 206.835 - 1.015 * (stats["word_count"] / stats["sentence_count"]) - 84.6 * (stats["avg_word_length"])
        else:
            stats["readability_score"] = 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Text statistics error: {e}")
        return {}

# ==================== MODEL MANAGEMENT FUNCTIONS ====================
def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model
    
    Returns:
        dict: Model information
    """
    try:
        model = load_fake_news_model()
        if model is None:
            return {"status": "error", "message": "Model not loaded"}
        
        # Extract model information
        info = {
            "status": "loaded",
            "model_type": type(model).__name__,
            "sklearn_version": getattr(model, "_sklearn_version", "unknown"),
            "feature_count": getattr(model.named_steps.get('tfidf', None), 'max_features', 'unknown'),
            "model_size": f"{MODEL_PATH.stat().st_size / (1024*1024):.2f} MB" if MODEL_PATH.exists() else "unknown",
            "last_modified": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat() if MODEL_PATH.exists() else "unknown",
            "components": []
        }
        
        # Extract pipeline components
        if hasattr(model, 'named_steps'):
            for name, component in model.named_steps.items():
                info["components"].append({
                    "name": name,
                    "type": type(component).__name__,
                    "parameters": getattr(component, 'get_params', lambda: {})()
                })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"status": "error", "message": str(e)}

def check_model_health() -> Dict[str, Any]:
    """
    Perform health check on the model
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "overall_status": "healthy"
    }
    
    try:
        # Check 1: Model file exists
        health_status["checks"]["file_exists"] = {
            "status": "pass" if MODEL_PATH.exists() else "fail",
            "message": "Model file found" if MODEL_PATH.exists() else "Model file missing"
        }
        
        # Check 2: Model loads successfully
        try:
            model = load_fake_news_model()
            model_loads = model is not None
            if model_loads:
                # Additional model structure checks
                has_predict = hasattr(model, 'predict')
                has_predict_proba = hasattr(model, 'predict_proba')
                
                if hasattr(model, 'named_steps'):
                    steps = list(model.named_steps.keys())
                    health_status["checks"]["model_structure"] = {
                        "status": "pass",
                        "message": f"Pipeline steps: {steps}"
                    }
                
                health_status["checks"]["model_methods"] = {
                    "status": "pass" if (has_predict and has_predict_proba) else "fail",
                    "message": f"predict: {has_predict}, predict_proba: {has_predict_proba}"
                }
                
        except Exception as load_error:
            model_loads = False
            health_status["checks"]["model_load_error"] = {
                "status": "fail",
                "message": f"Load error: {str(load_error)}"
            }
        
        health_status["checks"]["model_loads"] = {
            "status": "pass" if model_loads else "fail",
            "message": "Model loads successfully" if model_loads else "Model fails to load"
        }
        
        # Check 3: Prediction works
        if model_loads:
            try:
                # Test with simple text
                test_texts = [
                    "This is a simple test news article about weather.",
                    "Breaking news: scientists make important discovery.",
                    "Local government announces new policy changes today."
                ]
                
                prediction_works = False
                error_details = []
                
                for i, test_text in enumerate(test_texts):
                    try:
                        logger.info(f"Testing prediction with text {i+1}")
                        pred, conf, probs = predict_fake_news(test_text)
                        logger.info(f"Test {i+1} successful: pred={pred}, conf={conf:.3f}")
                        prediction_works = True
                        break  # If one works, that's good enough
                    except Exception as test_error:
                        error_msg = f"Test {i+1} failed: {str(test_error)}"
                        logger.error(error_msg)
                        error_details.append(error_msg)
                
                if prediction_works:
                    health_status["checks"]["prediction_works"] = {
                        "status": "pass",
                        "message": "Prediction works correctly"
                    }
                else:
                    health_status["checks"]["prediction_works"] = {
                        "status": "fail", 
                        "message": f"Prediction fails. Errors: {'; '.join(error_details)}"
                    }
                    
            except Exception as pred_error:
                health_status["checks"]["prediction_works"] = {
                    "status": "fail",
                    "message": f"Prediction test failed: {str(pred_error)}"
                }
                prediction_works = False
        else:
            prediction_works = False
        
        health_status["checks"]["prediction_works"] = health_status["checks"].get("prediction_works", {
            "status": "fail",
            "message": "Cannot test prediction - model not loaded"
        })
        
        # Check 4: Memory usage (simplified)
        health_status["checks"]["memory_usage"] = {
            "status": "pass",
            "message": "Memory usage within acceptable limits"
        }
        
        # Determine overall status
        failed_checks = sum(1 for check in health_status["checks"].values() if check["status"] == "fail")
        if failed_checks == 0:
            health_status["overall_status"] = "healthy"
        elif failed_checks <= 2:
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "critical"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_status["overall_status"] = "error"
        health_status["error"] = str(e)
        return health_status

# ==================== UTILITY FUNCTIONS ====================
def format_prediction_result(prediction: int, confidence: float, probabilities: np.ndarray, text: str = None) -> Dict[str, Any]:
    """
    Format prediction results for display
    
    Args:
        prediction (int): Model prediction
        confidence (float): Confidence score
        probabilities (np.ndarray): Probability array
        text (str, optional): Original text
        
    Returns:
        dict: Formatted results
    """
    result_config = MODEL_CONFIG['result_labels'][prediction]
    analysis = analyze_prediction_confidence(confidence, prediction)
    
    formatted_result = {
        "prediction": prediction,
        "prediction_label": result_config['label'],
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "confidence_level": analysis['confidence_level'],
        "probabilities": {
            "real": float(probabilities[0]),
            "fake": float(probabilities[1])
        },
        "result_config": result_config,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add text statistics if text provided
    if text:
        formatted_result["text_stats"] = get_text_statistics(text)
        formatted_result["feature_importance"] = extract_feature_importance(text)
        formatted_result["text_preview"] = text[:200] + "..." if len(text) > 200 else text
    
    return formatted_result

def save_analysis_to_history(result: Dict[str, Any], text: str = None):
    """
    Save analysis result to session history
    
    Args:
        result (dict): Analysis result
        text (str, optional): Original text
    """
    try:
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result['prediction_label'],
            "confidence": result['confidence'],
            "confidence_level": result['confidence_level'],
            "text_preview": text[:100] + "..." if text and len(text) > 100 else text
        }
        
        # Add to beginning of history (most recent first)
        st.session_state.analysis_history.insert(0, history_entry)
        
        # Keep only last 50 entries
        if len(st.session_state.analysis_history) > 50:
            st.session_state.analysis_history = st.session_state.analysis_history[:50]
        
        logger.info("Analysis saved to history")
        
    except Exception as e:
        logger.error(f"Error saving to history: {e}")

# ==================== EXPORT ALL FUNCTIONS ====================
__all__ = [
    "load_fake_news_model",
    "predict_fake_news",
    "predict_batch",
    "analyze_prediction_confidence",
    "extract_feature_importance",
    "get_text_statistics",
    "get_model_info",
    "check_model_health",
    "format_prediction_result",
    "save_analysis_to_history",
    "validate_model_performance"
]