"""
Configuration file for Fake News Detector App
Contains all constants, styling, and configuration settings
"""

import streamlit as st
from pathlib import Path

# ==================== PATH CONFIGURATIONS ====================
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "fake_news_model.pkl"
ASSETS_DIR = BASE_DIR / "assets"
CSS_FILE = ASSETS_DIR / "style.css"

# ==================== APP CONFIGURATIONS ====================
APP_CONFIG = {
    "page_title": "🔍 Fake News Detective",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        'Get Help': 'https://github.com/yourusername/fake-news-detector',
        'Report a bug': "https://github.com/yourusername/fake-news-detector/issues",
        'About': "# Fake News Detective\nAI-powered fake news detection system using advanced NLP techniques."
    }
}

# ==================== THEME AND STYLING ====================
THEME = {
    # Primary Colors - Modern Tech Theme
    "primary": "#6C5CE7",           # Purple
    "primary_dark": "#5A4FCF",      # Darker purple
    "secondary": "#00CEC9",         # Teal
    "secondary_dark": "#00B894",    # Darker teal
    
    # Status Colors
    "success": "#00B894",           # Green
    "warning": "#FDCB6E",           # Yellow
    "danger": "#E17055",            # Red/Orange
    "info": "#74B9FF",              # Blue
    
    # Neutral Colors
    "background": "#F8F9FA",        # Light gray
    "surface": "#FFFFFF",           # White
    "surface_dark": "#2D3436",      # Dark gray
    "text_primary": "#2D3436",      # Dark gray
    "text_secondary": "#636E72",    # Medium gray
    "text_light": "#B2BEC3",        # Light gray
    
    # Gradients
    "gradient_primary": "linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%)",
    "gradient_secondary": "linear-gradient(135deg, #00CEC9 0%, #55EFC4 100%)",
    "gradient_danger": "linear-gradient(135deg, #E17055 0%, #FF7675 100%)",
    "gradient_success": "linear-gradient(135deg, #00B894 0%, #55EFC4 100%)",
}

# ==================== UI COMPONENTS STYLING ====================
COMPONENT_STYLES = {
    "card": {
        "background": THEME["surface"],
        "border_radius": "15px",
        "box_shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "padding": "1.5rem",
        "margin": "1rem 0",
        "border": "1px solid #E9ECEF"
    },
    
    "button_primary": {
        "background": THEME["gradient_primary"],
        "color": "white",
        "border": "none",
        "border_radius": "25px",
        "padding": "0.75rem 2rem",
        "font_weight": "600",
        "font_size": "1rem",
        "transition": "all 0.3s ease"
    },
    
    "button_secondary": {
        "background": THEME["gradient_secondary"],
        "color": "white",
        "border": "none",
        "border_radius": "25px",
        "padding": "0.75rem 2rem",
        "font_weight": "600",
        "font_size": "1rem",
        "transition": "all 0.3s ease"
    },
    
    "input_field": {
        "border": f"2px solid {THEME['text_light']}",
        "border_radius": "10px",
        "padding": "0.75rem",
        "font_size": "1rem",
        "transition": "border-color 0.3s ease"
    }
}

# ==================== CONTENT CONSTANTS ====================
CONTENT = {
    "app_title": "🔍 Fake News Detective",
    "app_subtitle": "AI-Powered News Authenticity Analyzer",
    "app_description": "Leverage advanced machine learning to detect potentially misleading or fabricated news content with high accuracy.",
    
    "features": [
        {
            "icon": "🤖",
            "title": "AI-Powered Analysis",
            "description": "Advanced machine learning algorithms trained on thousands of news articles"
        },
        {
            "icon": "⚡",
            "title": "Real-time Detection",
            "description": "Get instant results with confidence scores and detailed analysis"
        },
        {
            "icon": "🎯",
            "title": "High Accuracy",
            "description": "Achieve reliable detection with our carefully tuned models"
        },
        {
            "icon": "📊",
            "title": "Detailed Insights",
            "description": "Comprehensive analysis with feature importance and explanations"
        }
    ],
    
    "instructions": {
        "title": "How to Use",
        "steps": [
            "📝 Paste or type the news article content in the text area",
            "🚀 Click 'Analyze News' to start the detection process",
            "📊 Review the results with confidence scores",
            "💡 Check the detailed analysis and key indicators"
        ]
    },
    
    "disclaimer": "⚠️ **Disclaimer:** This tool provides AI-based analysis and should be used as a supplementary resource. Always verify information through multiple reliable sources and apply critical thinking when evaluating news content.",
    
    "footer": {
        "text": "Built with ❤️ using Streamlit and Scikit-learn",
        "version": "v1.0.0",
        "last_updated": "2024"
    }
}

# ==================== MODEL CONFIGURATIONS ====================
MODEL_CONFIG = {
    "confidence_threshold": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    },
    
    "result_labels": {
        0: {
            "label": "Reliable News",
            "color": THEME["success"],
            "gradient": THEME["gradient_success"],
            "icon": "✅",
            "message": "This content appears to be from a reliable source."
        },
        1: {
            "label": "Potentially Fake",
            "color": THEME["danger"],
            "gradient": THEME["gradient_danger"],
            "icon": "⚠️",
            "message": "This content shows characteristics of potentially misleading information."
        }
    },
    
    "analysis_features": [
        "Word Count",
        "Sentence Length",
        "Punctuation Usage",
        "Emotional Language",
        "Source Credibility Indicators",
        "Fact-based Content Ratio"
    ]
}

# ==================== ANIMATION AND INTERACTION ====================
ANIMATIONS = {
    "fade_in": "fadeIn 0.8s ease-in-out",
    "slide_up": "slideUp 0.6s ease-out",
    "pulse": "pulse 2s ease-in-out infinite",
    "bounce": "bounce 1s ease-in-out",
    "spin": "spin 1s linear infinite"
}

# ==================== RESPONSIVE BREAKPOINTS ====================
BREAKPOINTS = {
    "mobile": "768px",
    "tablet": "1024px",
    "desktop": "1200px"
}

# ==================== UTILITY FUNCTIONS ====================
def get_confidence_level(confidence_score):
    """
    Determine confidence level based on score
    
    Args:
        confidence_score (float): Model confidence score (0-1)
        
    Returns:
        str: Confidence level (high, medium, low)
    """
    if confidence_score >= MODEL_CONFIG["confidence_threshold"]["high"]:
        return "high"
    elif confidence_score >= MODEL_CONFIG["confidence_threshold"]["medium"]:
        return "medium"
    else:
        return "low"

def get_confidence_color(level):
    """
    Get color based on confidence level
    
    Args:
        level (str): Confidence level
        
    Returns:
        str: Color hex code
    """
    color_map = {
        "high": THEME["success"],
        "medium": THEME["warning"],
        "low": THEME["danger"]
    }
    return color_map.get(level, THEME["text_secondary"])

def load_custom_css():
    """
    Load custom CSS from file or return default styles
    
    Returns:
        str: CSS content
    """
    try:
        if CSS_FILE.exists():
            with open(CSS_FILE, "r") as f:
                return f.read()
    except Exception as e:
        st.warning(f"Could not load custom CSS: {e}")
    
    # Return default CSS if file doesn't exist
    return generate_default_css()

def generate_default_css():
    """
    Generate default CSS styles
    
    Returns:
        str: Default CSS content
    """
    return f"""
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        font-family: 'Inter', sans-serif;
        background: {THEME["background"]};
    }}
    
    /* Hide Streamlit default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Custom animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes slideUp {{
        from {{ transform: translateY(30px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
    }}
    """

# ==================== SESSION STATE INITIALIZATION ====================
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    session_defaults = {
        "analysis_history": [],
        "current_analysis": None,
        "model_loaded": False,
        "show_advanced": False,
        "dark_mode": False,
        "tutorial_completed": False
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ==================== EXPORT CONFIGURATION ====================
__all__ = [
    "APP_CONFIG",
    "THEME", 
    "COMPONENT_STYLES",
    "CONTENT",
    "MODEL_CONFIG",
    "ANIMATIONS",
    "BREAKPOINTS",
    "MODEL_PATH",
    "get_confidence_level",
    "get_confidence_color", 
    "load_custom_css",
    "initialize_session_state"
]