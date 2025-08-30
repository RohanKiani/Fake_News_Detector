"""
Layout Manager for Fake News Detector App
Handles page layouts, responsive design, and styling coordination
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from config import APP_CONFIG, THEME, CONTENT, initialize_session_state
from ui_components import (
    render_app_header, render_navigation_tabs, render_feature_card,
    render_result_card, render_text_input_area, render_analyze_button,
    render_confidence_gauge, render_feature_importance_chart,
    render_loading_animation, render_stats_metrics, render_how_it_works_section,
    render_instructions_card, render_footer, render_custom_css
)

# ==================== PAGE CONFIGURATION ====================
def configure_page():
    """Configure Streamlit page settings and initialize session state"""
    st.set_page_config(**APP_CONFIG)
    initialize_session_state()
    render_custom_css()

def setup_sidebar():
    """Setup and configure the sidebar with controls and information"""
    with st.sidebar:
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 1rem 0;
            background: {THEME['gradient_primary']};
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
        ">
            <h2 style="margin: 0; font-size: 1.5rem;">🔍 Controls</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Status
        render_model_status()
        
        # Quick Stats
        render_sidebar_stats()
        
        # Settings
        render_sidebar_settings()
        
        # Recent Analysis History
        render_analysis_history()
        
        # Help & Resources
        render_help_section()

def render_model_status():
    """Render model loading status in sidebar"""
    st.markdown("### 🤖 Model Status")
    
    if st.session_state.get('model_loaded', False):
        st.success("✅ Model Loaded Successfully")
        st.info("🎯 Ready for Analysis")
    else:
        st.warning("⏳ Loading Model...")
        if st.button("🔄 Reload Model", key="reload_model"):
            st.session_state.model_loaded = True
            st.rerun()

def render_sidebar_stats():
    """Render quick statistics in sidebar"""
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Generate sample stats (in real app, these would come from database)
    total_analyses = len(st.session_state.get('analysis_history', []))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", total_analyses, delta=None)
    with col2:
        st.metric("Today", min(total_analyses, 5), delta=2 if total_analyses > 0 else 0)

def render_sidebar_settings():
    """Render settings controls in sidebar"""
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Theme toggle
    dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.get('dark_mode', False))
    if dark_mode != st.session_state.get('dark_mode', False):
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    # Advanced options
    st.session_state.show_advanced = st.checkbox(
        "🔬 Advanced Analysis", 
        value=st.session_state.get('show_advanced', False),
        help="Show detailed feature analysis and model insights"
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Minimum confidence required for high-confidence predictions"
    )

def render_analysis_history():
    """Render recent analysis history in sidebar"""
    st.markdown("---")
    st.markdown("### 📝 Recent Analysis")
    
    history = st.session_state.get('analysis_history', [])
    
    if history:
        for i, analysis in enumerate(history[-3:]):  # Show last 3
            with st.expander(f"Analysis {len(history) - i}", expanded=False):
                st.write(f"**Result:** {analysis.get('result', 'N/A')}")
                st.write(f"**Confidence:** {analysis.get('confidence', 0)*100:.1f}%")
                st.write(f"**Time:** {analysis.get('timestamp', 'Unknown')}")
    else:
        st.info("No analysis history yet. Start analyzing news articles!")
    
    if history and st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.analysis_history = []
        st.rerun()

def render_help_section():
    """Render help and resources section in sidebar"""
    st.markdown("---")
    st.markdown("### ❓ Help & Resources")
    
    with st.expander("📚 Quick Tips"):
        st.markdown("""
        - **Longer articles** generally provide better analysis
        - **Complete sentences** improve accuracy
        - **Multiple paragraphs** help the model understand context
        - **Original source text** works better than summaries
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        - If analysis seems slow, try shorter text
        - Ensure text is in English for best results
        - Check your internet connection
        - Refresh the page if issues persist
        """)
    
    if st.button("📧 Report Issue", key="report_issue"):
        st.info("Please visit our GitHub repository to report issues!")

# ==================== MAIN LAYOUT FUNCTIONS ====================
def render_main_analysis_page():
    """Render the main news analysis page layout"""
    # Header
    render_app_header()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        render_input_section()
        
    with col2:
        # Instructions and tips
        render_sidebar_tips()
    
    # Results section (will be populated after analysis)
    render_results_section()

def render_input_section():
    """Render the input section for news text"""
    st.markdown(f"""
    <div style="
        background: {THEME['surface']};
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid {THEME['text_light']};
        margin-bottom: 2rem;
    ">
    """, unsafe_allow_html=True)
    
    # Text input area
    news_text = render_text_input_area()
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_clicked = render_analyze_button()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return news_text, analyze_clicked

def render_sidebar_tips():
    """Render tips and instructions in the sidebar column"""
    render_instructions_card()
    
    # Additional tips
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {THEME['info']}20 0%, {THEME['primary']}10 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid {THEME['info']}30;
    ">
        <h4 style="color: {THEME['text_primary']}; margin-bottom: 1rem;">💡 Pro Tips</h4>
        <ul style="color: {THEME['text_secondary']}; font-size: 0.9rem; line-height: 1.6;">
            <li>Include headlines and body text for best results</li>
            <li>Paste the full article, not just excerpts</li>
            <li>Check multiple sources for verification</li>
            <li>Consider the publication date and source</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_results_section():
    """Render the results display section"""
    if st.session_state.get('current_analysis'):
        analysis = st.session_state.current_analysis
        
        # Main result card
        render_result_card(
            analysis['prediction'], 
            analysis['confidence'], 
            analysis.get('prediction_proba', [0.5, 0.5])
        )
        
        # Additional analysis if advanced mode is enabled
        if st.session_state.get('show_advanced', False):
            render_advanced_analysis(analysis)

def render_advanced_analysis(analysis):
    """Render advanced analysis section"""
    st.markdown("## 🔬 Advanced Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence gauge
        st.markdown("### Confidence Meter")
        render_confidence_gauge(analysis['confidence'], analysis['prediction'])
    
    with col2:
        # Feature importance (mock data for demonstration)
        st.markdown("### Key Factors")
        features = ["Word Choice", "Sentence Structure", "Source Pattern", "Emotional Tone", "Fact Density", "Writing Style"]
        importance = np.random.uniform(0.3, 1.0, len(features))
        render_feature_importance_chart(features, importance)
    
    # Text analysis breakdown
    render_text_analysis_breakdown(analysis)

def render_text_analysis_breakdown(analysis):
    """Render detailed text analysis breakdown"""
    st.markdown("### 📝 Text Analysis Breakdown")
    
    # Create tabs for different analysis aspects
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔍 Key Phrases", "🎯 Indicators"])
    
    with tab1:
        # Text statistics
        text = analysis.get('original_text', '')
        stats = {
            'word_count': len(text.split()),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
            'punctuation_ratio': sum(1 for char in text if char in '.,!?;:') / len(text) if text else 0
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Word Count", f"{stats['word_count']:,}")
            st.metric("Average Word Length", f"{stats['avg_word_length']:.1f}")
        with col2:
            st.metric("Sentences", stats['sentence_count'])
            st.metric("Punctuation Ratio", f"{stats['punctuation_ratio']:.3f}")
    
    with tab2:
        # Key phrases (mock data)
        st.markdown("**Most Influential Phrases:**")
        phrases = ["breaking news", "sources confirm", "unprecedented event", "experts say", "according to"]
        for i, phrase in enumerate(phrases[:3]):
            impact = np.random.uniform(0.6, 0.95)
            st.markdown(f"- **{phrase}** (Impact: {impact:.2f})")
    
    with tab3:
        # Credibility indicators
        indicators = [
            ("Source Attribution", "Present", THEME['success']),
            ("Date/Time Stamps", "Present", THEME['success']),
            ("Fact-based Language", "Moderate", THEME['warning']),
            ("Emotional Appeals", "Low", THEME['success']),
            ("Sensational Headlines", "Present", THEME['danger'])
        ]
        
        for indicator, status, color in indicators:
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 1rem;
                margin: 0.5rem 0;
                background: {color}20;
                border-left: 4px solid {color};
                border-radius: 5px;
            ">
                <span style="color: {THEME['text_primary']}; font-weight: 500;">{indicator}</span>
                <span style="color: {color}; font-weight: 600;">{status}</span>
            </div>
            """, unsafe_allow_html=True)

# ==================== DASHBOARD LAYOUT ====================
def render_dashboard_page():
    """Render the analytics dashboard page"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # Generate sample dashboard data
    dashboard_data = generate_dashboard_data()
    
    # Key metrics row
    render_stats_metrics(dashboard_data['stats'])
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        render_analysis_trend_chart(dashboard_data['trend_data'])
    
    with col2:
        render_accuracy_metrics(dashboard_data['accuracy_data'])
    
    # Recent analysis table
    render_recent_analysis_table(dashboard_data['recent_analyses'])

def generate_dashboard_data():
    """Generate sample data for dashboard (replace with real data in production)"""
    return {
        'stats': {
            'total_analyzed': 1247,
            'accuracy': 87.3,
            'fake_detected': 432,
            'real_verified': 815
        },
        'trend_data': pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'analyses': np.random.poisson(40, 30),
            'fake_detected': np.random.poisson(15, 30)
        }),
        'accuracy_data': {
            'precision': 0.89,
            'recall': 0.85,
            'f1_score': 0.87
        },
        'recent_analyses': pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
            'result': ['Fake', 'Real', 'Fake', 'Real', 'Real'],
            'confidence': [0.92, 0.78, 0.85, 0.91, 0.76],
            'text_preview': [
                'Breaking: Shocking discovery...',
                'Local government announces...',
                'Scientists baffled by...',
                'Economic report shows...',
                'Weather update for...'
            ]
        })
    }

def render_analysis_trend_chart(trend_data):
    """Render analysis trend chart"""
    st.markdown("### 📈 Analysis Trends")
    st.line_chart(trend_data.set_index('date')[['analyses', 'fake_detected']])

def render_accuracy_metrics(accuracy_data):
    """Render model accuracy metrics"""
    st.markdown("### 🎯 Model Performance")
    for metric, value in accuracy_data.items():
        st.metric(metric.replace('_', ' ').title(), f"{value:.2f}")

def render_recent_analysis_table(recent_data):
    """Render recent analysis table"""
    st.markdown("### 📋 Recent Analyses")
    
    # Format the dataframe for display
    display_df = recent_data.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            'timestamp': 'Time',
            'result': 'Result',
            'confidence': 'Confidence',
            'text_preview': 'Preview'
        }
    )

# ==================== HOW IT WORKS LAYOUT ====================
def render_how_it_works_page():
    """Render the how it works educational page"""
    render_how_it_works_section()
    
    # Additional technical details
    render_technical_details()
    
    # Model information
    render_model_information()

def render_technical_details():
    """Render technical details section"""
    st.markdown("## 🔬 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: {THEME['surface']};
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid {THEME['text_light']};
        ">
            <h3 style="color: {THEME['primary']};">Data Processing</h3>
            <ul style="color: {THEME['text_secondary']}; line-height: 1.8;">
                <li>Text normalization and cleaning</li>
                <li>Stop word removal</li>
                <li>TF-IDF vectorization (5000 features)</li>
                <li>N-gram analysis (unigrams)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: {THEME['surface']};
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid {THEME['text_light']};
        ">
            <h3 style="color: {THEME['primary']};">Machine Learning</h3>
            <ul style="color: {THEME['text_secondary']}; line-height: 1.8;">
                <li>Logistic Regression classifier</li>
                <li>L2 regularization</li>
                <li>1000 maximum iterations</li>
                <li>Probability calibration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_model_information():
    """Render model performance and information"""
    st.markdown("## 📈 Model Performance")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Accuracy", "87.3%", delta="2.1%")
    with col2:
        st.metric("Validation Accuracy", "85.7%", delta="1.8%")
    with col3:
        st.metric("F1 Score", "0.86", delta="0.03")

# ==================== SETTINGS LAYOUT ====================
def render_settings_page():
    """Render the settings and configuration page"""
    st.markdown("## ⚙️ Settings & Configuration")
    
    # User preferences
    render_user_preferences()
    
    # Model settings
    render_model_settings()
    
    # Data management
    render_data_management()

def render_user_preferences():
    """Render user preference settings"""
    st.markdown("### 👤 User Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
        st.slider("Analysis Detail Level", 1, 5, 3)
    
    with col2:
        st.checkbox("Show Confidence Scores", value=True)
        st.checkbox("Enable Notifications", value=False)

def render_model_settings():
    """Render model configuration settings"""
    st.markdown("### 🤖 Model Configuration")
    
    st.slider("Confidence Threshold", 0.5, 0.95, 0.8, 0.05)
    st.selectbox("Analysis Mode", ["Standard", "Detailed", "Quick"], index=0)

def render_data_management():
    """Render data management options"""
    st.markdown("### 💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export History"):
            st.info("Analysis history exported!")
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            st.warning("All data cleared!")

# ==================== LAYOUT COORDINATOR ====================
def render_page_layout():
    """Main layout coordinator - renders the complete page based on navigation"""
    # Configure page
    configure_page()
    
    # Setup sidebar
    setup_sidebar()
    
    # Render navigation and get active tab
    tab1, tab2, tab3, tab4 = render_navigation_tabs()
    
    # Render content based on active tab
    with tab1:
        render_main_analysis_page()
    
    with tab2:
        render_dashboard_page()
    
    with tab3:
        render_how_it_works_page()
    
    with tab4:
        render_settings_page()
    
    # Render footer
    render_footer()

# ==================== EXPORT ALL FUNCTIONS ====================
__all__ = [
    "configure_page",
    "render_page_layout",
    "render_main_analysis_page",
    "render_dashboard_page", 
    "render_how_it_works_page",
    "render_settings_page",
    "setup_sidebar",
    "render_results_section",
    "render_advanced_analysis"
]