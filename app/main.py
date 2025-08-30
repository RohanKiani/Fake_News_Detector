"""
Main Streamlit Application for Fake News Detector
Entry point that orchestrates all components and handles user interactions
"""

import streamlit as st
import sys
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent  # <-- parent of 'app', i.e., project root
sys.path.insert(0, str(project_root))

try:
    # Import application modules
    from config import (
        APP_CONFIG, THEME, CONTENT, MODEL_CONFIG, 
        initialize_session_state, load_custom_css
    )
    from layout_manager import (
        configure_page, setup_sidebar, render_navigation_tabs,
        render_main_analysis_page, render_dashboard_page,
        render_how_it_works_page, render_settings_page, render_footer
    )
    from ui_components import (
        render_app_header, render_custom_css, render_loading_animation,
        render_result_card, render_text_input_area, render_analyze_button
    )
    from src.model_utils import (
        load_fake_news_model, predict_fake_news, format_prediction_result,
        save_analysis_to_history, check_model_health, get_model_info
    )
    from src.feature_extraction import (
        extract_linguistic_features, extract_key_phrases, 
        create_feature_summary, analyze_text_similarity
    )
    
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please ensure all required modules are in the correct directories.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MAIN APPLICATION CLASS ====================
class FakeNewsDetectorApp:
    """
    Main application class that orchestrates the fake news detection system
    """
    
    def __init__(self):
        """Initialize the application"""
        self.model = None
        self.model_loaded = False
        self.initialization_complete = False
        
    def initialize_app(self):
        """Initialize all application components"""
        try:
            # Configure Streamlit page
            configure_page()
            
            # Initialize session state
            initialize_session_state()
            
            # Load custom CSS
            render_custom_css()
            
            # Try to load model
            self.load_model()
            
            self.initialization_complete = True
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.error(f"Application initialization error: {e}")
            st.error(f"❌ Initialization failed: {str(e)}")
            self.initialization_complete = False
    
    def load_model(self):
        """Load the machine learning model"""
        try:
            with st.spinner("🤖 Loading AI model..."):
                self.model = load_fake_news_model()
                
            if self.model is not None:
                self.model_loaded = True
                st.session_state.model_loaded = True
                logger.info("Model loaded successfully")
            else:
                self.model_loaded = False
                st.session_state.model_loaded = False
                logger.warning("Model loading failed")
                
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.model_loaded = False
            st.session_state.model_loaded = False
    
    def run_analysis(self, text: str) -> dict:
        """
        Run fake news analysis on input text
        
        Args:
            text (str): News text to analyze
            
        Returns:
            dict: Analysis results
        """
        try:
            if not self.model_loaded:
                raise ValueError("Model not loaded")
            
            if not text or not text.strip():
                raise ValueError("No text provided for analysis")
            
            # Show loading animation
            with st.spinner("🔍 Analyzing content..."):
                # Add artificial delay for better UX
                time.sleep(0.5)  # Reduced delay for faster debugging
                
                try:
                    # Run prediction with detailed error handling
                    logger.info(f"Starting prediction for text of length: {len(text)}")
                    prediction, confidence, probabilities = predict_fake_news(text)
                    logger.info("Prediction completed successfully")
                    
                except Exception as pred_error:
                    logger.error(f"Prediction failed: {pred_error}")
                    st.error(f"❌ Prediction Error: {str(pred_error)}")
                    
                    # Show detailed error information
                    with st.expander("🔍 Error Details", expanded=True):
                        st.code(f"""
Error Type: {type(pred_error).__name__}
Error Message: {str(pred_error)}

Text Length: {len(text)}
Text Preview: {text[:200]}...

Model Loaded: {self.model_loaded}
Model Object: {self.model is not None}
                        """)
                    
                    # Try to provide helpful suggestions
                    st.markdown("### 💡 Troubleshooting Suggestions:")
                    st.markdown("""
                    1. **Check Model File**: Ensure `fake_news_model.pkl` is in the correct location
                    2. **Verify Model Training**: The model might be corrupted or incompatible
                    3. **Text Format**: Try with simpler text (e.g., "This is a test news article")
                    4. **Restart App**: Use the restart button in the sidebar
                    """)
                    
                    return None
                
                # Format results
                result = format_prediction_result(
                    prediction, confidence, probabilities, text
                )
                
                # Extract additional features if advanced mode is enabled
                if st.session_state.get('show_advanced', False):
                    try:
                        result['linguistic_features'] = extract_linguistic_features(text)
                        result['key_phrases'] = extract_key_phrases(text)
                        result['feature_summary'] = create_feature_summary(
                            result['linguistic_features']
                        )
                    except Exception as feature_error:
                        logger.warning(f"Feature extraction failed: {feature_error}")
                        # Continue without advanced features
                
                # Save to history
                try:
                    save_analysis_to_history(result, text)
                except Exception as history_error:
                    logger.warning(f"Failed to save to history: {history_error}")
                
                # Store in session state
                st.session_state.current_analysis = result
                
                logger.info(f"Analysis completed: {result['prediction_label']} ({result['confidence']:.3f})")
                
                return result
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            st.error(f"❌ Analysis failed: {str(e)}")
            
            # Provide detailed error information for debugging
            with st.expander("🔧 Debug Information", expanded=False):
                import traceback
                st.code(traceback.format_exc())
            
            return None
    
    def render_main_interface(self):
        """Render the main application interface"""
        try:
            # Setup sidebar
            setup_sidebar()
            
            # Render main header
            render_app_header()
            
            # Create navigation tabs
            tab1, tab2, tab3, tab4 = render_navigation_tabs()
            
            # Handle each tab
            with tab1:
                self.render_analysis_tab()
            
            with tab2:
                self.render_dashboard_tab()
            
            with tab3:
                self.render_how_it_works_tab()
            
            with tab4:
                self.render_settings_tab()
            
            # Render footer
            render_footer()
            
        except Exception as e:
            logger.error(f"Interface rendering error: {e}")
            st.error(f"❌ Interface error: {str(e)}")
    
    def render_analysis_tab(self):
        """Render the main analysis tab"""
        try:
            # Check model status
            if not self.model_loaded:
                st.warning("⚠️ Model not loaded. Please check the model file.")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🔄 Retry Loading Model", type="primary"):
                        self.load_model()
                        st.rerun()
                return
            
            # Main analysis interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Text input section
                st.markdown(f"""
                <div style="
                    background: {THEME['surface']};
                    border-radius: 20px;
                    padding: 2rem;
                    border: 1px solid {THEME['text_light']};
                    margin-bottom: 2rem;
                ">
                """, unsafe_allow_html=True)
                
                # Input area
                news_text = render_text_input_area()
                
                # Analyze button
                analyze_clicked = render_analyze_button()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Handle analysis
                if analyze_clicked and news_text:
                    result = self.run_analysis(news_text)
                    if result:
                        st.success("✅ Analysis completed successfully!")
                
                # Display results if available
                if st.session_state.get('current_analysis'):
                    self.render_analysis_results()
            
            with col2:
                # Tips and instructions
                self.render_analysis_tips()
        
        except Exception as e:
            logger.error(f"Analysis tab error: {e}")
            st.error(f"❌ Analysis tab error: {str(e)}")
    
    def render_analysis_results(self):
        """Render analysis results"""
        try:
            analysis = st.session_state.current_analysis
            
            # Main result card
            render_result_card(
                analysis['prediction'],
                analysis['confidence'],
                analysis['probabilities']['fake']
            )
            
            # Advanced analysis if enabled
            if st.session_state.get('show_advanced', False):
                self.render_advanced_results(analysis)
                
        except Exception as e:
            logger.error(f"Results rendering error: {e}")
            st.error(f"❌ Results rendering error: {str(e)}")
    
    def render_advanced_results(self, analysis):
        """Render advanced analysis results"""
        try:
            st.markdown("## 🔬 Advanced Analysis")
            
            # Feature importance
            if 'linguistic_features' in analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Key Metrics")
                    features = analysis['linguistic_features']
                    
                    # Display key metrics
                    st.metric("Word Count", int(features.get('word_count', 0)))
                    st.metric("Reading Ease", f"{features.get('flesch_reading_ease', 0):.1f}")
                    st.metric("Sentiment Score", f"{features.get('sentiment_compound', 0):.2f}")
                    st.metric("Fake Indicators", int(features.get('total_fake_indicators', 0)))
                
                with col2:
                    st.markdown("### 🔍 Key Phrases")
                    if 'key_phrases' in analysis:
                        for phrase, score in analysis['key_phrases'][:5]:
                            st.write(f"• **{phrase}** ({score:.3f})")
            
            # Feature summary
            if 'feature_summary' in analysis:
                summary = analysis['feature_summary']
                
                if summary.get('risk_factors'):
                    st.markdown("### ⚠️ Risk Factors")
                    for factor in summary['risk_factors']:
                        st.warning(f"• {factor}")
                
                if summary.get('key_insights'):
                    st.markdown("### 💡 Key Insights")
                    for category, insight in summary['key_insights'].items():
                        st.info(f"**{category.title()}:** {insight}")
        
        except Exception as e:
            logger.error(f"Advanced results error: {e}")
            st.error(f"❌ Advanced results error: {str(e)}")
    
    def render_analysis_tips(self):
        """Render analysis tips and instructions"""
        try:
            # Instructions card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {THEME['secondary']}15 0%, {THEME['primary']}10 100%);
                border-radius: 20px;
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid {THEME['secondary']}50;
            ">
                <h3 style="color: {THEME['text_primary']}; margin-bottom: 1.5rem;">
                    📋 How to Use
                </h3>
            """, unsafe_allow_html=True)
            
            for i, step in enumerate(CONTENT['instructions']['steps'], 1):
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding: 1rem;
                    background: {THEME['surface']};
                    border-radius: 10px;
                ">
                    <span style="
                        background: {THEME['primary']};
                        color: white;
                        border-radius: 50%;
                        width: 24px;
                        height: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 0.8rem;
                        font-weight: bold;
                        margin-right: 1rem;
                        flex-shrink: 0;
                    ">{i}</span>
                    <span style="color: {THEME['text_primary']};">{step.split(' ', 1)[1]}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Pro tips
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {THEME['info']}20 0%, {THEME['primary']}10 100%);
                border-radius: 15px;
                padding: 1.5rem;
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
            
        except Exception as e:
            logger.error(f"Tips rendering error: {e}")
    
    def render_dashboard_tab(self):
        """Render the dashboard tab"""
        try:
            render_dashboard_page()
        except Exception as e:
            logger.error(f"Dashboard tab error: {e}")
            st.error(f"❌ Dashboard error: {str(e)}")
    
    def render_how_it_works_tab(self):
        """Render the how it works tab"""
        try:
            render_how_it_works_page()
        except Exception as e:
            logger.error(f"How it works tab error: {e}")
            st.error(f"❌ How it works error: {str(e)}")
    
    def render_settings_tab(self):
        """Render the settings tab"""
        try:
            render_settings_page()
        except Exception as e:
            logger.error(f"Settings tab error: {e}")
            st.error(f"❌ Settings error: {str(e)}")
    
    def run(self):
        """Main application entry point"""
        try:
            # Initialize application
            if not self.initialization_complete:
                self.initialize_app()
            
            # Check if initialization was successful
            if not self.initialization_complete:
                st.error("❌ Application failed to initialize properly.")
                st.stop()
            
            # Render main interface
            self.render_main_interface()
            
        except Exception as e:
            logger.error(f"Application runtime error: {e}")
            st.error("❌ An unexpected error occurred.")
            
            # Show error details in development
            if st.checkbox("🔧 Show Error Details (Development Mode)"):
                st.code(traceback.format_exc())
            
            # Offer restart option
            if st.button("🔄 Restart Application"):
                st.rerun()

# ==================== UTILITY FUNCTIONS ====================
def show_app_info():
    """Display application information"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📱 App Info")
    
    with st.sidebar.expander("ℹ️ About", expanded=False):
        st.markdown(f"""
        **{CONTENT['app_title']}**
        
        Version: {CONTENT['footer']['version']}
        
        An AI-powered fake news detection system using advanced 
        natural language processing and machine learning techniques.
        
        **Features:**
        - Real-time analysis
        - Confidence scoring
        - Advanced linguistics analysis
        - Interactive dashboard
        """)
    
    # Model health check
    with st.sidebar.expander("🏥 Model Health", expanded=False):
        health_status = check_model_health()
        
        if health_status['overall_status'] == 'healthy':
            st.success("✅ All systems operational")
        elif health_status['overall_status'] == 'warning':
            st.warning("⚠️ Some issues detected")
        else:
            st.error("❌ Critical issues found")
        
        for check_name, check_result in health_status.get('checks', {}).items():
            status_icon = "✅" if check_result['status'] == 'pass' else "❌"
            st.write(f"{status_icon} {check_name.replace('_', ' ').title()}")

def handle_error_recovery():
    """Handle application errors and provide recovery options"""
    st.error("🚨 Application Error Detected")
    
    st.markdown("""
    ### Possible Solutions:
    1. **Refresh the page** - Simple browser refresh often resolves temporary issues
    2. **Check model file** - Ensure `fake_news_model.pkl` exists in the root directory
    3. **Restart application** - Use the restart button below
    4. **Clear cache** - Clear Streamlit cache and reload
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart App"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    with col3:
        if st.button("🏠 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ==================== MAIN ENTRY POINT ====================
def main():
    """Main application entry point"""
    try:
        # Create and run application
        app = FakeNewsDetectorApp()
        app.run()
        
        # Show app info in sidebar
        show_app_info()
        
    except Exception as e:
        logger.critical(f"Critical application error: {e}")
        handle_error_recovery()

# ==================== APPLICATION EXECUTION ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        st.error("❌ Fatal application error. Please restart.")
        
        # Emergency error handling
        if st.button("🆘 Emergency Restart"):
            st.rerun()