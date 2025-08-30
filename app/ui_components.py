"""
Reusable UI Components for Fake News Detector App
Contains all custom widgets, cards, and interactive elements
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from config import THEME, COMPONENT_STYLES, CONTENT, MODEL_CONFIG, get_confidence_level, get_confidence_color

# ==================== HEADER COMPONENTS ====================
def render_app_header():
    """Render the main application header with title and subtitle"""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0; animation: fadeIn 1s ease-in;">
        <h1 style="
            font-size: 3.5rem; 
            font-weight: 700; 
            background: {THEME['gradient_primary']}; 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        ">
            {CONTENT['app_title']}
        </h1>
        <p style="
            font-size: 1.3rem; 
            color: {THEME['text_secondary']}; 
            font-weight: 400;
            margin-bottom: 1rem;
        ">
            {CONTENT['app_subtitle']}
        </p>
        <p style="
            font-size: 1rem; 
            color: {THEME['text_secondary']}; 
            max-width: 600px; 
            margin: 0 auto;
            line-height: 1.6;
        ">
            {CONTENT['app_description']}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_navigation_tabs():
    """Render navigation tabs for different sections"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 **Analyze News**", 
        "📊 **Dashboard**", 
        "📚 **How It Works**", 
        "⚙️ **Settings**"
    ])
    return tab1, tab2, tab3, tab4

# ==================== CARD COMPONENTS ====================
def render_feature_card(icon, title, description, col=None):
    """Render a feature highlight card"""
    card_html = f"""
    <div style="
        background: {COMPONENT_STYLES['card']['background']};
        border-radius: {COMPONENT_STYLES['card']['border_radius']};
        padding: {COMPONENT_STYLES['card']['padding']};
        box-shadow: {COMPONENT_STYLES['card']['box_shadow']};
        border: {COMPONENT_STYLES['card']['border']};
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.15)';" 
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='{COMPONENT_STYLES['card']['box_shadow']}';">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: {THEME['text_primary']}; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.8rem;">
            {title}
        </h3>
        <p style="color: {THEME['text_secondary']}; font-size: 0.95rem; line-height: 1.5; margin: 0;">
            {description}
        </p>
    </div>
    """
    
    if col:
        col.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

def render_result_card(prediction, confidence, prediction_proba):
    """Render the main analysis result card"""
    result_config = MODEL_CONFIG['result_labels'][prediction]
    confidence_level = get_confidence_level(confidence)
    confidence_color = get_confidence_color(confidence_level)
    
    # Calculate percentage
    confidence_percent = confidence * 100
    
    result_html = f"""
    <div style="
        background: linear-gradient(135deg, {result_config['color']}15 0%, {result_config['color']}05 100%);
        border: 2px solid {result_config['color']};
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: slideUp 0.8s ease-out;
        margin: 2rem 0;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{result_config['icon']}</div>
        <h2 style="
            color: {result_config['color']}; 
            font-size: 2rem; 
            font-weight: 700; 
            margin-bottom: 1rem;
        ">
            {result_config['label']}
        </h2>
        <p style="
            color: {THEME['text_primary']}; 
            font-size: 1.1rem; 
            margin-bottom: 1.5rem;
            line-height: 1.6;
        ">
            {result_config['message']}
        </p>
        <div style="
            background: {THEME['surface']};
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        ">
            <h4 style="color: {THEME['text_primary']}; margin-bottom: 1rem;">Confidence Score</h4>
            <div style="
                background: {THEME['background']};
                height: 20px;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 0.5rem;
            ">
                <div style="
                    background: {result_config['gradient']};
                    height: 100%;
                    width: {confidence_percent}%;
                    border-radius: 10px;
                    transition: width 1s ease-in-out;
                "></div>
            </div>
            <p style="
                color: {confidence_color}; 
                font-size: 1.2rem; 
                font-weight: 600;
                margin: 0;
            ">
                {confidence_percent:.1f}% ({confidence_level.upper()} confidence)
            </p>
        </div>
    </div>
    """
    
    st.markdown(result_html, unsafe_allow_html=True)

# ==================== INPUT COMPONENTS ====================
def render_text_input_area():
    """Render the main text input area for news content"""
    st.markdown(f"""
    <div style="margin: 2rem 0;">
        <h3 style="
            color: {THEME['text_primary']}; 
            font-size: 1.4rem; 
            font-weight: 600; 
            margin-bottom: 1rem;
        ">
            📝 Enter News Article Content
        </h3>
        <p style="color: {THEME['text_secondary']}; margin-bottom: 1rem;">
            Paste the full text of the news article you want to analyze for authenticity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    news_text = st.text_area(
        label="",
        placeholder="Paste your news article content here...\n\nExample: 'Breaking: Scientists discover new species of...'",
        height=300,
        help="Enter the complete text of the news article for the most accurate analysis.",
        label_visibility="collapsed"
    )
    
    return news_text

def render_analyze_button():
    """Render the main analyze button"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "🚀 Analyze News",
            type="primary",
            use_container_width=True,
            help="Click to analyze the news content using AI"
        )
    
    return analyze_clicked

# ==================== VISUALIZATION COMPONENTS ====================
def render_confidence_gauge(confidence, prediction):
    """Render a confidence gauge chart"""
    result_config = MODEL_CONFIG['result_labels'][prediction]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 20}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': THEME['text_light']},
            'bar': {'color': result_config['color']},
            'bgcolor': THEME['background'],
            'borderwidth': 2,
            'bordercolor': THEME['text_light'],
            'steps': [
                {'range': [0, 50], 'color': THEME['danger'] + '30'},
                {'range': [50, 80], 'color': THEME['warning'] + '30'},
                {'range': [80, 100], 'color': THEME['success'] + '30'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': THEME['text_primary']},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance_chart(features, importance_scores):
    """Render feature importance horizontal bar chart"""
    fig = go.Figure(go.Bar(
        x=importance_scores,
        y=features,
        orientation='h',
        marker=dict(
            color=importance_scores,
            colorscale=[[0, THEME['danger']], [0.5, THEME['warning']], [1, THEME['success']]],
            colorbar=dict(title="Impact Score")
        )
    ))
    
    fig.update_layout(
        title="Key Factors in Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        font={'color': THEME['text_primary']},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== LOADING AND FEEDBACK COMPONENTS ====================
def render_loading_animation(message="Analyzing content..."):
    """Render loading animation with message"""
    loading_html = f"""
    <div style="
        text-align: center; 
        padding: 3rem 2rem;
        background: {THEME['surface']};
        border-radius: 20px;
        border: 1px solid {THEME['text_light']};
        margin: 2rem 0;
    ">
        <div style="
            width: 60px;
            height: 60px;
            border: 4px solid {THEME['text_light']};
            border-top: 4px solid {THEME['primary']};
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 2rem auto;
        "></div>
        <h3 style="color: {THEME['text_primary']}; margin-bottom: 1rem;">{message}</h3>
        <p style="color: {THEME['text_secondary']};">
            Please wait while our AI analyzes the content...
        </p>
    </div>
    
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    
    return st.markdown(loading_html, unsafe_allow_html=True)

def render_progress_bar(progress_value, message="Processing..."):
    """Render animated progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(progress_value + 1):
        progress_bar.progress(i)
        status_text.text(f'{message} {i}%')
        time.sleep(0.01)
    
    return progress_bar, status_text

# ==================== STATISTICS AND METRICS ====================
def render_stats_metrics(stats_data):
    """Render key statistics in metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Analyzed", stats_data.get('total_analyzed', 0), "📊"),
        ("Accuracy Rate", f"{stats_data.get('accuracy', 0):.1f}%", "🎯"),
        ("Fake Detected", stats_data.get('fake_detected', 0), "⚠️"),
        ("Real Verified", stats_data.get('real_verified', 0), "✅")
    ]
    
    for col, (label, value, icon) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div style="
                background: {THEME['surface']};
                padding: 1.5rem;
                border-radius: 15px;
                text-align: center;
                border: 1px solid {THEME['text_light']};
                transition: transform 0.3s ease;
            " onmouseover="this.style.transform='scale(1.05)';" 
               onmouseout="this.style.transform='scale(1)';">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <h3 style="color: {THEME['primary']}; font-size: 1.8rem; margin: 0;">{value}</h3>
                <p style="color: {THEME['text_secondary']}; margin: 0; font-size: 0.9rem;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== TUTORIAL AND HELP COMPONENTS ====================
def render_how_it_works_section():
    """Render the how it works section"""
    st.markdown(f"""
    <div style="
        background: {THEME['surface']};
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid {THEME['text_light']};
    ">
        <h2 style="
            color: {THEME['text_primary']};
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2rem;
        ">
            🧠 How Our AI Works
        </h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
    """, unsafe_allow_html=True)
    
    steps = [
        {
            "number": "1",
            "title": "Text Preprocessing",
            "description": "Clean and normalize the input text, removing noise and standardizing format",
            "icon": "🔤"
        },
        {
            "number": "2", 
            "title": "Feature Extraction",
            "description": "Extract key linguistic patterns using TF-IDF vectorization",
            "icon": "🔍"
        },
        {
            "number": "3",
            "title": "ML Analysis",
            "description": "Apply trained Logistic Regression model to classify content authenticity",
            "icon": "🤖"
        },
        {
            "number": "4",
            "title": "Results & Insights",
            "description": "Generate confidence scores and provide detailed analysis breakdown",
            "icon": "📊"
        }
    ]
    
    for step in steps:
        st.markdown(f"""
        <div style="
            background: {THEME['background']};
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border-left: 4px solid {THEME['primary']};
        ">
            <div style="
                background: {THEME['primary']};
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem auto;
                font-weight: bold;
            ">
                {step['number']}
            </div>
            <div style="font-size: 2rem; margin-bottom: 1rem;">{step['icon']}</div>
            <h3 style="color: {THEME['text_primary']}; margin-bottom: 1rem;">{step['title']}</h3>
            <p style="color: {THEME['text_secondary']}; font-size: 0.9rem; line-height: 1.5;">
                {step['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_instructions_card():
    """Render usage instructions card"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {THEME['secondary']}15 0%, {THEME['primary']}10 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid {THEME['secondary']}50;
    ">
        <h3 style="
            color: {THEME['text_primary']};
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        ">
            {CONTENT['instructions']['title']}
        </h3>
    """, unsafe_allow_html=True)
    
    for step in CONTENT['instructions']['steps']:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background: {THEME['surface']};
            border-radius: 10px;
        ">
            <span style="font-size: 1.2rem; margin-right: 1rem;">{step}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER COMPONENTS ====================
def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: {THEME['text_secondary']};
        font-size: 0.9rem;
    ">
        <p>{CONTENT['footer']['text']}</p>
        <p style="margin-top: 0.5rem;">
            {CONTENT['footer']['version']} | Last Updated: {CONTENT['footer']['last_updated']}
        </p>
        <p style="
            margin-top: 1rem;
            font-size: 0.8rem;
            color: {THEME['text_light']};
        ">
            {CONTENT['disclaimer']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== UTILITY COMPONENTS ====================
def render_custom_css():
    """Render custom CSS styles"""
    st.markdown(f"""
    <style>
        /* Custom Streamlit Styling */
        .stTextArea textarea {{
            border-radius: 10px !important;
            border: 2px solid {THEME['text_light']} !important;
            font-size: 1rem !important;
            transition: border-color 0.3s ease !important;
        }}
        
        .stTextArea textarea:focus {{
            border-color: {THEME['primary']} !important;
            box-shadow: 0 0 0 2px {THEME['primary']}20 !important;
        }}
        
        .stButton > button {{
            background: {THEME['gradient_primary']} !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.3s ease !important;
            height: auto !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px {THEME['primary']}40 !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2rem;
            background: {THEME['surface']};
            border-radius: 15px;
            padding: 0.5rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px !important;
            padding: 1rem 2rem !important;
            font-weight: 600 !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {THEME['gradient_primary']} !important;
            color: white !important;
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(30px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .stApp {{
                padding: 1rem !important;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

# ==================== EXPORT ALL FUNCTIONS ====================
__all__ = [
    "render_app_header",
    "render_navigation_tabs",
    "render_feature_card", 
    "render_result_card",
    "render_text_input_area",
    "render_analyze_button",
    "render_confidence_gauge",
    "render_feature_importance_chart",
    "render_loading_animation",
    "render_progress_bar",
    "render_stats_metrics",
    "render_how_it_works_section",
    "render_instructions_card",
    "render_footer",
    "render_custom_css"
]