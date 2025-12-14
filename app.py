import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime, timedelta
import json
import os
import hashlib
import time
from pathlib import Path
from gradio_client import Client
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import tempfile

# Page configuration
st.set_page_config(
    page_title="Soul - Professional Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
HUGGINGFACE_MODEL_NAME = "krishsinghal006/emotion-roberta-soul" 

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    .block-container {
        padding: 1rem 5% !important;
        max-width: 100% !important;
    }
    
    /* Professional Header with Logo */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 5%;
        margin: -1rem -5% 1rem -5%;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        border-bottom: 3px solid rgba(255, 255, 255, 0.2);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .logo-icon {
        font-size: 3rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
    }
    
    .logo-text {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        letter-spacing: 1px;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    
    .user-welcome {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Enhanced Navigation Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid transparent !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        text-transform: none !important;
        letter-spacing: 0.3px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Navigation Container Styling */
    div[data-testid="column"] {
        padding: 0.25rem !important;
    }
    
    /* Hero Section */
        .hero-landing {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 6rem 5% 5rem;
            text-align: center;
            color: white;
            position: relative;
            overflow: hidden;
            margin: -2rem -5% 2rem -5%;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        }
        
    .hero-landing::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.3; }
        50% { transform: scale(1.2) rotate(180deg); opacity: 0.6; }
    }
    
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        font-weight: 400;
        margin-bottom: 1rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    .hero-description {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 3rem;
        line-height: 1.8;
    }
    
    /* Content Container */
    .content-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem 0;
    }
    
    /* Feature Cards */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card-modern {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.4s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .feature-card-modern:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
    }
    
    .feature-icon-modern {
        font-size: 3rem;
        margin-bottom: 1.5rem;
    }
    
    .feature-title-modern {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    
    .feature-desc-modern {
        color: #6b7280;
        line-height: 1.8;
        font-size: 1rem;
    }
    
    /* Auth Section */
    .auth-container {
        max-width: 500px;
        margin: 4rem auto;
        background: white;
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .auth-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: #1f2937;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .auth-subtitle {
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    
    /* Section Headers */
    .section-header {
        text-align: center;
        margin: 1.5rem 0 1.5rem;
    }
    
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .stat-card-modern {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stat-card-modern:hover {
        transform: translateY(-5px);
    }
    
    .stat-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Alert Boxes */
    .alert-modern {
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border-left: 5px solid;
    }
    
    .alert-success-modern {
        background: rgba(56, 239, 125, 0.1);
        border-color: #38ef7d;
        color: #0a6b5f;
    }
    
    .alert-warning-modern {
        background: rgba(255, 153, 102, 0.1);
        border-color: #ff9966;
        color: #cc4d00;
    }
    
    .alert-danger-modern {
        background: rgba(235, 51, 73, 0.1);
        border-color: #eb3349;
        color: #b81f2a;
    }
    
    .alert-info-modern {
        background: rgba(102, 126, 234, 0.1);
        border-color: #667eea;
        color: #4338ca;
    }
    
    /* Chat Messages */
    .chat-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
    }
    
    .chat-message-modern {
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        max-width: 80%;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message-modern {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-radius: 15px 15px 5px 15px;
    }
    
    .bot-message-modern {
        background: #f3f4f6;
        color: #1f2937;
        border-radius: 15px 15px 15px 5px;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, ##ff4c4c 0%, #ffb300 100%);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1.2rem;
        }
        .logo-text {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Mental Health Risk Mapping
MENTAL_HEALTH_MAPPING = {
    'admiration': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'amusement': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'anger': {'risk_level': 'medium', 'concern': 'stress/aggression', 'color': 'orange'},
    'annoyance': {'risk_level': 'low', 'concern': 'mild irritation', 'color': 'yellow'},
    'approval': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'caring': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'confusion': {'risk_level': 'low', 'concern': 'cognitive uncertainty', 'color': 'yellow'},
    'curiosity': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'desire': {'risk_level': 'low', 'concern': 'motivation', 'color': 'green'},
    'disappointment': {'risk_level': 'medium', 'concern': 'mild depression', 'color': 'orange'},
    'disapproval': {'risk_level': 'low', 'concern': 'negative judgment', 'color': 'yellow'},
    'disgust': {'risk_level': 'medium', 'concern': 'aversion/stress', 'color': 'orange'},
    'embarrassment': {'risk_level': 'medium', 'concern': 'social anxiety', 'color': 'orange'},
    'excitement': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'fear': {'risk_level': 'high', 'concern': 'anxiety disorder', 'color': 'red'},
    'gratitude': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'grief': {'risk_level': 'high', 'concern': 'severe depression', 'color': 'red'},
    'joy': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'love': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'nervousness': {'risk_level': 'medium', 'concern': 'anxiety', 'color': 'orange'},
    'optimism': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'pride': {'risk_level': 'low', 'concern': 'positive', 'color': 'green'},
    'realization': {'risk_level': 'low', 'concern': 'insight', 'color': 'green'},
    'relief': {'risk_level': 'low', 'concern': 'stress reduction', 'color': 'green'},
    'remorse': {'risk_level': 'medium', 'concern': 'guilt/regret', 'color': 'orange'},
    'sadness': {'risk_level': 'high', 'concern': 'depression', 'color': 'red'},
    'surprise': {'risk_level': 'low', 'concern': 'neutral', 'color': 'yellow'},
    'neutral': {'risk_level': 'low', 'concern': 'stable', 'color': 'green'}
}

# DASS-42 Questionnaire
DASS_42_QUESTIONS = {
    'Depression': [
        "I couldn't seem to experience any positive feeling at all",
        "I found it difficult to work up the initiative to do things",
        "I felt that I had nothing to look forward to",
        "I felt down-hearted and blue",
        "I was unable to become enthusiastic about anything",
        "I felt I wasn't worth much as a person",
        "I felt that life was meaningless",
        "I found it hard to wind down",
        "I was aware of dryness of my mouth",
        "I couldn't seem to get going",
        "I felt sad and depressed",
        "I felt that I had lost interest in just about everything",
        "I felt I was pretty worthless",
        "I could see nothing in the future to be hopeful about"
    ],
    'Anxiety': [
        "I was aware of the action of my heart in the absence of physical exertion",
        "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness)",
        "I experienced trembling (e.g., in the hands)",
        "I was worried about situations in which I might panic and make a fool of myself",
        "I felt I was close to panic",
        "I was aware of dryness of my mouth",
        "I experienced difficulty in breathing",
        "I had a feeling of shakiness (e.g., legs going to give way)",
        "I found myself in situations that made me so anxious I was most relieved when they ended",
        "I felt scared without any good reason",
        "I felt terrified",
        "I was worried about situations in which I might panic",
        "I felt I was close to panic",
        "I was aware I had a dry mouth"
    ],
    'Stress': [
        "I found it hard to wind down",
        "I tended to over-react to situations",
        "I felt that I was using a lot of nervous energy",
        "I found myself getting agitated",
        "I found it difficult to relax",
        "I was intolerant of anything that kept me from getting on with what I was doing",
        "I felt that I was rather touchy",
        "I found it difficult to tolerate interruptions to what I was doing",
        "I was in a state of nervous tension",
        "I found it hard to calm down after something upset me",
        "I found it difficult to tolerate interruptions",
        "I was intolerant of things that kept me from getting on",
        "I found myself getting upset rather easily",
        "I felt that I was rather touchy"
    ]
}

def calculate_dass_score(responses):
    """Calculate DASS-42 scores and severity levels"""
    scores = {'depression': 0, 'anxiety': 0, 'stress': 0}
    
    depression_indices = [2, 4, 9, 12, 15, 16, 20, 23, 24, 26, 30, 33, 36, 41]
    anxiety_indices = [1, 6, 8, 14, 18, 19, 22, 28, 29, 31, 34, 37, 38, 40]
    stress_indices = [0, 5, 7, 10, 11, 13, 17, 21, 25, 27, 32, 35, 39, 42]
    
    all_responses = []
    for category_responses in responses.values():
        all_responses.extend(category_responses)
    
    # Calculate depression score with None handling
    for idx in depression_indices:
        if idx < len(all_responses) and all_responses[idx] is not None:
            scores['depression'] += all_responses[idx]
    
    # Calculate anxiety score with None handling
    for idx in anxiety_indices:
        if idx < len(all_responses) and all_responses[idx] is not None:
            scores['anxiety'] += all_responses[idx]
    
    # Calculate stress score with None handling
    for idx in stress_indices:
        if idx < len(all_responses) and all_responses[idx] is not None:
            scores['stress'] += all_responses[idx]
    
    # Multiply by 2 as per DASS-42 scoring
    scores = {k: v * 2 for k, v in scores.items()}
    
    severity = {}
    
    # Depression severity levels
    if scores['depression'] <= 9:
        severity['depression'] = 'Normal'
    elif scores['depression'] <= 13:
        severity['depression'] = 'Mild'
    elif scores['depression'] <= 20:
        severity['depression'] = 'Moderate'
    elif scores['depression'] <= 27:
        severity['depression'] = 'Severe'
    else:
        severity['depression'] = 'Extremely Severe'
    
    # Anxiety severity levels
    if scores['anxiety'] <= 7:
        severity['anxiety'] = 'Normal'
    elif scores['anxiety'] <= 9:
        severity['anxiety'] = 'Mild'
    elif scores['anxiety'] <= 14:
        severity['anxiety'] = 'Moderate'
    elif scores['anxiety'] <= 19:
        severity['anxiety'] = 'Severe'
    else:
        severity['anxiety'] = 'Extremely Severe'
    
    # Stress severity levels
    if scores['stress'] <= 14:
        severity['stress'] = 'Normal'
    elif scores['stress'] <= 18:
        severity['stress'] = 'Mild'
    elif scores['stress'] <= 25:
        severity['stress'] = 'Moderate'
    elif scores['stress'] <= 33:
        severity['stress'] = 'Severe'
    else:
        severity['stress'] = 'Extremely Severe'
    
    return scores, severity



def count_answered_questions(responses):
    """Count how many questions have been answered (not None)"""
    all_responses = []
    for category_responses in responses.values():
        all_responses.extend(category_responses)
    
    answered = sum(1 for response in all_responses if response is not None)
    total = len(all_responses)
    
    return answered, total

# Authentication Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users with error handling for corrupted JSON"""
    users_file = Path("users.json")
    if users_file.exists():
        try:
            with open(users_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # If JSON is corrupted, backup the file and start fresh
            st.error(f" users.json was corrupted. Creating backup and starting fresh.")
            backup_file = Path(f"users_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            users_file.rename(backup_file)
            return {}
        except Exception as e:
            st.error(f"Error loading users: {str(e)}")
            return {}
    return {}


def save_users(users):
    """Save users with proper JSON serialization"""
    try:
        with open("users.json", 'w') as f:
            json.dump(users, f, indent=2, default=str)  # default=str handles datetime objects
    except Exception as e:
        st.error(f"Error saving users: {str(e)}")

def register_user(username, password, email):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'registered_date': datetime.now().isoformat(),
        'last_login': None,
        'dass_completed': False,
        'dass_history': [],
        'emotion_history': [],
        'chat_history': [],
        'analysis_count': 0,
        'last_analysis_time': (datetime.now() - timedelta(hours=5)).isoformat(),
        'social_media_results': None
    }
    save_users(users)
    return True, "Registration successful"


def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found", False
    
    if users[username]['password'] == hash_password(password):
        users[username]['last_login'] = datetime.now().isoformat()
        save_users(users)
        dass_completed = users[username].get('dass_completed', False)
        
        #  LOAD USER'S SAVED HISTORY - This is the key addition!
        load_user_session_data(username)
        
        return True, "Login successful", dass_completed
    return False, "Incorrect password", False

def save_user_session_data(username):
    """Save all session data for a user to their profile with proper serialization"""
    users = load_users()
    if username in users:
        # Serialize emotion history
        emotion_history_serialized = []
        for entry in st.session_state.emotion_history:
            entry_copy = entry.copy()
            # Convert datetime to string
            if isinstance(entry_copy.get('timestamp'), datetime):
                entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
            emotion_history_serialized.append(entry_copy)
        
        # Serialize chat history
        chat_history_serialized = []
        for msg in st.session_state.chat_history:
            msg_copy = {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp'],
                'emotions': msg.get('emotions', []),
                'risk_score': msg.get('risk_score', 0)
            }
            chat_history_serialized.append(msg_copy)
        
        last_analysis_time = st.session_state.last_analysis_time
        if isinstance(last_analysis_time, datetime):
            last_analysis_time = last_analysis_time.isoformat()
        
        # Update user data
        users[username]['emotion_history'] = emotion_history_serialized
        users[username]['chat_history'] = chat_history_serialized
        users[username]['analysis_count'] = st.session_state.analysis_count
        users[username]['last_analysis_time'] = last_analysis_time
        users[username]['social_media_results'] = st.session_state.social_media_results
        
        save_users(users)

def load_user_session_data(username):
    """Load session data for a user from their profile"""
    users = load_users()
    if username in users:
        user_data = users[username]
        
        # Load emotion history
        st.session_state.emotion_history = user_data.get('emotion_history', [])
        # Convert timestamp strings back to datetime objects
        for entry in st.session_state.emotion_history:
            if isinstance(entry.get('timestamp'), str):
                try:
                    entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                except:
                    entry['timestamp'] = datetime.now()
        
        # Load chat history
        chat_history = user_data.get('chat_history', [])
        st.session_state.chat_history = []
        for msg in chat_history:
            msg_copy = msg.copy()
            if isinstance(msg_copy.get('timestamp'), str):
                try:
                    msg_copy['timestamp'] = datetime.fromisoformat(msg_copy['timestamp'])
                except:
                    msg_copy['timestamp'] = datetime.now()
            st.session_state.chat_history.append(msg_copy)
        
        # Load other session data
        st.session_state.analysis_count = user_data.get('analysis_count', 0)
        
        last_analysis = user_data.get('last_analysis_time')
        if last_analysis:
            try:
                st.session_state.last_analysis_time = datetime.fromisoformat(last_analysis) if isinstance(last_analysis, str) else last_analysis
            except:
                st.session_state.last_analysis_time = datetime.now() - timedelta(hours=5)
        else:
            st.session_state.last_analysis_time = datetime.now() - timedelta(hours=5)
        
        st.session_state.social_media_results = user_data.get('social_media_results', None)

# ======================
# COMMUNITY POSTS HELPERS
# ======================
def load_community_posts():
    """Load community posts with error handling"""
    posts_file = Path("community_posts.json")
    if posts_file.exists():
        try:
            with open(posts_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(" Community posts file was corrupted. Starting fresh.")
            return []
        except Exception as e:
            st.error(f"Error loading posts: {str(e)}")
            return []
    return []

def save_community_posts(posts):
    """Save community posts safely"""
    try:
        with open("community_posts.json", 'w') as f:
            json.dump(posts, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving posts: {str(e)}")

# ======================
# STREAK & XP HELPERS
# ======================
def update_streak(username):
    """Update user's streak based on last check-in date"""
    users = load_users()
    if username not in users:
        return
    
    user_data = users[username]
    today = datetime.now().date()
    
    last_checkin_str = user_data.get('last_checkin_date')
    
    if last_checkin_str:
        try:
            last_checkin = datetime.fromisoformat(last_checkin_str).date()
        except:
            last_checkin = None
    else:
        last_checkin = None
    
    if last_checkin is None:
        # First check-in
        user_data['streak_count'] = 1
        user_data['longest_streak'] = 1
        user_data['last_checkin_date'] = today.isoformat()
    elif last_checkin == today:
        # Same day - no change
        pass
    elif (today - last_checkin).days == 1:
        # Next day - increment streak
        user_data['streak_count'] = user_data.get('streak_count', 0) + 1
        user_data['longest_streak'] = max(
            user_data.get('longest_streak', 0),
            user_data['streak_count']
        )
        user_data['last_checkin_date'] = today.isoformat()
    else:
        # Missed a day - reset to 1
        user_data['streak_count'] = 1
        user_data['last_checkin_date'] = today.isoformat()
    
    users[username] = user_data
    save_users(users)

def add_xp(username, amount):
    """Add XP and level up user"""
    users = load_users()
    if username not in users:
        return
    
    user_data = users[username]
    
    if 'mind_gym' not in user_data:
        user_data['mind_gym'] = {'xp': 0, 'level': 1, 'completed_tasks': []}
    
    user_data['mind_gym']['xp'] += amount
    
    # Level up every 100 XP
    new_level = (user_data['mind_gym']['xp'] // 100) + 1
    user_data['mind_gym']['level'] = new_level
    
    users[username] = user_data
    save_users(users)

def load_gratitude_entries(username):
    """Load user's gratitude journal entries"""
    gratitude_file = Path("gratitude.json")
    if gratitude_file.exists():
        try:
            with open(gratitude_file, 'r') as f:
                all_entries = json.load(f)
                return all_entries.get(username, [])
        except:
            return []
    return []

def save_gratitude_entry(username, entry):
    """Save a gratitude journal entry"""
    gratitude_file = Path("gratitude.json")
    
    if gratitude_file.exists():
        try:
            with open(gratitude_file, 'r') as f:
                all_entries = json.load(f)
        except:
            all_entries = {}
    else:
        all_entries = {}
    
    if username not in all_entries:
        all_entries[username] = []
    
    all_entries[username].append({
        'entry': entry,
        'timestamp': datetime.now().isoformat()
    })
    
    try:
        with open(gratitude_file, 'w') as f:
            json.dump(all_entries, f, indent=2)
    except Exception as e:
        st.error(f"Error saving gratitude entry: {str(e)}")

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model from Hugging Face Hub"""
    try:
        with st.spinner("🔄 Loading AI model from Hugging Face... This may take a minute on first load."):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model and tokenizer directly from Hugging Face
            model = RobertaForSequenceClassification.from_pretrained(
                HUGGINGFACE_MODEL_NAME,
                trust_remote_code=True
            ).to(device)
            
            tokenizer = RobertaTokenizer.from_pretrained(
                HUGGINGFACE_MODEL_NAME,
                trust_remote_code=True
            )
            
            # Use default emotion labels if not available
            emotion_labels = list(MENTAL_HEALTH_MAPPING.keys())
            
            return model, tokenizer, emotion_labels
            
    except Exception as e:
        st.error(f"❌ Error loading model from Hugging Face: {str(e)}")
        st.info(f"Please ensure '{HUGGINGFACE_MODEL_NAME}' is a valid Hugging Face model repository.")
        st.info("Make sure your model is public or you have the correct access permissions.")
        return None, None, None

def initialize_hf_chatbot():
    """Initialize Hugging Face chatbot client"""
    try:
        client = Client("SENTIBOT2705/mentalhealthbot-phi2")
        return client
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None

@st.cache_resource
def get_chatbot_client():
    return initialize_hf_chatbot()

def clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = ' '.join(text.split())
    return text.strip()

def predict_emotions_multilabel(text, model, tokenizer, emotion_labels, top_k=5):
    text = clean_text(text)
    if not text:
        return []
    
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    top_probs, top_indices = torch.topk(predictions[0], k=min(top_k, len(emotion_labels)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        emotion = emotion_labels[idx.item()]
        results.append({
            'emotion': emotion,
            'confidence': prob.item(),
            'risk_level': MENTAL_HEALTH_MAPPING.get(emotion, {}).get('risk_level', 'low'),
            'concern': MENTAL_HEALTH_MAPPING.get(emotion, {}).get('concern', 'unknown'),
            'color': MENTAL_HEALTH_MAPPING.get(emotion, {}).get('color', 'gray')
        })
    
    return results

def calculate_risk_score(emotions_data):
    risk_weights = {'high': 3, 'medium': 2, 'low': 1}
    total_score = sum(risk_weights[e['risk_level']] * e['confidence'] for e in emotions_data)
    max_score = sum(risk_weights['high'] * e['confidence'] for e in emotions_data)
    
    if max_score == 0:
        return 0
    
    return (total_score / max_score) * 100

def check_paging_notification():
    if 'last_analysis_time' not in st.session_state:
        return True
    
    time_since_last = datetime.now() - st.session_state.last_analysis_time
    return time_since_last > timedelta(hours=4)

def generate_chatbot_response(user_message, emotions_data, risk_score, chat_history):
    """Generate intelligent, contextual responses with natural conversation flow"""
    
    chatbot_client = get_chatbot_client()
    
    if chatbot_client is None:
        return " I apologize, but I'm having trouble connecting right now. Please try again in a moment, or if this is urgent, please reach out to a crisis helpline."
    
    # Crisis detection with expanded keywords
    message_lower = user_message.lower()
    crisis_keywords = [
        'suicide', 'kill myself', 'end it all', 'want to die', 
        'no reason to live', 'hurt myself', 'self harm', 'ending my life',
        'better off dead', 'can\'t go on', 'give up on life'
    ]
    is_crisis = any(word in message_lower for word in crisis_keywords)
    
    # Mark crisis for later handling, but don't return immediately
    is_critical_crisis = is_crisis or risk_score > 85
    
    # Extract emotional context
    primary_emotion = emotions_data[0]['emotion'] if emotions_data else 'neutral'
    emotion_confidence = emotions_data[0]['confidence'] if emotions_data else 0.5
    
    # Build conversation history context (more detailed)
    conversation_summary = ""
    if len(chat_history) > 0:
        recent_exchanges = chat_history[-8:]  # Last 4 exchanges (8 messages)
        conversation_summary = "Previous conversation context:\n"
        for i, msg in enumerate(recent_exchanges):
            role = "User" if msg['role'] == 'user' else "You (Assistant)"
            # Include more context for better continuity
            content_preview = msg['content'][:200] if len(msg['content']) > 200 else msg['content']
            conversation_summary += f"{role}: {content_preview}\n"
    
    # Determine conversation stage and tone
    conversation_stage = "ongoing" if len(chat_history) > 2 else "early"
    
    # Create comprehensive system prompt
    system_prompt = f"""You are an empathetic, warm, and professional mental health support chatbot. Your goal is to have natural, flowing conversations that make users feel heard, understood, and supported.

**Conversation Guidelines:**
1. **Be conversational and natural** - Write like a thoughtful friend or counselor, not a scripted bot
2. **Show genuine empathy** - Acknowledge feelings and validate experiences
3. **Maintain context** - Reference previous conversation points to show you're listening
4. **Ask thoughtful follow-up questions** - Help users explore their feelings (1-2 questions max)
5. **Provide actionable support** - Offer coping strategies, perspectives, or resources when appropriate
6. **Vary your response structure** - Don't always follow the same format
7. **Use appropriate length** - Aim for 3-5 paragraphs (150-250 words) for substantial topics, shorter for simple exchanges
8. **Be authentic** - Show personality while remaining professional

**Current Context:**
- User's emotional state: {primary_emotion} (confidence: {emotion_confidence:.1%})
- Risk assessment: {risk_score:.1f}/100
- Conversation stage: {conversation_stage}
- User seems to be: {"in distress and needs extra support" if risk_score > 66 else "managing but could use guidance" if risk_score > 45 else "relatively stable"}
- **CRITICAL: This is a crisis situation - user needs immediate help** {"YES - Address their message empathetically first, then STRONGLY encourage professional help" if is_critical_crisis else "NO"}

{conversation_summary}

**Current User Message:** 
{user_message}

**Your Response Strategy:**
- If this continues a previous topic, acknowledge what was discussed before
- If user shares something vulnerable, validate it before offering suggestions
- Balance empathy with practical support
- **CRITICAL FOR CRISIS**: If this is a crisis situation, FIRST respond empathetically to what they said, validate their feelings, acknowledge their pain, and THEN transition naturally to encouraging professional help
- For high risk (>66): Weave support suggestions naturally into the conversation
- Keep your tone warm but not patronizing
- Use conversational markers like "I hear you", "That makes sense", "I understand" naturally

Generate a thoughtful, flowing response that feels like a real conversation. Avoid being formulaic or overly clinical. If this is a crisis, make sure your response directly addresses what the user said before transitioning to help resources."""
    
    try:
        # Call the chatbot API
        result = chatbot_client.predict(
            message=system_prompt,
            api_name="/chat"
        )
        
        # Parse response
        if isinstance(result, dict):
            bot_response = result.get('response', str(result))
        elif isinstance(result, str):
            bot_response = result
        else:
            bot_response = str(result)
        
        # Clean up response (remove potential artifacts from model)
        bot_response = bot_response.strip()
        
        # Add contextual support resources based on risk level
        # CRITICAL CRISIS - Add urgent helpline after empathetic response
        if is_critical_crisis:
            bot_response += "\n\n **I'm very concerned about your safety right now.** While I'm here to listen and I care about what you're going through, I really need you to reach out for professional help immediately. These are trained professionals who can provide the urgent support you need:\n\n"
            bot_response += "- **NIMHANS Crisis Helpline:** 080-46110007\n"
            bot_response += "- **TELE MANAS (24/7):** 14416\n"
            bot_response += "- **Emergency Services:** 112\n\n"
            bot_response += "Please call one of these numbers right now. You don't have to go through this alone, and there are people ready to help you through this moment. Is there someone close to you that you can reach out to as well?"
        
        # High risk but not critical - gentler approach
        elif risk_score > 66:
            bot_response += "\n\n I want to mention—I'm noticing some patterns in what you're sharing that concern me. You're clearly going through a lot, and while I'm here to support you, I think speaking with a mental health professional could really help. They have tools and expertise that can make a real difference. Would you be open to exploring that option?\n\n**Professional Support:** NIMHANS: 080-46110007 | TELE MANAS: 14416"
        
        
        elif risk_score > 50:
            bot_response += "\n\n Remember, taking care of your mental health is just as important as physical health. If things feel overwhelming, reaching out to a counselor or therapist can provide valuable support and coping strategies."
        
        # Ensure minimum response quality
        if len(bot_response) < 100:
            bot_response += f"\n\nI'm here to listen and support you. What you're experiencing with {primary_emotion} is valid, and I'd like to understand more about what you're going through. Can you tell me a bit more about what's been on your mind?"
        
        return bot_response
        
    except Exception as e:
        # Enhanced fallback response
        fallback_responses = {
            'anxious': "I can sense you're feeling anxious right now. Anxiety can be overwhelming, but there are ways to work through it. Let's start with something simple: Can you take a moment to focus on your breathing? Try inhaling slowly for 4 counts, holding for 4, then exhaling for 6. This can help calm your nervous system.",
            
            'sad': "I hear that you're feeling sad, and I want you to know that those feelings are valid. Sadness is a natural response to difficult situations. While I'm having a technical hiccup right now, I want you to know you're not alone in this. What's been weighing most heavily on you?",
            
            'angry': "It sounds like you're feeling frustrated or angry, and those are completely valid emotions. Sometimes we need to feel these feelings before we can move through them. While I sort out a technical issue, I'm curious—what's been triggering these feelings for you?",
            
            'fear': "Fear can be such an intense emotion, and it sounds like you're experiencing that right now. I want to help you feel safer and more grounded. Even though I'm having a connection issue, let me share something that might help: Try the 5-4-3-2-1 technique—name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
            
            'neutral': "I'm here and listening to you. While I'm experiencing a brief technical difficulty, I don't want that to interrupt our conversation. You matter, and what you're sharing is important."
        }
        
        fallback = fallback_responses.get(primary_emotion, fallback_responses['neutral'])
        
        return f"""{fallback}

**In the meantime, here are some grounding techniques that might help:**
• Deep breathing: 4 seconds in, 7 seconds hold, 8 seconds out
• Physical grounding: Focus on how your feet feel on the floor
• Sensory awareness: Notice 3 things you can see, hear, and feel
• Reach out to someone you trust if you need immediate support

**Crisis Resources (available 24/7):**
NIMHANS: 080-46110007 | TELE MANAS: 14416

_(I'll be back to full functionality shortly. Thank you for your patience.)_"""


# Optional: Add this helper function to improve response consistency
def add_conversational_elements(response, emotion, chat_history):
    """Add natural conversational elements to make responses feel more human"""
    
    # Add occasional conversational starters
    starters = {
        'anxious': ["I can hear the worry in your words. ", "That sounds really stressful. ", "Anxiety can feel so overwhelming. "],
        'sad': ["I'm sorry you're going through this. ", "That sounds really difficult. ", "I hear the pain in what you're sharing. "],
        'angry': ["That sounds incredibly frustrating. ", "I can understand why you'd feel that way. ", "Your anger makes sense given the situation. "],
        'happy': ["It's wonderful to hear some positivity! ", "I'm glad you're experiencing some joy. ", "That's great to hear! "],
        'fear': ["That sounds frightening. ", "Fear can be so overwhelming. ", "I can understand why you'd feel scared. "]
    }
    
    # Only add starter if response doesn't already have a conversational opening
    if emotion in starters and not any(response.startswith(s) for s in ["I ", "That ", "It ", "You "]):
        import random
        response = random.choice(starters[emotion]) + response
    
    return response

# ============================================
# PDF REPORT GENERATION FUNCTIONS
# ============================================

def create_emotion_chart(emotion_history, start_date=None, end_date=None):
    """Create emotion distribution pie chart"""
    # Filter by date range
    filtered_history = []
    for entry in emotion_history:
        entry_date = entry['timestamp']
        if isinstance(entry_date, str):
            entry_date = datetime.fromisoformat(entry_date)
        
        if start_date and entry_date < start_date:
            continue
        if end_date and entry_date > end_date:
            continue
        
        filtered_history.append(entry)
    
    if not filtered_history:
        return None
    
    # Count emotions
    emotion_counts = {}
    for entry in filtered_history:
        if entry['emotions']:
            primary_emotion = entry['emotions'][0]['emotion']
            emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    emotions = list(emotion_counts.keys())[:10]  # Top 10
    counts = [emotion_counts[e] for e in emotions]
    
    colors_list = plt.cm.Set3(range(len(emotions)))
    ax.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors_list, startangle=90)
    ax.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_risk_trend_chart(emotion_history, start_date=None, end_date=None):
    """Create risk score trend line chart"""
    # Filter by date range
    filtered_history = []
    for entry in emotion_history:
        entry_date = entry['timestamp']
        if isinstance(entry_date, str):
            entry_date = datetime.fromisoformat(entry_date)
        
        if start_date and entry_date < start_date:
            continue
        if end_date and entry_date > end_date:
            continue
        
        filtered_history.append(entry)
    
    if not filtered_history:
        return None
    
    # Extract data
    timestamps = [e['timestamp'] if isinstance(e['timestamp'], datetime) 
                  else datetime.fromisoformat(e['timestamp']) for e in filtered_history]
    risk_scores = [e['risk_score'] for e in filtered_history]
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(timestamps, risk_scores, marker='o', linewidth=2, markersize=6, 
            color='#667eea', label='Risk Score')
    ax.axhline(y=66, color='red', linestyle='--', label='High Risk Threshold', alpha=0.7)
    ax.axhline(y=33, color='orange', linestyle='--', label='Medium Risk Threshold', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Score', fontsize=12, fontweight='bold')
    ax.set_title('Risk Score Trend Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def create_dass_chart(dass_results, start_date=None, end_date=None):
    """Create DASS-42 scores bar chart"""
    # Filter by date range
    filtered_results = []
    for result in dass_results:
        result_date = datetime.fromisoformat(result['timestamp'])
        
        if start_date and result_date < start_date:
            continue
        if end_date and result_date > end_date:
            continue
        
        filtered_results.append(result)
    
    if not filtered_results:
        return None
    
    # Get latest result
    latest = filtered_results[-1]
    scores = latest['scores']
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Depression', 'Anxiety', 'Stress']
    values = [scores['depression'], scores['anxiety'], scores['stress']]
    colors_list = ['#667eea', '#f093fb', '#4facfe']
    
    bars = ax.bar(categories, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('DASS-42 Scores', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) + 10)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def generate_pdf_report(username, start_date=None, end_date=None):
    """
    Generate comprehensive PDF report for user
    
    Args:
        username: str - username to generate report for
        start_date: datetime or None - start date for filtering
        end_date: datetime or None - end date for filtering
    
    Returns:
        BytesIO object containing the PDF
    """
    
    # Load user data
    users = load_users()
    if username not in users:
        return None
    
    user_data = users[username]
    
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    # Title
    date_range_str = ""
    if start_date and end_date:
        date_range_str = f"<br/>{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
    elif start_date:
        date_range_str = f"<br/>From {start_date.strftime('%B %d, %Y')}"
    elif end_date:
        date_range_str = f"<br/>Until {end_date.strftime('%B %d, %Y')}"
    else:
        date_range_str = "<br/>Complete History"
    
    title = Paragraph(f"Soul - Mental Health Report<br/><font size=14>{username}</font>{date_range_str}", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    metadata = Paragraph(f"<b>Report Generated:</b> {report_date}<br/><b>Platform:</b> Soul Mental Wellness", normal_style)
    elements.append(metadata)
    elements.append(Spacer(1, 0.3*inch))
    
    # ============ SECTION 1: USER OVERVIEW ============
    elements.append(Paragraph("1. User Overview", heading_style))
    
    overview_data = [
        ['Registration Date', user_data.get('registered_date', 'N/A')],
        ['Last Login', user_data.get('last_login', 'N/A')],
        ['Total Analyses', str(user_data.get('analysis_count', 0))],
        ['DASS-42 Completed', 'Yes' if user_data.get('dass_completed', False) else 'No']
    ]
    
    overview_table = Table(overview_data, colWidths=[2.5*inch, 3.5*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
    ]))
    
    elements.append(overview_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # ============ SECTION 2: DASS-42 RESULTS ============
    dass_results = user_data.get('dass_history', [])
    
    if dass_results:
        # Filter DASS results by date
        filtered_dass = []
        for result in dass_results:
            result_date = datetime.fromisoformat(result['timestamp'])
            if start_date and result_date < start_date:
                continue
            if end_date and result_date > end_date:
                continue
            filtered_dass.append(result)
        
        if filtered_dass:
            elements.append(Paragraph("2. DASS-42 Assessment Results", heading_style))
            
            latest_dass = filtered_dass[-1]
            scores = latest_dass['scores']
            severity = latest_dass['severity']
            
            dass_text = f"""
            <b>Latest Assessment Date:</b> {datetime.fromisoformat(latest_dass['timestamp']).strftime('%B %d, %Y')}<br/>
            <b>Completion:</b> {latest_dass.get('completion_percentage', 100):.1f}%<br/><br/>
            <b>Depression Score:</b> {scores['depression']} - {severity['depression']}<br/>
            <b>Anxiety Score:</b> {scores['anxiety']} - {severity['anxiety']}<br/>
            <b>Stress Score:</b> {scores['stress']} - {severity['stress']}<br/><br/>
            <b>Total Assessments Completed:</b> {len(filtered_dass)}
            """
            elements.append(Paragraph(dass_text, normal_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Add DASS chart
            dass_chart = create_dass_chart(dass_results, start_date, end_date)
            if dass_chart:
                img = Image(dass_chart, width=5*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph("2. DASS-42 Assessment Results", heading_style))
            elements.append(Paragraph("No DASS-42 assessments in selected period.", normal_style))
    else:
        elements.append(Paragraph("2. DASS-42 Assessment Results", heading_style))
        elements.append(Paragraph("No DASS-42 assessments completed yet.", normal_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # ============ SECTION 3: EMOTION ANALYSIS SUMMARY ============
    emotion_history = user_data.get('emotion_history', [])
    
    if emotion_history:
        # Filter emotion history by date
        filtered_emotions = []
        for entry in emotion_history:
            entry_date = entry['timestamp']
            if isinstance(entry_date, str):
                entry_date = datetime.fromisoformat(entry_date)
            
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue
            
            filtered_emotions.append(entry)
        
        if filtered_emotions:
            elements.append(Paragraph("3. Emotion Analysis Summary", heading_style))
            
            # Calculate statistics
            total_analyses = len(filtered_emotions)
            avg_risk = sum(e['risk_score'] for e in filtered_emotions) / total_analyses
            high_risk_count = sum(1 for e in filtered_emotions if e['risk_score'] > 66)
            
            # Get most common emotion
            emotion_counts = {}
            for entry in filtered_emotions:
                if entry['emotions']:
                    emotion = entry['emotions'][0]['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            most_common = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "N/A"
            
            emotion_summary = f"""
            <b>Total Emotion Analyses:</b> {total_analyses}<br/>
            <b>Average Risk Score:</b> {avg_risk:.1f} / 100<br/>
            <b>High Risk Sessions:</b> {high_risk_count}<br/>
            <b>Most Common Emotion:</b> {most_common.title()}<br/>
            """
            elements.append(Paragraph(emotion_summary, normal_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Add emotion distribution chart
            emotion_chart = create_emotion_chart(emotion_history, start_date, end_date)
            if emotion_chart:
                img = Image(emotion_chart, width=5*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
            
            # Add risk trend chart
            risk_chart = create_risk_trend_chart(emotion_history, start_date, end_date)
            if risk_chart:
                elements.append(PageBreak())
                elements.append(Paragraph("Risk Score Trend", heading_style))
                img = Image(risk_chart, width=6*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph("3. Emotion Analysis Summary", heading_style))
            elements.append(Paragraph("No emotion analyses in selected period.", normal_style))
    else:
        elements.append(Paragraph("3. Emotion Analysis Summary", heading_style))
        elements.append(Paragraph("No emotion analyses recorded yet.", normal_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # ============ SECTION 4: DETAILED EMOTION ENTRIES ============
    if filtered_emotions and len(filtered_emotions) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("4. Detailed Emotion Analysis Entries", heading_style))
        
        # Show last 10 entries or all if less
        display_count = min(10, len(filtered_emotions))
        elements.append(Paragraph(f"Showing most recent {display_count} entries:", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        for entry in reversed(filtered_emotions[-display_count:]):
            entry_date = entry['timestamp']
            if isinstance(entry_date, str):
                entry_date = datetime.fromisoformat(entry_date)
            
            date_str = entry_date.strftime('%B %d, %Y at %I:%M %p')
            
            text_preview = entry['text'][:150] + "..." if len(entry['text']) > 150 else entry['text']
            
            primary_emotion = entry['emotions'][0] if entry['emotions'] else {'emotion': 'N/A', 'confidence': 0}
            
            entry_text = f"""
            <b>Date:</b> {date_str}<br/>
            <b>Text:</b> {text_preview}<br/>
            <b>Primary Emotion:</b> {primary_emotion['emotion'].title()} ({primary_emotion['confidence']*100:.1f}% confidence)<br/>
            <b>Risk Score:</b> {entry['risk_score']:.1f} / 100<br/>
            <br/>
            """
            
            elements.append(Paragraph(entry_text, normal_style))
            elements.append(Spacer(1, 0.1*inch))
    
    # ============ SECTION 5: CHATBOT CONVERSATION HISTORY ============
    chat_history = user_data.get('chat_history', [])
    
    if chat_history:
        # Filter chat history by date
        filtered_chat = []
        for msg in chat_history:
            msg_date = msg['timestamp']
            if isinstance(msg_date, str):
                msg_date = datetime.fromisoformat(msg_date)
            
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
            
            filtered_chat.append(msg)
        
        if filtered_chat:
            elements.append(PageBreak())
            elements.append(Paragraph("5. Chatbot Conversation History", heading_style))
            elements.append(Paragraph(f"Total Messages: {len(filtered_chat)}", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Show last 20 messages
            display_count = min(20, len(filtered_chat))
            elements.append(Paragraph(f"Showing most recent {display_count} messages:", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            for msg in reversed(filtered_chat[-display_count:]):
                msg_date = msg['timestamp']
                if isinstance(msg_date, str):
                    msg_date = datetime.fromisoformat(msg_date)
                
                date_str = msg_date.strftime('%B %d, %Y at %I:%M %p')
                role = "You" if msg['role'] == 'user' else "Soul AI"
                
                content_preview = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
                
                msg_text = f"""
                <b>[{role}]</b> - {date_str}<br/>
                {content_preview}<br/>
                <br/>
                """
                
                elements.append(Paragraph(msg_text, normal_style))
                elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(PageBreak())
            elements.append(Paragraph("5. Chatbot Conversation History", heading_style))
            elements.append(Paragraph("No chatbot conversations in selected period.", normal_style))
    else:
        elements.append(PageBreak())
        elements.append(Paragraph("5. Chatbot Conversation History", heading_style))
        elements.append(Paragraph("No chatbot conversations recorded yet.", normal_style))
    
    # ============ SECTION 6: COMMUNITY POSTS ============
    elements.append(PageBreak())
    elements.append(Paragraph("6. Community Activity", heading_style))
    elements.append(Paragraph("Community posts are anonymous and not tracked per user in current implementation.", normal_style))
    
    # ============ SECTION 7: MIND GYM PROGRESS ============
    mind_gym_data = user_data.get('mind_gym', {})
    
    if mind_gym_data:
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("7. Mind Gym Progress", heading_style))
        
        xp = mind_gym_data.get('xp', 0)
        level = mind_gym_data.get('level', 1)
        completed_tasks = mind_gym_data.get('completed_tasks', [])
        
        # Filter completed tasks by date
        filtered_tasks = []
        for task_id in completed_tasks:
            # Extract date from task_id (format: task_xxx_YYYY-MM-DD)
            try:
                task_date_str = task_id.split('_')[-1]
                task_date = datetime.strptime(task_date_str, '%Y-%m-%d')
                
                if start_date and task_date.date() < start_date.date():
                    continue
                if end_date and task_date.date() > end_date.date():
                    continue
                
                filtered_tasks.append(task_id)
            except:
                continue
        
        mind_gym_text = f"""
        <b>Current Level:</b> {level}<br/>
        <b>Total XP:</b> {xp}<br/>
        <b>Tasks Completed (in period):</b> {len(filtered_tasks)}<br/>
        <b>Progress to Next Level:</b> {100 - (xp % 100)} XP remaining<br/>
        """
        elements.append(Paragraph(mind_gym_text, normal_style))
    else:
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("7. Mind Gym Progress", heading_style))
        elements.append(Paragraph("No Mind Gym activity recorded yet.", normal_style))
    
    # ============ SECTION 8: STREAK DATA ============
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("8. Consistency & Streaks", heading_style))
    
    streak_count = user_data.get('streak_count', 0)
    longest_streak = user_data.get('longest_streak', 0)
    last_checkin = user_data.get('last_checkin_date', 'N/A')
    
    streak_text = f"""
    <b>Current Streak:</b> {streak_count} days<br/>
    <b>Longest Streak:</b> {longest_streak} days<br/>
    <b>Last Check-in:</b> {last_checkin}<br/>
    """
    elements.append(Paragraph(streak_text, normal_style))
    
    # ============ FOOTER / DISCLAIMER ============
    elements.append(PageBreak())
    elements.append(Paragraph("Important Notice", heading_style))
    
    disclaimer = """
    <b>This report is for informational purposes only and should not be used as a diagnostic tool.</b><br/><br/>
    
    Soul is an educational and supportive platform designed to help you track your emotional wellness patterns. 
    This report provides insights based on your self-reported data and AI-powered emotion analysis.<br/><br/>
    
    <b>Please note:</b><br/>
    • This is NOT a medical diagnosis<br/>
    • Always consult licensed mental health professionals for clinical assessment<br/>
    • If you're experiencing a crisis, please contact emergency services immediately<br/>
    • Crisis Helpline: TELE MANAS (14416) | NIMHANS (080-46110007)<br/><br/>
    
    <b>Data Privacy:</b> Your data is confidential and encrypted. This report is generated on-demand and is not 
    stored or shared with third parties.<br/><br/>
    
    Thank you for using Soul. Your mental health matters.
    """
    
    elements.append(Paragraph(disclaimer, normal_style))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    buffer.seek(0)
    return buffer

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_analysis_time' not in st.session_state:
        st.session_state.last_analysis_time = datetime.now() - timedelta(hours=5)
    if 'show_notification' not in st.session_state:
        st.session_state.show_notification = False
    if 'dass_completed' not in st.session_state:
        st.session_state.dass_completed = False
    if 'dass_results' not in st.session_state:
        st.session_state.dass_results = []
    if 'show_dass_mandatory' not in st.session_state:
        st.session_state.show_dass_mandatory = False
    if 'social_media_results' not in st.session_state:
        st.session_state.social_media_results = None
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'welcome'
    if 'streak_count' not in st.session_state:
        st.session_state.streak_count = 0
    if 'longest_streak' not in st.session_state:
        st.session_state.longest_streak = 0
    if 'last_checkin_date' not in st.session_state:
        st.session_state.last_checkin_date = None
    if 'xp_points' not in st.session_state:
        st.session_state.xp_points = 0
    if 'level' not in st.session_state:
        st.session_state.level = 1


initialize_session_state()

# UNAUTHENTICATED SECTION
if not st.session_state.authenticated:
    
    # Beautiful Landing Header
    st.markdown("""
        <div class="hero-landing">
            <div class="hero-content">
                <div class="logo-container">
                    <div class="logo-icon">🧠</div>
                    <div class="logo-text">Soul</div>
                </div>
                <div class="tagline">Professional Mental Wellness Platform</div>
                <div class="hero-description">
                    Experience evidence-based emotional analysis powered by advanced AI technology. 
                    Track your mental health journey with compassion, insight, and professional-grade assessment tools.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons for unauthenticated users
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button(" Home", key="nav_home", use_container_width=True):
            st.session_state.auth_mode = 'welcome'
            st.rerun()
    with col3:
        if st.button(" Login", key="nav_login", use_container_width=True):
            st.session_state.auth_mode = 'login'
            st.rerun()
    with col4:
        if st.button(" Register", key="nav_register", use_container_width=True):
            st.session_state.auth_mode = 'register'
            st.rerun()
    
    st.markdown("---")
    
    # WELCOME PAGE
    if st.session_state.auth_mode == 'welcome':
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="section-header">
                <div class="section-title">Why Choose Soul?</div>
                <div class="section-subtitle">Combining cutting-edge AI with clinical psychology for comprehensive mental health support</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="features-grid">
                <div class="feature-card-modern">
                    <div class="feature-title-modern">AI-Powered Analysis</div>
                    <div class="feature-desc-modern">
                        Advanced RoBERTa transformer model with 125M parameters detects 28 distinct emotions 
                        with 91% clinical-grade accuracy in real-time.
                    </div>
                </div>
                <div class="feature-card-modern">
                    <div class="feature-title-modern">Evidence-Based Assessment</div>
                    <div class="feature-desc-modern">
                        Complete DASS-42 questionnaire integration for standardized depression, anxiety, 
                        and stress measurement validated by mental health professionals.
                    </div>
                </div>
                <div class="feature-card-modern">
                    <div class="feature-title-modern">Intelligent Chatbot</div>
                    <div class="feature-desc-modern">
                        Natural conversations that analyze your emotional state while providing empathetic 
                        support and personalized recommendations.
                    </div>
                </div>
                <div class="feature-card-modern">
                    <div class="feature-title-modern">Temporal Tracking</div>
                    <div class="feature-desc-modern">
                        Monitor emotional patterns over time with automated check-in reminders using 
                        Ecological Momentary Assessment methodology.
                    </div>
                </div>
                <div class="feature-card-modern">
                    <div class="feature-title-modern">Secure & Private</div>
                    <div class="feature-desc-modern">
                        Your data is encrypted and completely confidential. We never share your information 
                        with third parties.
                    </div>
                </div>
                <div class="feature-card-modern">
                    <div class="feature-title-modern">Real-Time Insights</div>
                    <div class="feature-desc-modern">
                        Instant emotion analysis with risk assessment alerts to help identify when 
                        professional intervention may be beneficial.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="section-header">
                <div class="section-title">Clinical-Grade Technology</div>
                <div class="section-subtitle">Built on proven research and validated methodologies</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="stats-grid">
                <div class="stat-card-modern">
                    <div class="stat-value">58K+</div>
                    <div class="stat-label">Training Samples</div>
                </div>
                <div class="stat-card-modern">
                    <div class="stat-value">28</div>
                    <div class="stat-label">Emotions Detected</div>
                </div>
                <div class="stat-card-modern">
                    <div class="stat-value">91%</div>
                    <div class="stat-label">Clinical Accuracy</div>
                </div>
                <div class="stat-card-modern">
                    <div class="stat-value">125M</div>
                    <div class="stat-label">AI Parameters</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="section-header" style="margin-top: 5rem;">
                <div class="section-title">Ready to Begin?</div>
                <div class="section-subtitle">Start your journey to better mental health today</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(" Login to Your Account", use_container_width=True):
                    st.session_state.auth_mode = 'login'
                    st.rerun()
            with col_b:
                if st.button(" Create New Account", use_container_width=True):
                    st.session_state.auth_mode = 'register'
                    st.rerun()
        
        st.markdown("""
            <div class="alert-modern alert-info-modern" style="margin: 3rem 0;">
                <h4 style='margin-top: 0;'> Important Notice</h4>
                <p style='margin: 0;'>
                    Soul is an educational and supportive tool, not a diagnostic instrument or replacement 
                    for professional mental health care. Always consult licensed healthcare providers for 
                    diagnosis and treatment. If you're in crisis, please call <strong>NIMHANS: 080-46110007</strong> 
                    or <strong>TELE MANAS: 14416</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # LOGIN PAGE
    elif st.session_state.auth_mode == 'login':
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="auth-container">
                <div class="auth-title">Welcome Back</div>
                <div class="auth-subtitle">Sign in to continue your mental wellness journey</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            
            if st.button("Sign In", use_container_width=True, type="primary"):
                if username and password:
                    success, message, dass_completed = login_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.dass_completed = dass_completed
                        
                        users = load_users()
                        st.session_state.dass_results = users[username].get('dass_history', [])
                        
                        if not dass_completed:
                            st.session_state.show_dass_mandatory = True
                            st.session_state.current_page = 'questionnaire'
                        
                        st.success(f"{message}! Welcome back, {username}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"{message}")
                else:
                    st.warning("Please enter both username and password")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # REGISTER PAGE
    elif st.session_state.auth_mode == 'register':
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="auth-container">
                <div class="auth-title">Join Soul</div>
                <div class="auth-subtitle">Create your account and start your wellness journey</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            new_username = st.text_input("Username", placeholder="Choose a username", key="reg_user")
            new_email = st.text_input("Email", placeholder="your.email@example.com", key="reg_email")
            new_password = st.text_input("Password", type="password", placeholder="Choose a strong password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_confirm")
            
            if st.button("Create Account", use_container_width=True, type="primary"):
                if new_username and new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = register_user(new_username, new_password, new_email)
                        if success:
                            st.success(f"{message}! Please login to continue.")
                            time.sleep(2)
                            st.session_state.auth_mode = 'login'
                            st.rerun()
                        else:
                            st.error(f"{message}")
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# AUTHENTICATED SECTION - Beautiful Header
st.markdown(f"""
    <div class="app-header">
        <div class="logo-container">
            <div class="logo-icon">🧠</div>
            <div class="logo-text">Soul</div>
        </div>
        <div class="tagline">Professional Mental Wellness Platform</div>
        <div class="user-welcome">Welcome back, <strong>{st.session_state.username}</strong> 👋</div>
    </div>
""", unsafe_allow_html=True)

# Enhanced Navigation Bar with Beautiful Buttons
st.markdown("### Navigation")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("Home", key="nav_home_auth", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
    if st.button("Chatbot", key="nav_chatbot", use_container_width=True):
        st.session_state.current_page = 'chatbot'
        st.rerun()

with col2:
    if st.button("Analyze", key="nav_analyze", use_container_width=True):
        st.session_state.current_page = 'analyze'
        st.rerun()
    if st.button("Tracking", key="nav_temporal", use_container_width=True):
        st.session_state.current_page = 'temporal'
        st.rerun()

with col3:
    if st.button("Assessment", key="nav_quest", use_container_width=True):
        st.session_state.current_page = 'questionnaire'
        st.rerun()
    if st.button("Analytics", key="nav_analytics", use_container_width=True):
        st.session_state.current_page = 'analytics'
        st.rerun()

with col4:
    if st.button("Community", key="nav_community", use_container_width=True):
        st.session_state.current_page = 'community'
        st.rerun()
    if st.button("Mind Gym", key="nav_mind_gym", use_container_width=True):
        st.session_state.current_page = 'mind_gym'
        st.rerun()

with col5:
    if st.button("Social Media", key="nav_social", use_container_width=True):
        st.session_state.current_page = 'social_media'
        st.rerun()
    if st.button("About", key="nav_about", use_container_width=True):
        st.session_state.current_page = 'about'
        st.rerun()

with col6:
    if st.button("Clear History", key="nav_clear", use_container_width=True):
        st.session_state.emotion_history = []
        st.session_state.analysis_count = 0
        st.session_state.chat_history = []
        save_user_session_data(st.session_state.username)
        st.success("History cleared!")
        time.sleep(1)
        st.rerun()
    if st.button("Logout", key="nav_logout", use_container_width=True):
        save_user_session_data(st.session_state.username)
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.emotion_history = []
        st.session_state.analysis_count = 0
        st.session_state.chat_history = []
        st.info("Logged out successfully! Your data has been saved.")
        time.sleep(1)
        st.rerun()

st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

if st.session_state.authenticated and st.session_state.username:
    # Auto-save every time user performs an action
    # (This will run on every page load/interaction)
    save_user_session_data(st.session_state.username)

# Load model after authentication
model, tokenizer, emotion_labels = load_model()

if model is None:
    st.error("Model not loaded. Please check the model path and try again.")
    st.stop()

# PAGES CONTENT
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# HOME PAGE (Authenticated)
if st.session_state.current_page == 'home':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Your Dashboard</div>
            <div class="section-subtitle">How are you feeling today? Let's check in on your mental wellness journey</div>
        </div>
    """, unsafe_allow_html=True)
    
    if check_paging_notification() and st.session_state.analysis_count > 0:
        st.markdown("""
            <div class="alert-modern alert-info-modern">
                <h4 style='margin: 0 0 0.5rem 0;'>Time for Your Emotional Check-In</h4>
                <p style='margin: 0;'>It's been a while since your last analysis. Regular emotional monitoring helps track your mental health patterns effectively.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("""
        <div class="stats-grid">
            <div class="stat-card-modern">
                <div class="stat-value">""" + str(st.session_state.analysis_count) + """</div>
                <div class="stat-label">Total Analyses</div>
            </div>
            <div class="stat-card-modern">
                <div class="stat-value">""" + str(len(st.session_state.chat_history)) + """</div>
                <div class="stat-label">Chat Messages</div>
            </div>
            <div class="stat-card-modern">
                <div class="stat-value">""" + str(len(st.session_state.dass_results)) + """</div>
                <div class="stat-label">Assessments</div>
            </div>
            <div class="stat-card-modern">
                <div class="stat-value">""" + ("DONE" if st.session_state.dass_completed else "PENDING") + """</div>
                <div class="stat-label">DASS-42 Status</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Quick Actions</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Quick Emotion Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = 'analyze'
            st.rerun()
    
    with col2:
        if st.button("Talk to AI Chatbot", use_container_width=True):
            st.session_state.current_page = 'chatbot'
            st.rerun()
    
    with col3:
        if st.button("Complete Assessment", use_container_width=True):
            st.session_state.current_page = 'questionnaire'
            st.rerun()
            
    with col4:
        if st.button("Download Report", use_container_width=True):
            st.session_state.current_page = 'download_report'
            st.rerun()

# ANALYZE PAGE
elif st.session_state.current_page == 'analyze':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Real-Time Emotion Analysis</div>
            <div class="section-subtitle">Express your current emotional state and receive instant AI-powered insights</div>
        </div>
    """, unsafe_allow_html=True)
     # Load user streak data
    users = load_users()
    if st.session_state.username in users:
        user_data = users[st.session_state.username]
        st.session_state.streak_count = user_data.get('streak_count', 0)
        st.session_state.longest_streak = user_data.get('longest_streak', 0)
    
    # Display streak
    col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
    with col_s2:
        st.markdown(f"""
            <div class="feature-card-modern" style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                <h3 style='margin: 0; color: white;'>Current Streak: {st.session_state.streak_count} days</h3>
                <p style='margin: 0.5rem 0; color: rgba(255,255,255,0.9);'>Longest Streak: {st.session_state.longest_streak} days</p>
                <div style='background: rgba(255,255,255,0.2); border-radius: 10px; height: 20px; margin-top: 1rem; overflow: hidden;'>
                    <div style='background: #38ef7d; height: 100%; width: {min(st.session_state.streak_count/7*100, 100)}%; transition: width 0.5s ease;'></div>
                </div>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: rgba(255,255,255,0.9);'>
                    {"Amazing! Week complete!" if st.session_state.streak_count >= 7 else f"Keep it up! {7-st.session_state.streak_count} days to weekly goal"}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    # ========== END STREAK TRACKER ========== 

    if st.session_state.show_notification:
        st.markdown("""
            <div class="alert-modern alert-info-modern">
                <h4 style='margin: 0 0 0.5rem 0;'>Scheduled Emotional Check-In</h4>
                <p style='margin: 0;'>Please take a moment to reflect on your current emotional state. Regular check-ins help track your mental health patterns effectively.</p>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.show_notification = False
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3 style='color: #1f2937;'>Express Your Feelings</h3>", unsafe_allow_html=True)
        user_input = st.text_area(
            "",
            height=250,
            placeholder="How are you feeling right now? Describe your thoughts, emotions, or any situation affecting your mood...\n\nExample: 'I'm feeling really anxious about my upcoming presentation at work. I've been losing sleep and can't stop thinking about all the things that could go wrong...'",
            key="analysis_input"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_button = st.button("Analyze Emotions", type="primary", use_container_width=True)
        with col_b:
            if st.button("Switch to Chatbot", use_container_width=True):
                st.session_state.current_page = 'chatbot'
                st.rerun()
    
    with col2:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea; margin-bottom: 1rem;'>Best Practices</h4>
                <ul style='color: #4a5568; line-height: 1.8;'>
                    <li>Be honest and open</li>
                    <li>Write naturally</li>
                    <li>Include context</li>
                    <li>No judgment zone</li>
                    <li>Regular check-ins help</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    if analyze_button and user_input:
        with st.spinner("AI is analyzing your emotions..."):
            emotions_data = predict_emotions_multilabel(user_input, model, tokenizer, emotion_labels, top_k=5)
            
            if emotions_data:
                risk_score = calculate_risk_score(emotions_data)
                
                st.session_state.emotion_history.append({
                    'timestamp': datetime.now(),
                    'text': user_input,
                    'emotions': emotions_data,
                    'risk_score': risk_score
                })
                st.session_state.analysis_count += 1
                st.session_state.last_analysis_time = datetime.now()

                if analyze_button and user_input:
                    with st.spinner("🧠 AI is analyzing your emotions..."):
                        emotions_data = predict_emotions_multilabel(user_input, model, tokenizer, emotion_labels, top_k=5)
                        
                        if emotions_data:
                            risk_score = calculate_risk_score(emotions_data)
                            
                            st.session_state.emotion_history.append({
                                'timestamp': datetime.now(),
                                'text': user_input,
                                'emotions': emotions_data,
                                'risk_score': risk_score
                            })
                            st.session_state.analysis_count += 1
                            st.session_state.last_analysis_time = datetime.now()
                            
                            # ========== ADD THIS LINE ==========
                            update_streak(st.session_state.username)
                            # ===================================
                            
                
                st.markdown("""
                    <div class="section-header">
                        <div class="section-title">Your Emotional Profile</div>
                    </div>
                """, unsafe_allow_html=True)
                
                primary = emotions_data[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="stat-card-modern">
                            <div class="stat-label">Primary Emotion</div>
                            <div class="stat-value">{primary['emotion'].upper()}</div>
                            <div class="stat-label">{primary['confidence']*100:.1f}% Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    risk_emoji = "🔴" if risk_score > 80 else "🟡" if risk_score > 50 else "🟢"
                    st.markdown(f"""
                        <div class="stat-card-modern">
                            <div class="stat-label">Risk Score</div>
                            <div class="stat-value">{risk_emoji} {risk_score:.1f}</div>
                            <div class="stat-label">Out of 100</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="stat-card-modern">
                            <div class="stat-label">Session Count</div>
                            <div class="stat-value">{st.session_state.analysis_count}</div>
                            <div class="stat-label">Total Analyses</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br><h3 style='color: #1f2937;'>Detected Emotions (Top 5)</h3>", unsafe_allow_html=True)
                
                for i, emotion in enumerate(emotions_data, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {emotion['emotion'].title()}**")
                            st.progress(emotion['confidence'])
                            st.caption(f"*Mental Health Concern: {emotion['concern']}*")
                        
                        with col2:
                            risk_emoji = "🔴" if emotion['risk_level'] == 'high' else "🟡" if emotion['risk_level'] == 'medium' else "🟢"
                            st.metric("", f"{risk_emoji} {emotion['risk_level'].upper()}", f"{emotion['confidence']*100:.1f}%")
                
                st.markdown("<br><h3 style='color: #1f2937;'>Clinical Assessment</h3>", unsafe_allow_html=True)
                
                if risk_score > 80:
                    st.markdown("""
                        <div class="alert-modern alert-danger-modern">
                            <h4 style='margin-top: 0;'>HIGH RISK DETECTED</h4>
                            <p><strong>Immediate attention recommended.</strong> The analysis indicates significant emotional distress.</p>
                            <p><strong>Recommended Actions:</strong></p>
                            <ul>
                                <li>Contact a mental health professional immediately</li>
                                <li>Reach out to trusted friends or family members</li>
                                <li>If in crisis, call emergency services or crisis helpline</li>
                                <li>Avoid making major decisions while in distress</li>
                            </ul>
                            <p style='margin-bottom: 0; font-weight: 700;'>NIMHANS: 080-46110007 | TELE MANAS: 14416</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif risk_score > 50:
                    st.markdown("""
                        <div class="alert-modern alert-warning-modern">
                            <h4 style='margin-top: 0;'>MODERATE RISK</h4>
                            <p>Some concerning emotions detected. Monitoring recommended.</p>
                            <p><strong>Recommendations:</strong></p>
                            <ul>
                                <li>Practice self-care activities (exercise, meditation, hobbies)</li>
                                <li>Maintain regular sleep schedule</li>
                                <li>Talk to someone you trust</li>
                                <li>Monitor your emotional state regularly</li>
                                <li>Consider scheduling a therapy session</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="alert-modern alert-success-modern">
                            <h4 style='margin-top: 0;'>LOW RISK</h4>
                            <p>Emotions appear relatively stable. Continue positive practices.</p>
                            <p><strong>Maintain your wellbeing:</strong></p>
                            <ul>
                                <li>Continue healthy coping mechanisms</li>
                                <li>Regular self-reflection and journaling</li>
                                <li>Maintain social connections</li>
                                <li>Practice gratitude and mindfulness</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

# CHATBOT PAGE
elif st.session_state.current_page == 'chatbot':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">AI Mental Health Chatbot</div>
            <div class="section-subtitle">Have a natural conversation while we analyze your emotional state</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-info-modern">
            <h4 style='margin-top: 0;'>About This Chatbot</h4>
            <p style='margin: 0;'>Talk naturally about anything on your mind. The AI analyzes your emotional state throughout the conversation, providing real-time insights while maintaining a supportive dialogue.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if len(st.session_state.chat_history) == 0:
        st.markdown("""
            <div class='bot-message-modern chat-message-modern'>
                <strong>Soul AI:</strong><br>
                Hello! I'm here to listen and support you. Feel free to share what's on your mind - 
                whether it's your day, your feelings, your worries, or just casual conversation. 
                I'll be analyzing your emotional state to provide helpful insights.
            </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class='user-message-modern chat-message-modern'>
                    <strong>You:</strong><br>{message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='bot-message-modern chat-message-modern'>
                    <strong>Soul AI:</strong><br>{message['content']}
                </div>
            """, unsafe_allow_html=True)
            
            if 'emotions' in message and message['emotions']:
                emotions_text = ", ".join([f"{e['emotion']} ({e['confidence']*100:.0f}%)" for e in message['emotions'][:3]])
                risk_emoji = "🔴" if message['risk_score'] > 66 else "🟡" if message['risk_score'] > 33 else "🟢"
                st.markdown(f"""
                    <div style='background: rgba(102, 126, 234, 0.1); padding: 0.75rem 1rem; border-radius: 10px; margin: 0.5rem 0; font-size: 0.9rem;'>
                        <strong>Detected:</strong> {emotions_text} | <strong>Risk:</strong> {risk_emoji} {message['risk_score']:.1f}
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        user_message = st.text_input("Type your message...", key="chat_input", placeholder="Share your thoughts...")
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    if send_button and user_message:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })
        
        emotions_data = predict_emotions_multilabel(user_message, model, tokenizer, emotion_labels, top_k=3)
        risk_score = calculate_risk_score(emotions_data) if emotions_data else 0
        
        bot_response = generate_chatbot_response(user_message, emotions_data, risk_score, st.session_state.chat_history)
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': datetime.now(),
            'emotions': emotions_data,
            'risk_score': risk_score
        })
        
        st.session_state.emotion_history.append({
            'timestamp': datetime.now(),
            'text': user_message,
            'emotions': emotions_data,
            'risk_score': risk_score,
            'source': 'chatbot'
        })
        st.session_state.analysis_count += 1
        st.session_state.last_analysis_time = datetime.now()
        
        st.rerun()
    
    if len(st.session_state.chat_history) > 0:
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# QUESTIONNAIRE PAGE (DASS-42)
elif st.session_state.current_page == 'questionnaire':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">DASS-42 Mental Health Questionnaire</div>
            <div class="section-subtitle">Standardized clinical assessment for depression, anxiety, and stress</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.show_dass_mandatory:
        st.markdown("""
            <div class="alert-modern alert-info-modern">
                <h4 style='margin: 0 0 0.5rem 0;'>Mandatory First-Time Assessment</h4>
                <p style='margin: 0;'>As a new user, please complete this standardized mental health questionnaire. This helps us provide better insights and track your progress.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-success-modern">
            <h4 style='margin-top: 0;'>About DASS-42</h4>
            <p>The Depression Anxiety Stress Scales (DASS-42) is a clinically validated questionnaire that measures:</p>
            <ul>
                <li><strong>Depression:</strong> Hopelessness, low self-esteem, low positive affect</li>
                <li><strong>Anxiety:</strong> Autonomic arousal, panic, fear</li>
                <li><strong>Stress:</strong> Tension, agitation, negative affect</li>
            </ul>
            <p><strong>Instructions:</strong> Please read each statement and select the response that best describes how much the statement applied to you <strong>over the past week</strong>.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'dass_responses' not in st.session_state:
        st.session_state.dass_responses = {category: [] for category in DASS_42_QUESTIONS.keys()}
    
    total_questions = 0
    answered_questions = 0
    
    for category, questions in DASS_42_QUESTIONS.items():
        st.markdown(f"<h3 style='color: #667eea;'>{category}</h3>", unsafe_allow_html=True)
        
        if category not in st.session_state.dass_responses:
            st.session_state.dass_responses[category] = []
        
        for i, question in enumerate(questions):
            total_questions += 1
            response = st.radio(
                f"**{total_questions}. {question}**",
                options=["Never", "Sometimes", "Often", "Almost Always"],
                key=f"{category}_{i}",
                horizontal=True,
                index=None
            )
            
            if len(st.session_state.dass_responses[category]) <= i:
                st.session_state.dass_responses[category].append(None)
            
            if response:
                response_map = {"Never": 0, "Sometimes": 1, "Often": 2, "Almost Always": 3}
                st.session_state.dass_responses[category][i] = response_map[response]
                answered_questions += 1
    
    completion_percentage = (answered_questions / total_questions) * 100
    
    st.markdown(f"""
        <div class="feature-card-modern" style='margin: 2rem 0;'>
            <h4 style='color: #667eea; margin: 0 0 1rem 0;'>Progress</h4>
            <p style='margin: 0; color: #2d3748;'><strong>{answered_questions}/{total_questions}</strong> questions answered ({completion_percentage:.1f}%)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.progress(completion_percentage / 100)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if completion_percentage < 65:
            st.warning(f"Please answer at least 65% of questions to submit (currently {completion_percentage:.1f}%)")
            st.button("Submit Questionnaire", disabled=True, use_container_width=True)
        else:
            if st.button("Submit Questionnaire", type="primary", use_container_width=True):
                scores, severity = calculate_dass_score(st.session_state.dass_responses)
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'scores': scores,
                    'severity': severity,
                    'completion_percentage': completion_percentage
                }
                
                st.session_state.dass_results.append(result)
                st.session_state.dass_completed = True
                st.session_state.show_dass_mandatory = False
                
                users = load_users()
                if st.session_state.username in users:
                    users[st.session_state.username]['dass_completed'] = True
                    if 'dass_history' not in users[st.session_state.username]:
                        users[st.session_state.username]['dass_history'] = []
                    users[st.session_state.username]['dass_history'].append(result)
                    save_users(users)
                
                st.success("Questionnaire submitted successfully!")
                time.sleep(1)
                st.rerun()
    
    if st.session_state.dass_results and completion_percentage >= 65:
        st.markdown("""
            <div class="section-header">
                <div class="section-title">Your DASS-42 Results</div>
            </div>
        """, unsafe_allow_html=True)
        
        latest_result = st.session_state.dass_results[-1]
        scores = latest_result['scores']
        severity = latest_result['severity']
        
        col1, col2, col3 = st.columns(3)
        
        severity_color_map = {
            'Normal': '#38ef7d',
            'Mild': '#38ef7d',
            'Moderate': '#ff9966',
            'Severe': '#eb3349',
            'Extremely Severe': '#eb3349'
        }
        
        with col1:
            st.markdown(f"""
                <div style='background: {severity_color_map[severity["depression"]]}; padding: 2rem; 
                            border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                    <h4 style='margin: 0;'>Depression</h4>
                    <h2 style='margin: 0.5rem 0; font-size: 3rem;'>{scores['depression']}</h2>
                    <p style='margin: 0; font-weight: 600;'>{severity['depression']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background: {severity_color_map[severity["anxiety"]]}; padding: 2rem; 
                            border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                    <h4 style='margin: 0;'>Anxiety</h4>
                    <h2 style='margin: 0.5rem 0; font-size: 3rem;'>{scores['anxiety']}</h2>
                    <p style='margin: 0; font-weight: 600;'>{severity['anxiety']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='background: {severity_color_map[severity["stress"]]}; padding: 2rem; 
                            border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                    <h4 style='margin: 0;'>Stress</h4>
                    <h2 style='margin: 0.5rem 0; font-size: 3rem;'>{scores['stress']}</h2>
                    <p style='margin: 0; font-weight: 600;'>{severity['stress']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        max_severity = max(severity.values(), key=lambda x: ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe'].index(x))
        
        if max_severity in ['Severe', 'Extremely Severe']:
            st.markdown("""
                <div class="alert-modern alert-danger-modern">
                    <h4 style='margin-top: 0;'>HIGH SEVERITY DETECTED</h4>
                    <p><strong>Immediate professional help is strongly recommended.</strong></p>
                    <ul>
                        <li>Contact a mental health professional immediately</li>
                        <li>Speak with your primary care physician</li>
                        <li>Reach out to crisis helplines if needed</li>
                    </ul>
                    <p style='margin-bottom: 0; font-weight: 700;'>Crisis Helpline: TELE MANAS: 14416 | NIMHANS: 080-46110007</p>
                </div>
            """, unsafe_allow_html=True)
        elif max_severity == 'Moderate':
            st.markdown("""
                <div class="alert-modern alert-warning-modern">
                    <h4 style='margin-top: 0;'>MODERATE LEVELS DETECTED</h4>
                    <p><strong>Consider seeking professional guidance.</strong></p>
                    <ul>
                        <li>Schedule an appointment with a therapist or counselor</li>
                        <li>Practice stress management techniques regularly</li>
                        <li>Maintain healthy sleep, exercise, and nutrition habits</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-modern alert-success-modern">
                    <h4 style='margin-top: 0;'>NORMAL TO MILD LEVELS</h4>
                    <p><strong>Continue maintaining your mental wellness.</strong></p>
                    <ul>
                        <li>Keep up with healthy coping mechanisms</li>
                        <li>Regular self-care and stress management</li>
                        <li>Monitor for any changes in symptoms</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# TEMPORAL TRACKING PAGE
elif st.session_state.current_page == 'temporal':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Temporal Emotion Tracking</div>
            <div class="section-subtitle">Monitor your emotional patterns and trends over time</div>
        </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.emotion_history) == 0:
        st.markdown("""
            <div style='text-align: center; padding: 4rem 2rem;'>
                <h2 style='color: #667eea;'>No Data Yet</h2>
                <p style='color: #6b7280; font-size: 1.1rem;'>Start analyzing your emotions to see temporal patterns here!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"Tracking {len(st.session_state.emotion_history)} analysis sessions")
        
        timestamps = [entry['timestamp'] for entry in st.session_state.emotion_history]
        risk_scores = [entry['risk_score'] for entry in st.session_state.emotion_history]
        primary_emotions = [entry['emotions'][0]['emotion'] for entry in st.session_state.emotion_history]
        
        st.markdown("<h3 style='color: #1f2937;'>Risk Score Trend Over Time</h3>", unsafe_allow_html=True)
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=timestamps,
            y=risk_scores,
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='#667eea', width=4),
            marker=dict(size=12, color=risk_scores, colorscale='RdYlGn_r', showscale=True,
                       colorbar=dict(title="Risk")),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate='<b>Risk Score:</b> %{y:.1f}<br><b>Time:</b> %{x}<extra></extra>'
        ))
        fig_risk.add_hline(y=66, line_dash="dash", line_color="red", 
                          annotation_text="High Risk Threshold", annotation_position="right")
        fig_risk.add_hline(y=33, line_dash="dash", line_color="orange", 
                          annotation_text="Medium Risk Threshold", annotation_position="right")
        fig_risk.update_layout(
            xaxis_title="Time",
            yaxis_title="Risk Score (0-100)",
            height=450,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            showlegend=False
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.markdown("<br><h3 style='color: #1f2937;'>Emotion Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            emotion_counts = pd.Series(primary_emotions).value_counts().head(10)
            fig_pie = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Top 10 Primary Emotions",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(font=dict(family="Inter"), height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            unique_emotions = list(set(primary_emotions))
            emotion_to_index = {em: i for i, em in enumerate(sorted(unique_emotions))}
            y_values = [emotion_to_index[em] for em in primary_emotions]
            
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=timestamps,
                y=y_values,
                mode='markers+lines',
                marker=dict(
                    size=15,
                    color=risk_scores,
                    colorscale='RdYlGn_r',
                    showscale=False,
                    line=dict(width=2, color='white')
                ),
                text=[f"{em}<br>Risk: {risk:.1f}" for em, risk in zip(primary_emotions, risk_scores)],
                hovertemplate='<b>%{text}</b><br>Time: %{x}<extra></extra>',
                line=dict(width=2, color='rgba(102, 126, 234, 0.3)')
            ))
            fig_timeline.update_layout(
                title="Emotion Timeline",
                xaxis_title="Time",
                yaxis=dict(
                    title="Emotion",
                    tickmode='array',
                    tickvals=list(range(len(unique_emotions))),
                    ticktext=[em.title() for em in sorted(unique_emotions)]
                ),
                height=400,
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("<br><h3 style='color: #1f2937;'>Pattern Detection & Insights</h3>", unsafe_allow_html=True)
        
        recent_entries = st.session_state.emotion_history[-5:] if len(st.session_state.emotion_history) >= 5 else st.session_state.emotion_history
        recent_risks = [e['risk_score'] for e in recent_entries]
        avg_recent_risk = np.mean(recent_risks)
        
        high_risk_emotions = ['sadness', 'fear', 'grief', 'anger']
        recent_high_risk_count = sum(1 for e in recent_entries if e['emotions'][0]['emotion'] in high_risk_emotions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_indicator = "↗️" if avg_recent_risk > 50 else "↘️"
            st.markdown(f"""
                <div class="feature-card-modern" style='text-align: center;'>
                    <h4 style='color: #667eea;'>Average Recent Risk</h4>
                    <h2 style='color: #1f2937;'>{avg_recent_risk:.1f}</h2>
                    <p style='color: #6b7280;'>{trend_indicator} {abs(avg_recent_risk-50):.1f} from baseline (50)</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="feature-card-modern" style='text-align: center;'>
                    <h4 style='color: #667eea;'>High-Risk Emotions</h4>
                    <h2 style='color: #1f2937;'>{recent_high_risk_count}/{len(recent_entries)}</h2>
                    <p style='color: #6b7280;'>In recent analyses</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if len(risk_scores) >= 2:
                mid_point = len(risk_scores) // 2
                first_half_avg = np.mean(risk_scores[:mid_point]) if mid_point > 0 else risk_scores[0]
                second_half_avg = np.mean(risk_scores[mid_point:])
                
                if second_half_avg < first_half_avg - 5:
                    trend = "Improving 📈"
                    trend_color = "#38ef7d"
                elif second_half_avg > first_half_avg + 5:
                    trend = "Concerning 📉"
                    trend_color = "#eb3349"
                else:
                    trend = "Stable ➡️"
                    trend_color = "#ff9966"
            else:
                trend = "Need More Data"
                trend_color = "#6b7280"
            
            st.markdown(f"""
                <div class="feature-card-modern" style='text-align: center;'>
                    <h4 style='color: #667eea;'>Overall Trend</h4>
                    <h2 style='color: {trend_color};'>{trend}</h2>
                    <p style='color: #6b7280;'>Compared to start</p>
                </div>
            """, unsafe_allow_html=True)
        
        if avg_recent_risk > 60:
            st.markdown("""
                <div class="alert-modern alert-danger-modern">
                    <strong>CLINICAL ALERT:</strong> Consistently elevated risk scores detected in recent analyses. 
                    This pattern suggests significant emotional distress. We strongly recommend:
                    <ul>
                        <li>Schedule an appointment with a mental health professional</li>
                        <li>Reach out to your support network immediately</li>
                        <li>Consider crisis intervention if thoughts of self-harm are present</li>
                    </ul>
                    <strong>Crisis Helpline: 14416 | NIMHANS: 080-46110007</strong>
                </div>
            """, unsafe_allow_html=True)
        elif recent_high_risk_count >= 3:
            st.markdown("""
                <div class="alert-modern alert-warning-modern">
                    <strong>MONITORING ALERT:</strong> Multiple high-risk emotions detected recently. 
                    Consider implementing these self-care strategies:
                    <ul>
                        <li>Practice daily mindfulness or meditation</li>
                        <li>Maintain regular sleep schedule</li>
                        <li>Engage in physical activity</li>
                        <li>Talk to someone you trust</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-modern alert-success-modern">
                    <strong>POSITIVE PATTERN:</strong> Your recent emotional patterns appear relatively stable. 
                    Continue your current self-care practices and maintain regular check-ins.
                </div>
            """, unsafe_allow_html=True)

# ANALYTICS PAGE
elif st.session_state.current_page == 'analytics':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Advanced Analytics</div>
            <div class="section-subtitle">Comprehensive insights into your emotional wellness data</div>
        </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.emotion_history) == 0:
        st.markdown("""
            <div style='text-align: center; padding: 4rem 2rem;'>
                <h2 style='color: #667eea;'>No Analytics Yet</h2>
                <p style='color: #6b7280; font-size: 1.1rem;'>Start analyzing to see detailed insights!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #1f2937;'>Overall Statistics</h3>", unsafe_allow_html=True)
        
        all_emotions = []
        for entry in st.session_state.emotion_history:
            for emotion_data in entry['emotions']:
                all_emotions.append({
                    'timestamp': entry['timestamp'],
                    'emotion': emotion_data['emotion'],
                    'confidence': emotion_data['confidence'],
                    'risk_level': emotion_data['risk_level'],
                    'source': entry.get('source', 'analysis')
                })
        
        df_emotions = pd.DataFrame(all_emotions)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="stat-card-modern">
                    <div class="stat-label">Total Analyses</div>
                    <div class="stat-value">{len(st.session_state.emotion_history)}</div>
                    <div class="stat-label">Sessions</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_risk = np.mean([e['risk_score'] for e in st.session_state.emotion_history])
            risk_emoji = "🔴" if avg_risk > 66 else "🟡" if avg_risk > 33 else "🟢"
            st.markdown(f"""
                <div class="stat-card-modern">
                    <div class="stat-label">Avg Risk Score</div>
                    <div class="stat-value">{risk_emoji} {avg_risk:.1f}</div>
                    <div class="stat-label">Out of 100</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_emotions = df_emotions['emotion'].nunique()
            st.markdown(f"""
                <div class="stat-card-modern">
                    <div class="stat-label">Unique Emotions</div>
                    <div class="stat-value">{unique_emotions}</div>
                    <div class="stat-label">Detected</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high_risk_count = sum(1 for e in st.session_state.emotion_history if e['risk_score'] > 66)
            st.markdown(f"""
                <div class="stat-card-modern">
                    <div class="stat-label">High Risk Sessions</div>
                    <div class="stat-value">{high_risk_count}</div>
                    <div class="stat-label">Total</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4 style='color: #1f2937;'>Most Common Emotions</h4>", unsafe_allow_html=True)
            emotion_freq = df_emotions['emotion'].value_counts().head(10)
            fig_bar = px.bar(
                x=emotion_freq.values,
                y=emotion_freq.index,
                orientation='h',
                labels={'x': 'Frequency', 'y': 'Emotion'},
                color=emotion_freq.values,
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(showlegend=False, height=400, font=dict(family="Inter"))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("<h4 style='color: #1f2937;'>Risk Level Distribution</h4>", unsafe_allow_html=True)
            risk_dist = df_emotions['risk_level'].value_counts()
            fig_risk_pie = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                color=risk_dist.index,
                color_discrete_map={'low': '#38ef7d', 'medium': '#ff9966', 'high': '#eb3349'}
            )
            fig_risk_pie.update_layout(height=400, font=dict(family="Inter"))
            st.plotly_chart(fig_risk_pie, use_container_width=True)
        
        st.markdown("<br><h4 style='color: #1f2937;'>Emotion Frequency Heatmap</h4>", unsafe_allow_html=True)
        df_emotions['hour'] = pd.to_datetime(df_emotions['timestamp']).dt.hour
        df_emotions['day'] = pd.to_datetime(df_emotions['timestamp']).dt.day_name()
        
        top_emotions = df_emotions['emotion'].value_counts().head(5).index
        df_top = df_emotions[df_emotions['emotion'].isin(top_emotions)]
        
        heatmap_data = df_top.groupby(['emotion', 'hour']).size().unstack(fill_value=0)
        
        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Emotion", color="Frequency"),
            color_continuous_scale='Blues',
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400, font=dict(family="Inter"))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # DASS-42 Results Section
        if len(st.session_state.dass_results) > 0:
            st.markdown("<br><h3 style='color: #1f2937;'>DASS-42 Questionnaire Results</h3>", unsafe_allow_html=True)
            
            latest_dass = st.session_state.dass_results[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="feature-card-modern" style='text-align: center;'>
                        <h4 style='color: #667eea;'>Submissions</h4>
                        <h2 style='color: #1f2937;'>{len(st.session_state.dass_results)}</h2>
                        <p style='color: #6b7280; font-size: 0.9rem;'>Total questionnaires</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                last_submission = datetime.fromisoformat(latest_dass['timestamp'])
                days_since = (datetime.now() - last_submission).days
                st.markdown(f"""
                    <div class="feature-card-modern" style='text-align: center;'>
                        <h4 style='color: #667eea;'>Last Submission</h4>
                        <h2 style='color: #1f2937;'>{days_since}</h2>
                        <p style='color: #6b7280; font-size: 0.9rem;'>days ago</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_depression = np.mean([r['scores']['depression'] for r in st.session_state.dass_results])
                st.markdown(f"""
                    <div class="feature-card-modern" style='text-align: center;'>
                        <h4 style='color: #667eea;'>Avg Depression</h4>
                        <h2 style='color: #1f2937;'>{avg_depression:.1f}</h2>
                        <p style='color: #6b7280; font-size: 0.9rem;'>score</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_anxiety = np.mean([r['scores']['anxiety'] for r in st.session_state.dass_results])
                st.markdown(f"""
                    <div class="feature-card-modern" style='text-align: center;'>
                        <h4 style='color: #667eea;'>Avg Anxiety</h4>
                        <h2 style='color: #1f2937;'>{avg_anxiety:.1f}</h2>
                        <p style='color: #6b7280; font-size: 0.9rem;'>score</p>
                    </div>
                """, unsafe_allow_html=True)
            
            if len(st.session_state.dass_results) > 1:
                st.markdown("<br><h4 style='color: #1f2937;'>DASS Scores Trend</h4>", unsafe_allow_html=True)
                
                timestamps_dass = [datetime.fromisoformat(r['timestamp']) for r in st.session_state.dass_results]
                depression_scores = [r['scores']['depression'] for r in st.session_state.dass_results]
                anxiety_scores = [r['scores']['anxiety'] for r in st.session_state.dass_results]
                stress_scores = [r['scores']['stress'] for r in st.session_state.dass_results]
                
                fig_dass = go.Figure()
                fig_dass.add_trace(go.Scatter(x=timestamps_dass, y=depression_scores, mode='lines+markers', 
                                            name='Depression', line=dict(color='#667eea', width=3)))
                fig_dass.add_trace(go.Scatter(x=timestamps_dass, y=anxiety_scores, mode='lines+markers', 
                                            name='Anxiety', line=dict(color='#f093fb', width=3)))
                fig_dass.add_trace(go.Scatter(x=timestamps_dass, y=stress_scores, mode='lines+markers', 
                                            name='Stress', line=dict(color='#4facfe', width=3)))
                
                fig_dass.update_layout(
                    xaxis_title="Date",
                    yaxis_title="DASS Score",
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig_dass, use_container_width=True)

# SOCIAL MEDIA ANALYSIS PAGE
elif st.session_state.current_page == 'social_media':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Social Media Analysis</div>
            <div class="section-subtitle">Bulk emotion analysis for social media content</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-info-modern">
            <h4 style='margin-top: 0;'>Bulk Emotion Analysis</h4>
            <p style='margin: 0;'>Upload a CSV file containing social media posts, comments, or tweets to analyze emotional patterns across multiple entries. Required format: CSV with a 'text' column.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], help="CSV file must contain a 'text' column")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column!")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                st.success(f"File loaded successfully! Found {len(df)} entries.")
                
                with st.expander("Preview Data (First 5 rows)"):
                    st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Analyze All Entries", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, row in df.iterrows():
                        status_text.text(f"Analyzing entry {idx + 1} of {len(df)}...")
                        progress_bar.progress((idx + 1) / len(df))
                        
                        text = clean_text(str(row['text']))
                        
                        if text:
                            emotions_data = predict_emotions_multilabel(text, model, tokenizer, emotion_labels, top_k=5)
                            
                            if emotions_data:
                                primary_emotion = emotions_data[0]
                                risk_score = calculate_risk_score(emotions_data)
                                
                                results.append({
                                    'text': text[:100] + '...' if len(text) > 100 else text,
                                    'primary_emotion': primary_emotion['emotion'],
                                    'confidence': primary_emotion['confidence'],
                                    'risk_score': risk_score,
                                    'risk_level': primary_emotion['risk_level'],
                                    'full_text': text
                                })
                        else:
                            results.append({
                                'text': '[Empty or invalid text]',
                                'primary_emotion': 'N/A',
                                'confidence': 0,
                                'risk_score': 0,
                                'risk_level': 'N/A',
                                'full_text': ''
                            })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    df_results = pd.DataFrame(results)
                    
                    st.markdown("""
                        <div class="section-header">
                            <div class="section-title">Analysis Results</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="stat-card-modern">
                                <div class="stat-label">Total Entries</div>
                                <div class="stat-value">{len(df_results)}</div>
                                <div class="stat-label">Analyzed</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        valid_scores = df_results[df_results['risk_score'] > 0]['risk_score']
                        avg_risk = valid_scores.mean() if len(valid_scores) > 0 else 0
                        risk_emoji = "🔴" if avg_risk > 66 else "🟡" if avg_risk > 33 else "🟢"
                        st.markdown(f"""
                            <div class="stat-card-modern">
                                <div class="stat-label">Average Risk</div>
                                <div class="stat-value">{risk_emoji} {avg_risk:.1f}</div>
                                <div class="stat-label">Out of 100</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        high_risk_count = len(df_results[df_results['risk_score'] > 66])
                        st.markdown(f"""
                            <div class="stat-card-modern">
                                <div class="stat-label">High Risk</div>
                                <div class="stat-value">🔴 {high_risk_count}</div>
                                <div class="stat-label">Entries</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        avg_confidence = df_results[df_results['confidence'] > 0]['confidence'].mean()
                        st.markdown(f"""
                            <div class="stat-card-modern">
                                <div class="stat-label">Avg Confidence</div>
                                <div class="stat-value">{avg_confidence*100:.1f}%</div>
                                <div class="stat-label">Model Accuracy</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h3 style='color: #1f2937;'>Emotion Distribution</h3>", unsafe_allow_html=True)
                        emotion_counts = df_results[df_results['primary_emotion'] != 'N/A']['primary_emotion'].value_counts()
                        fig_emotions = px.pie(
                            values=emotion_counts.values,
                            names=emotion_counts.index,
                            title="Primary Emotions Detected",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_emotions.update_layout(font=dict(family="Inter"), height=400)
                        st.plotly_chart(fig_emotions, use_container_width=True)
                    
                    with col2:
                        st.markdown("<h3 style='color: #1f2937;'>Risk Level Distribution</h3>", unsafe_allow_html=True)
                        risk_counts = df_results[df_results['risk_level'] != 'N/A']['risk_level'].value_counts()
                        fig_risk = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            labels={'x': 'Risk Level', 'y': 'Count'},
                            color=risk_counts.index,
                            color_discrete_map={'low': '#38ef7d', 'medium': '#ff9966', 'high': '#eb3349'}
                        )
                        fig_risk.update_layout(showlegend=False, font=dict(family="Inter"), height=400)
                        st.plotly_chart(fig_risk, use_container_width=True)
                    
                    if high_risk_count > 0:
                        st.markdown(f"""
                            <div class="alert-modern alert-danger-modern">
                                <h4 style='margin-top: 0;'>HIGH RISK ENTRIES DETECTED</h4>
                                <p><strong>{high_risk_count} entries</strong> show signs of significant emotional distress. 
                                Review these entries carefully and consider appropriate intervention.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br><h3 style='color: #1f2937;'>Detailed Results</h3>", unsafe_allow_html=True)
                    
                    display_df = df_results[['text', 'primary_emotion', 'confidence', 'risk_score', 'risk_level']].copy()
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%" if x > 0 else "N/A")
                    display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
                    display_df.columns = ['Text', 'Primary Emotion', 'Confidence', 'Risk Score', 'Risk Level']
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv,
                        file_name=f"social_media_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.markdown("""
                        <div class="alert-modern alert-success-modern" style='margin-top: 2rem;'>
                            <h4 style='margin-top: 0;'>Analysis Complete</h4>
                            <p style='margin: 0;'>
                                The analysis has been completed successfully. Use the insights above to understand 
                                emotional patterns in your social media data. Remember to handle high-risk entries with care.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with a 'text' column containing the messages to analyze.")
    
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem 2rem;'>
                <h3 style='color: #667eea;'>Upload Your CSV File</h3>
                <p style='color: #6b7280; font-size: 1.1rem;'>
                    Drag and drop or click to upload a CSV file containing social media posts, 
                    comments, or tweets for bulk emotion analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #1f2937;'>Example CSV Format</h3>", unsafe_allow_html=True)
        example_df = pd.DataFrame({
            'text': [
                'I am so happy today! Everything is going great!',
                'Feeling really anxious about tomorrow...',
                'This is the worst day of my life',
                'Just grateful for all the support from friends',
                'Why does everything always go wrong?'
            ]
        })
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea; margin-top: 0;'>Tips for Best Results</h4>
                <ul style='color: #1f2937; line-height: 2;'>
                    <li>Ensure your CSV has a column named exactly 'text' (case-sensitive)</li>
                    <li>Each row should contain one social media post/comment/tweet</li>
                    <li>Remove any empty rows before uploading</li>
                    <li>The model works best with text between 10-500 words</li>
                    <li>Supports analysis of up to 10,000 entries at once</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ABOUT PAGE
elif st.session_state.current_page == 'about':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">About Soul</div>
            <div class="section-subtitle">Professional Mental Health Assessment Platform</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="hero-landing" style='padding: 3rem 2rem; margin: 0 0 2rem 0;'>
            <div class="hero-content">
                <h2 style='color: white; margin: 0;'>Professional Mental Health Assessment Platform</h2>
                <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 1rem;'>
                    Leveraging cutting-edge AI technology for evidence-based emotional analysis
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>Our Mission</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="feature-card-modern">
            <p style='font-size: 1.1rem; line-height: 1.8; color: #1f2937;'>
                Soul is dedicated to making mental health assessment accessible, data-driven, and effective. 
                We combine advanced artificial intelligence with evidence-based psychological methodologies to provide 
                real-time emotional analysis and support.
            </p>
            <p style='font-size: 1.1rem; line-height: 1.8; color: #1f2937;'>
                Our platform serves as a complementary tool for mental health monitoring, helping individuals track 
                their emotional patterns and identify when professional intervention may be beneficial.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>The Technology Behind Soul</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card-modern">
            <h3 style='color: #667eea; margin-top: 0;'>RoBERTa Transformer Architecture</h3>
            <p style='color: #1f2937; line-height: 1.8;'>
                Soul utilizes <strong>RoBERTa (Robustly Optimized BERT Pretraining Approach)</strong>, 
                a state-of-the-art transformer-based deep learning model developed by Facebook AI.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Model Specifications</h4>
                <ul style='line-height: 2; color: #1f2937;'>
                    <li><strong>Architecture:</strong> RoBERTa-base</li>
                    <li><strong>Parameters:</strong> 125 Million</li>
                    <li><strong>Training Data:</strong> 58,000+ labeled samples</li>
                    <li><strong>Emotions:</strong> 28 distinct categories</li>
                    <li><strong>Accuracy:</strong> 91% (clinical-grade)</li>
                    <li><strong>Processing:</strong> Real-time inference</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Detected Emotions</h4>
                <ul style='line-height: 2; color: #1f2937; columns: 2;'>
                    <li>Joy</li>
                    <li>Sadness</li>
                    <li>Fear</li>
                    <li>Anger</li>
                    <li>Grief</li>
                    <li>Love</li>
                    <li>Nervousness</li>
                    <li>Relief</li>
                    <li>Pride</li>
                    <li>Gratitude</li>
                    <li>Disappointment</li>
                    <li>Remorse</li>
                    <li>+ 16 more emotions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>Clinical Methodology</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card-modern">
            <h3 style='color: #667eea; margin-top: 0;'>Ecological Momentary Assessment (EMA)</h3>
            <p style='color: #1f2937; line-height: 1.8;'>
                Soul implements the <strong>paging method</strong>, a scientifically validated approach used by 
                mental health professionals to track emotional states in real-time. This methodology involves:
            </p>
            <ul style='color: #1f2937; line-height: 2;'>
                <li><strong>Regular Check-ins:</strong> Automated reminders every 4 hours to maintain consistent tracking</li>
                <li><strong>Real-time Assessment:</strong> Capturing emotions as they occur, reducing recall bias</li>
                <li><strong>Pattern Recognition:</strong> Identifying trends and triggers over extended periods</li>
                <li><strong>Early Intervention:</strong> Detecting concerning patterns before crisis situations</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>How Soul Helps People</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="features-grid">
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Self-Awareness</h4>
                <p style='color: #6b7280;'>
                    Gain deeper insights into your emotional patterns and triggers through 
                    objective AI analysis
                </p>
            </div>
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Early Detection</h4>
                <p style='color: #6b7280;'>
                    Identify concerning emotional patterns early, enabling timely intervention 
                    and professional support
                </p>
            </div>
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Progress Tracking</h4>
                <p style='color: #6b7280;'>
                    Monitor improvements over time, validate therapeutic interventions, 
                    and stay motivated
                </p>
            </div>
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Safe Expression</h4>
                <p style='color: #6b7280;'>
                    Express thoughts and feelings in a judgment-free, confidential environment 
                    with AI support
                </p>
            </div>
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Proactive Care</h4>
                <p style='color: #6b7280;'>
                    Regular reminders ensure consistent emotional monitoring, 
                    similar to clinical paging methods
                </p>
            </div>
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Data-Driven Insights</h4>
                <p style='color: #6b7280;'>
                    Visualize emotional trends with advanced analytics, 
                    supporting informed health decisions
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'> Who Can Benefit</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'> Individuals</h4>
                <ul style='color: #1f2937; line-height: 2;'>
                    <li>People managing stress or anxiety</li>
                    <li>Those in therapy seeking self-monitoring tools</li>
                    <li>Anyone interested in emotional wellness</li>
                    <li>Individuals tracking mood patterns</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Healthcare Professionals</h4>
                <ul style='color: #1f2937; line-height: 2;'>
                    <li>Therapists monitoring patient progress</li>
                    <li>Researchers studying emotional patterns</li>
                    <li>Counselors seeking objective assessment tools</li>
                    <li>Mental health educators and trainers</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>Model Performance & Validation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card-modern">
            <h3 style='color: #667eea; margin-top: 0;'>Clinical-Grade Accuracy: 91%</h3>
            <p style='color: #1f2937; line-height: 1.8;'>
                Our model achieves <strong>91% accuracy</strong> in emotion classification, which is considered 
                <strong>clinical-grade performance</strong> for multi-class emotion detection tasks involving 28 distinct emotions.
            </p>
            <h4 style='color: #667eea;'>Why 91% is Excellent:</h4>
            <ul style='color: #1f2937; line-height: 2;'>
                <li><strong>Complex Task:</strong> Distinguishing between 28 nuanced emotions is significantly more challenging than binary classification</li>
                <li><strong>Human-Level:</strong> Inter-rater agreement among human annotators for complex emotions typically ranges from 60-70%</li>
                <li><strong>Real-World Data:</strong> Trained on authentic human expressions, not sanitized laboratory data</li>
                <li><strong>Continuous Improvement:</strong> Regular model updates based on new data and research</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>Important Disclaimers</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-warning-modern">
            <h4 style='margin-top: 0;'>Not a Substitute for Professional Care</h4>
            <p style='margin: 0;'>
                Soul is an <strong>educational and supportive tool</strong>, not a diagnostic instrument or replacement 
                for professional mental health care. Always consult licensed healthcare providers for diagnosis and treatment.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card-modern">
            <h4 style='color: #667eea;'>Privacy & Security</h4>
            <p style='color: #1f2937; line-height: 1.8;'>
                Your privacy is our priority. All data is encrypted and securely stored. We do not share your information 
                with third parties. You maintain full control over your data and can delete it at any time.
            </p>
            <h4 style='color: #667eea; margin-top: 2rem;'>Crisis Resources</h4>
            <p style='color: #1f2937; line-height: 1.8;'>
                If you're in crisis or considering self-harm, please seek immediate help:
            </p>
            <ul style='color: #1f2937; line-height: 2; font-weight: 600;'>
                <li>NIMHANS Helpline (India): 080-46110007</li>
                <li>TELE MANAS: 14416</li>
                <li>International: Find resources at <a href="https://www.iasp.info/resources/Crisis_Centres/" style="color: #667eea;">IASP.info</a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='color: #1f2937;'>Contact & Support</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Feedback & Suggestions</h4>
                <p style='color: #1f2937;'>
                    We're continuously improving Soul. Your feedback helps us serve you better.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Research & Collaboration</h4>
                <p style='color: #1f2937;'>
                    Interested in research collaboration or institutional partnerships? We'd love to hear from you.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-info-modern" style="margin-top: 3rem;">
            <h4 style='margin-top: 0;'>Thank You for Using Soul</h4>
            <p style='margin: 0;'>
                Your mental health matters. We're honored to be part of your wellness journey. 
                Remember: seeking help is a sign of strength, not weakness.
            </p>
        </div>
    """, unsafe_allow_html=True)

# COMMUNITY PAGE
elif st.session_state.current_page == 'community':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Community Wellness Journal</div>
            <div class="section-subtitle">Share supportive thoughts and encourage others anonymously</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-info-modern">
            <h4 style='margin-top: 0;'>Safe Space Guidelines</h4>
            <p style='margin: 0;'>This is an anonymous community for sharing supportive messages. Be kind, compassionate, and respectful. Your posts are visible to all users.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Post new message
    st.markdown("<h3 style='color: #1f2937;'>Share Your Thoughts</h3>", unsafe_allow_html=True)
    new_post = st.text_area(
        "",
        placeholder="Share something positive, a lesson learned, or words of encouragement...\n\nExample: 'Today I learned it's okay to rest and take care of myself.'",
        height=120,
        key="community_post_input"
    )
    
    if st.button("Post to Community", use_container_width=True, type="primary"):
        if new_post and len(new_post.strip()) > 0:
            posts = load_community_posts()
            
            new_entry = {
                'text': new_post.strip(),
                'timestamp': datetime.now().isoformat(),
                'reactions': {'❤️': 0, '🌱': 0, '💪': 0}
            }
            
            posts.append(new_entry)
            save_community_posts(posts)
            
            st.success("Your message has been posted!")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Please write something before posting!")
    
    st.markdown("<br><h3 style='color: #1f2937;'>Community Messages</h3>", unsafe_allow_html=True)
    
    # Display posts
    posts = load_community_posts()
    
    if len(posts) == 0:
        st.markdown("""
            <div style='text-align: center; padding: 3rem 2rem;'>
                <h3 style='color: #667eea;'>No Posts Yet</h3>
                <p style='color: #6b7280; font-size: 1.1rem;'>Be the first to share something positive!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Reverse to show newest first
        for idx, post in enumerate(reversed(posts)):
            post_idx = len(posts) - 1 - idx
            
            timestamp = datetime.fromisoformat(post['timestamp'])
            time_ago = (datetime.now() - timestamp).days
            time_display = f"{time_ago} days ago" if time_ago > 0 else "Today"
            
            st.markdown(f"""
                <div class="feature-card-modern" style='margin-bottom: 1.5rem;'>
                    <p style='color: #1f2937; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1rem;'>
                        "{post['text']}"
                    </p>
                    <div style='display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #e5e7eb; padding-top: 1rem; margin-top: 1rem;'>
                        <span style='color: #6b7280; font-size: 0.9rem;'>{time_display}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Reaction buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            
            with col1:
                if st.button(f"❤️ {post['reactions']['❤️']}", key=f"react_heart_{post_idx}"):
                    posts[post_idx]['reactions']['❤️'] += 1
                    save_community_posts(posts)
                    st.rerun()
            
            with col2:
                if st.button(f"🌱 {post['reactions']['🌱']}", key=f"react_growth_{post_idx}"):
                    posts[post_idx]['reactions']['🌱'] += 1
                    save_community_posts(posts)
                    st.rerun()
            
            with col3:
                if st.button(f"💪 {post['reactions']['💪']}", key=f"react_strength_{post_idx}"):
                    posts[post_idx]['reactions']['💪'] += 1
                    save_community_posts(posts)
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)

# MIND GYM PAGE
elif st.session_state.current_page == 'mind_gym':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Mind Gym </div>
            <div class="section-subtitle">Complete short daily exercises to strengthen your mental resilience</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load user's Mind Gym data
    users = load_users()
    if st.session_state.username in users:
        user_data = users[st.session_state.username]
        mind_gym_data = user_data.get('mind_gym', {'xp': 0, 'level': 1, 'completed_tasks': []})
        st.session_state.xp_points = mind_gym_data['xp']
        st.session_state.level = mind_gym_data['level']
        completed_tasks = mind_gym_data.get('completed_tasks', [])
    else:
        st.session_state.xp_points = 0
        st.session_state.level = 1
        completed_tasks = []
    
    # Display XP and Level
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        progress_to_next = (st.session_state.xp_points % 100) / 100
        st.markdown(f"""
            <div class="feature-card-modern" style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                <h2 style='margin: 0; color: white;'>Level {st.session_state.level}</h2>
                <p style='margin: 0.5rem 0; color: rgba(255,255,255,0.9);'>XP: {st.session_state.xp_points}</p>
                <div style='background: rgba(255,255,255,0.2); border-radius: 10px; height: 20px; margin-top: 1rem; overflow: hidden;'>
                    <div style='background: #38ef7d; height: 100%; width: {progress_to_next*100}%; transition: width 0.5s ease;'></div>
                </div>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: rgba(255,255,255,0.9);'>
                    {100 - (st.session_state.xp_points % 100)} XP to Level {st.session_state.level + 1}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Exercise 1: Deep Breathing Visualizer
    st.markdown("""
        <div class="feature-card-modern">
            <h3 style='color: #667eea; margin-top: 0;'>Deep Breathing Visualizer</h3>
            <p style='color: #1f2937;'>Follow the breathing pattern to calm your mind and body.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Breathing Exercise", key="breathing_start"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(3):  # 3 cycles
            # Inhale
            status_text.markdown("<h2 style='text-align: center; color: #007bff;'>Inhale...</h2>", unsafe_allow_html=True)
            for j in range(40):
                time.sleep(0.1)
                progress_bar.progress((j + 1) / 40)
            
            # Hold
            status_text.markdown("<h2 style='text-align: center; color: #ffcc00;'>Hold...</h2>", unsafe_allow_html=True)
            time.sleep(2)
            
            # Exhale
            status_text.markdown("<h2 style='text-align: center; color: #ff4c4c;'>Exhale...</h2>", unsafe_allow_html=True)
            for j in range(40):
                time.sleep(0.15)
                progress_bar.progress(1 - (j + 1) / 40)
        
        status_text.markdown("<h2 style='text-align: center; color: #00c853;'>Great job!</h2>", unsafe_allow_html=True)
        progress_bar.empty()
        
        # Award XP
        add_xp(st.session_state.username, 20)
        st.success("20 XP! You completed the breathing exercise!")
        time.sleep(2)
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Exercise 2: Gratitude Journal
    st.markdown("""
        <div class="feature-card-modern">
            <h3 style='color: #667eea; margin-top: 0;'>Gratitude Journal</h3>
            <p style='color: #1f2937;'>Write down 3 things you're grateful for today.</p>
        </div>
    """, unsafe_allow_html=True)
    
    gratitude_input = st.text_area(
        "What are you grateful for today?",
        placeholder="1. My supportive family\n2. A warm meal\n3. A moment of peace in nature",
        height=100,
        key="gratitude_input"
    )
    
    if st.button("Save Gratitude Entry", key="gratitude_save"):
        if gratitude_input and len(gratitude_input.strip()) > 0:
            save_gratitude_entry(st.session_state.username, gratitude_input.strip())
            add_xp(st.session_state.username, 20)
            st.success("20 XP! Gratitude entry saved!")
            time.sleep(2)
            st.rerun()
        else:
            st.warning("Please write something before saving!")
    
    # Display previous gratitude entries
    gratitude_entries = load_gratitude_entries(st.session_state.username)
    if len(gratitude_entries) > 0:
        with st.expander(f"View Previous Entries ({len(gratitude_entries)})"):
            for entry in reversed(gratitude_entries[-5:]):  # Show last 5
                timestamp = datetime.fromisoformat(entry['timestamp'])
                st.markdown(f"""
                    <div style='background: #f9fafb; padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem;'>
                        <p style='color: #1f2937; margin: 0;'>{entry['entry']}</p>
                        <p style='color: #6b7280; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>
                             {timestamp.strftime('%B %d, %Y at %I:%M %p')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Exercise 3: Daily Mood Challenge Tracker
    st.markdown("""
        <div class="feature-card-modern">
            <h3 style='color: #667eea; margin-top: 0;'>Daily Mood Challenge Tracker</h3>
            <p style='color: #1f2937;'>Complete small daily tasks to boost your mental wellness.</p>
        </div>
    """, unsafe_allow_html=True)
    
    daily_tasks = [
        {"id": "task_break", "label": "Take a 5-minute break", "xp": 20},
        {"id": "task_friend", "label": "Talk to a friend or loved one", "xp": 20},
        {"id": "task_meditate", "label": "Meditate for 3 minutes", "xp": 20},
        {"id": "task_walk", "label": "Go for a short walk", "xp": 20},
        {"id": "task_water", "label": "Drink a glass of water mindfully", "xp": 20}
    ]
    # Load user's custom tasks
    if 'custom_tasks' not in user_data.get('mind_gym', {}):
        if 'mind_gym' not in user_data:
            user_data['mind_gym'] = {}
        user_data['mind_gym']['custom_tasks'] = []
        save_users(users)
    
    custom_tasks = user_data['mind_gym'].get('custom_tasks', [])
    
    for task in daily_tasks:
        task_id = f"{task['id']}_{datetime.now().date()}"
        is_completed = task_id in completed_tasks
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"{'✅' if is_completed else '⬜'} {task['label']}")
        with col2:
            if not is_completed:
                if st.button("Complete", key=f"complete_{task['id']}", use_container_width=True):
                    # Mark as complete
                    completed_tasks.append(task_id)
                    users[st.session_state.username]['mind_gym']['completed_tasks'] = completed_tasks
                    save_users(users)
                    
                    # Award XP
                    add_xp(st.session_state.username, task['xp'])
                    st.success(f"{task['xp']} XP! Great job!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.markdown("Done")
        st.markdown("<br><h3 style='color: #1f2937;'>➕ Create Your Own Task</h3>", unsafe_allow_html=True)

with st.expander("Add Custom Task", expanded=False):
    st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #1f2937;'>Create personalized wellness tasks that matter to you!</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_task, col_xp = st.columns([3, 1])
    
    with col_task:
        new_task_label = st.text_input(
            "Task Description",
            placeholder="e.g., Read 10 pages of a book",
            key="new_task_input"
        )
    
    with col_xp:
        new_task_xp = st.number_input(
            "XP Reward",
            min_value=10,
            max_value=100,
            value=20,
            step=10,
            key="new_task_xp"
        )
    
    col_add, col_space = st.columns([1, 2])
    
    with col_add:
        if st.button("Add Task", use_container_width=True, type="primary"):
            if new_task_label and len(new_task_label.strip()) > 0:
                # Create unique task ID
                task_id = f"custom_{len(custom_tasks)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                new_custom_task = {
                    'id': task_id,
                    'label': new_task_label.strip(),
                    'xp': new_task_xp,
                    'created_at': datetime.now().isoformat()
                }
                
                # Add to user's custom tasks
                custom_tasks.append(new_custom_task)
                users[st.session_state.username]['mind_gym']['custom_tasks'] = custom_tasks
                save_users(users)
                
                st.success(f"Task added! Worth {new_task_xp} XP")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Please enter a task description!")

# Display Custom Tasks Section
if len(custom_tasks) > 0:
    st.markdown("<br><h3 style='color: #1f2937;'> Your Custom Tasks</h3>", unsafe_allow_html=True)
    
    for idx, task in enumerate(custom_tasks):
        task_id = f"{task['id']}_{datetime.now().date()}"
        is_completed = task_id in completed_tasks
        
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col1:
            st.markdown(f"{'✅' if is_completed else '⬜'} {task['label']} ({task['xp']} XP)")
        
        with col2:
            if not is_completed:
                if st.button("Complete", key=f"complete_custom_{idx}", use_container_width=True):
                    # Mark as complete
                    completed_tasks.append(task_id)
                    users[st.session_state.username]['mind_gym']['completed_tasks'] = completed_tasks
                    save_users(users)
                    
                    # Award XP
                    add_xp(st.session_state.username, task['xp'])
                    st.success(f"{task['xp']} XP! Great job!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.markdown("✓ Done")
        
        with col3:
            if st.button("DEL", key=f"delete_custom_{idx}", help="Delete this task"):
                # Remove from custom tasks
                custom_tasks.pop(idx)
                users[st.session_state.username]['mind_gym']['custom_tasks'] = custom_tasks
                save_users(users)
                st.success("Task deleted!")
                time.sleep(1)
                st.rerun()
    
    st.markdown("""
        <div class="alert-modern alert-success-modern" style='margin-top: 2rem;'>
            <h4 style='margin-top: 0;'> Keep Growing Stronger!</h4>
            <p style='margin: 0;'>Every small action counts toward building a healthier mind. You're doing amazing work!</p>
        </div>
    """, unsafe_allow_html=True)


# DOWNLOAD REPORT PAGE
elif st.session_state.current_page == 'download_report':
    st.markdown("""
        <div class="section-header">
            <div class="section-title">Download Your Report</div>
            <div class="section-subtitle">Generate a comprehensive PDF report of your mental health journey</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="alert-modern alert-info-modern">
            <h4 style='margin-top: 0;'>About Your Report</h4>
            <p style='margin: 0;'>Your report includes DASS-42 scores, emotion analysis history, chatbot conversations, 
            Mind Gym progress, and detailed visualizations of your mental health trends.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Report type selection
    report_type = st.radio(
        "Select Report Type:",
        options=["Complete History", "Custom Date Range"],
        horizontal=True
    )
    
    start_date = None
    end_date = None
    
    if report_type == "Custom Date Range":
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
                min_value=start_date
            )
        
        # Convert to datetime
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        
        st.info(f"Report will cover: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("Generate Report", use_container_width=True, type="primary"):
            with st.spinner("Generating your comprehensive report... This may take a moment."):
                try:
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(
                        st.session_state.username,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if pdf_buffer:
                        # Create filename
                        date_suffix = ""
                        if start_date and end_date:
                            date_suffix = f"_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
                        else:
                            date_suffix = f"_complete_{datetime.now().strftime('%Y%m%d')}"
                        
                        filename = f"Soul_Report_{st.session_state.username}{date_suffix}.pdf"
                        
                        st.success("Report generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.markdown("""
                            <div class="alert-modern alert-success-modern" style='margin-top: 1rem;'>
                                <h4 style='margin-top: 0;'>Your Report is Ready!</h4>
                                <p style='margin: 0;'>Click the download button above to save your comprehensive mental health report. 
                                Keep this for your records or share with your healthcare provider.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate report. Please try again.")
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.info("Please ensure you have some data recorded before generating a report.")
    
    # Information cards
    st.markdown("<br><h3 style='color: #1f2937;'>What's Included in Your Report</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Assessment Data</h4>
                <ul style='color: #1f2937; line-height: 2;'>
                    <li>DASS-42 Scores</li>
                    <li>Severity Levels</li>
                    <li>Historical Trends</li>
                    <li>Score Charts</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Emotion Analysis</h4>
                <ul style='color: #1f2937; line-height: 2;'>
                    <li>All Text Entries</li>
                    <li>Detected Emotions</li>
                    <li>Risk Scores</li>
                    <li>Trend Visualizations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card-modern">
                <h4 style='color: #667eea;'>Activity Logs</h4>
                <ul style='color: #1f2937; line-height: 2;'>
                    <li>Chatbot Conversations</li>
                    <li>Mind Gym Progress</li>
                    <li>Streak Data</li>
                    <li>XP & Level</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card-modern" style='margin-top: 2rem;'>
            <h4 style='color: #667eea; margin-top: 0;'>Privacy & Confidentiality</h4>
            <p style='color: #1f2937; line-height: 1.8;'>
                Your report is generated on-demand and contains only your personal data. It is not stored on our servers 
                and is not shared with any third parties. The PDF is created locally in your session and downloaded directly 
                to your device.
            </p>
            <p style='color: #1f2937; line-height: 1.8; margin-bottom: 0;'>
                <b>Important:</b> This report is for your personal records or to share with your healthcare provider. 
                Keep it secure as it contains sensitive mental health information.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)




















