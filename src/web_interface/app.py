"""
Emotify - Emotion-Driven Music Recommendation System
Main Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import tempfile

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.emotion_detection.text_emotion import TextEmotionDetector
    from src.emotion_detection.audio_emotion import AudioEmotionDetector
    from src.recommendation_engine.recommender import EmotionBasedRecommender
    from src.utils.conversational_bot import ConversationalBot
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are installed and accessible.")

# Configure Streamlit page settings
st.set_page_config(
    page_title="Emotify - AI Music Recommendation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
st.markdown("""
<style>
    /* Full screen layout */
    .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Content layout */
    .content-section {
        margin: 0 !important;
        padding: 2rem !important;
        border-radius: 0 !important;
    }
    
    /* Sidebar layout */
    [data-testid="stSidebar"] {
        padding-top: 0;
    }
    
    [data-testid="stSidebarContent"] {
        padding: 1rem;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    .row-widget.stButton {
        margin-bottom: 1rem;
    }
    
    /* Full width containers */
    [data-testid="stHorizontalBlock"] {
        width: 100%;
        gap: 1rem;
    }
    
    [data-testid="column"] {
        padding: 0.5rem !important;
    }
    
    /* Chart layout */
    .plotly-graph-div, .stPlotlyChart {
        width: 100% !important;
    }
    
    [data-testid="stPlotlyChart"] > div {
        width: 100% !important;
    }
    
    /* Hide footer */
    footer {
        display: none;
    }
    
    /* Responsive layout */
    @media (max-width: 1200px) {
        .main .block-container {
            padding: 0 0.5rem !important;
        }
        
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        
        [data-testid="stHorizontalBlock"] > div {
            width: 100%;
        }
    }
    
    @media (max-width: 768px) {
        .content-section {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'text_detector' not in st.session_state:
        st.session_state.text_detector = TextEmotionDetector()

    if 'audio_detector' not in st.session_state:
        st.session_state.audio_detector = AudioEmotionDetector()

    if 'recommender' not in st.session_state:
        # Construct the absolute path to the dataset
        base_path = os.path.dirname(__file__)
        dataset_path = os.path.join(base_path, '..', '..', 'data', 'datasets', 'spotify_data.csv')
        st.session_state.recommender = EmotionBasedRecommender(songs_data_path=dataset_path)

    if 'bot' not in st.session_state:
        st.session_state.bot = ConversationalBot(
            emotion_detector=st.session_state.text_detector,
            recommender=st.session_state.recommender
        )

    # NEW CHAT HISTORY
    if 'messages' not in st.session_state:
        # Start with the bot's greeting
        st.session_state.messages = [{"role": "assistant", "content": st.session_state.bot.initial_prompt}]

    # ... rest of initializations ...
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None

    if 'user_history' not in st.session_state:
        st.session_state.user_history = []

    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None

def get_bot_response(user_message):
    """Generate more dynamic and contextual bot responses."""
    import random
    
    # Add emotion detection
    emotion_result = st.session_state.text_detector.predict(user_message)
    
    # Update current emotion if confidence is high enough
    if emotion_result['confidence'] > 0.6:
        st.session_state.current_emotion = emotion_result

    # Dynamic greetings
    greetings = [
        "Hello! How's your musical journey going today?",
        "Hi there! Ready to discover some great music?",
        "Hey! Let's find the perfect soundtrack for your mood!",
        "Welcome! I'm here to help you find music that resonates with you!"
    ]

    # Dynamic recommendation responses
    recommendation_responses = [
        f"Based on your {emotion_result['emotion']} mood, I think you'll love these tracks! Want me to show you?",
        f"I've got some amazing {emotion_result['emotion']}-inspired songs that might speak to you right now.",
        f"Let me find you some music that matches your {emotion_result['emotion']} energy!",
        "I know just the right songs for this mood. Should I share my recommendations?"
    ]

    # Dynamic emotion responses
    emotion_responses = [
        f"I sense you're feeling {emotion_result['emotion']}. Music can be a great companion for this mood.",
        f"Your words suggest a {emotion_result['emotion']} state of mind. Let's find music that resonates with that.",
        f"I'm picking up on {emotion_result['emotion']} vibes. Would you like some matching music suggestions?",
        f"It sounds like you're {emotion_result['emotion']}. I can help you find the perfect soundtrack for this feeling."
    ]

    # Dynamic help responses
    help_responses = [
        "I use AI to understand your emotions and find music that matches your mood. Want to give it a try?",
        "I can analyze your emotions and suggest music that either matches or helps enhance your mood. What interests you?",
        "Think of me as your personal DJ who understands how you feel. Would you like some music suggestions?",
        "I'm here to connect your emotions with the perfect music. How can I help you today?"
    ]

    # Default varied responses
    default_responses = [
        f"I noticed a {emotion_result['emotion']} tone. Let me know if you'd like some music suggestions!",
        "Tell me more about what kind of music you're in the mood for.",
        "I'm here to help you find the perfect music. What kind of songs are you looking for?",
        "Would you like me to suggest some songs based on your current mood?"
    ]

    # Store conversation history in session state if not exists
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Add current message to history
    st.session_state.conversation_history.append(user_message.lower())

    # Check for repetitive user messages
    if len(st.session_state.conversation_history) >= 2:
        if st.session_state.conversation_history[-1] == st.session_state.conversation_history[-2]:
            return "I notice you're repeating yourself. Let me help you differently - would you like to try our emotion detection feature or get some fresh music recommendations?"

    # Generate contextual response
    if any(greeting in user_message.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
        response = random.choice(greetings)
    elif any(word in user_message.lower() for word in ['recommend', 'suggestion', 'music', 'song']):
        response = random.choice(recommendation_responses)
    elif any(word in user_message.lower() for word in ['emotion', 'feeling', 'mood', 'feel']):
        response = random.choice(emotion_responses)
    elif any(word in user_message.lower() for word in ['how', 'what', 'help', 'work']):
        response = random.choice(help_responses)
    else:
        response = random.choice(default_responses)

    # Add interactive follow-up based on context
    if 'recommendations' in st.session_state and len(st.session_state.recommendations) > 0:
        response += "\n\nBy the way, what did you think of the previous recommendations?"
    
    # Add call-to-action if no emotion detected yet
    if 'current_emotion' not in st.session_state or not st.session_state.current_emotion:
        response += "\n\nFeel free to share how you're feeling, and I'll find the perfect music for your mood!"

    return response

def add_chat_message(role, content):
    """Add a message to the chat history."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M")
    })

def display_emotion_visualization(emotion_data):
    """Display emotion detection results with visualization."""
    if not emotion_data:
        return

    # Create emotion visualization
    fig = go.Figure(data=go.Scatterpolar(
        r=[emotion_data.get('confidence', 0.5) * 100],
        theta=[emotion_data.get('emotion', 'calm').title()],
        fill='toself',
        name='Current Emotion'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Emotion Detection Result",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False,
        'responsive': True
    })

def display_song_recommendations(recommendations):
    """Display song recommendations in a nice format."""
    if not recommendations:
        st.warning("No recommendations found.")
        return

    st.subheader("🎵 Recommended Songs")

    for i, song in enumerate(recommendations[:10], 1):
        with st.container():
            col1, col2 = st.columns([1, 6])

            with col1:
                st.markdown(f"### #{i}")

            with col2:
                st.markdown(f"**{song.get('title', 'Unknown Title')}**")
                st.markdown(f"*by {song.get('artist', 'Unknown Artist')}*")
                
                # Get track ID from the song data - check both potential column names
                track_id = song.get('song_id', '') or song.get('track_id', '')
                print(f"Track ID for {song.get('title')}: {track_id}")
                
                # Clean up the track ID if it's in a Spotify URI format
                if track_id and ':' in str(track_id):
                    track_id = str(track_id).split(':')[-1]
                
                if track_id and str(track_id).strip():
                    # Direct Spotify track URL
                    spotify_url = f"https://open.spotify.com/track/{track_id}"
                    print(f"Using direct Spotify track URL: {spotify_url}")
                else:
                    # Fallback to search if no track ID is available
                    from urllib.parse import quote
                    search_query = quote(f"{song.get('title', '')} {song.get('artist', '')}")
                    spotify_url = f"https://open.spotify.com/search/{search_query}"
                    print(f"Using search fallback URL: {spotify_url}")
                
                # Create a custom button with CSS styling
                spotify_button = f"""
                <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
                    <div style="
                        background-color: #1DB954;
                        color: white;
                        padding: 10px 20px;
                        border-radius: 20px;
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        font-weight: bold;
                        margin: 10px 0;
                        cursor: pointer;
                        transition: background-color 0.3s;">
                        <img src="https://open.spotify.com/favicon.ico" style="width: 20px; height: 20px;"/>
                        Listen on Spotify
                    </div>
                </a>
                """
                st.markdown(spotify_button, unsafe_allow_html=True)

                if st.button(f"👍 Like", key=f"like_{track_id or i}"):
                    st.session_state.user_history.append(track_id or i)
                    st.success("Added to your preferences!")

def create_emotion_distribution_chart(recommender):
    """Create a chart showing emotion distribution in the dataset."""
    try:
        emotion_dist = recommender.songs_df['emotion'].value_counts()

        fig = px.pie(
            values=emotion_dist.values,
            names=emotion_dist.index,
            title="Emotion Distribution in Music Database",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False,
            'responsive': True
        })

    except Exception as e:
        st.error(f"Error creating emotion distribution chart: {e}")

def render_chat_interface():
    """Render the chat interface in the bottom right corner."""
    # Chat toggle button
    chat_button_html = f"""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
        <button
            onclick="window.chat_visible = !window.chat_visible; document.getElementById('chat-container').style.display = window.chat_visible ? 'block' : 'none';"
            style="background-color: #1e1e1e; color: white; border: none; border-radius: 50%; width: 45px; height: 45px; cursor: pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 20px;">💬</span>
        </button>
    </div>
    """
    st.markdown(chat_button_html, unsafe_allow_html=True)

    # Chat container
    chat_container_html = """
    <div id="chat-container" style="display: none; position: fixed; bottom: 75px; right: 20px; width: 280px; height: 350px; background: #1e1e1e; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); z-index: 1000; overflow: hidden; display: flex; flex-direction: column;">
        <div style="padding: 8px; background: #252526; color: white; font-size: 14px; font-weight: 500;">
            Emotify Assistant
        </div>
        <div id="chat-messages" style="flex: 1; overflow-y: auto; padding: 10px; background: #1e1e1e;">
            {messages}
        </div>
        <div style="padding: 8px; border-top: 1px solid #333;">
            <input type="text" id="chat-input" placeholder="Ask me anything..."
                style="width: 100%; padding: 6px 12px; border: 1px solid #333; border-radius: 15px; outline: none; background: #252526; color: white; font-size: 13px;"
                onkeypress="if(event.key === 'Enter') {{ sendMessage(); }}">
        </div>
    </div>
    """

    # Display chat messages
    messages_html = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            messages_html += f"""
            <div style="margin-bottom: 8px;">
                <div style="background: #333; padding: 6px 10px; border-radius: 12px; max-width: 80%; float: right; font-size: 13px; color: white;">
                    {msg["content"]}
                </div>
                <div style="clear: both;"></div>
            </div>
            """
        else:
            messages_html += f"""
            <div style="margin-bottom: 8px;">
                <div style="background: #2d2d2d; color: white; padding: 6px 10px; border-radius: 12px; max-width: 80%; font-size: 13px;">
                    {msg["content"]}
                </div>
            </div>
            """

    st.markdown(chat_container_html.format(messages=messages_html), unsafe_allow_html=True)

    # Add chat functionality
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            user_message = st.text_input("Message Emotify:", key="chat_input", label_visibility="collapsed")
        with col2:
            if st.button("Send", key="send_button"):
                if user_message:
                    add_chat_message("user", user_message)
                    bot_response = get_bot_response(user_message)
                    add_chat_message("assistant", bot_response)
                    st.session_state.chat_input = ""

def main():
    initialize_session_state()
    
    # Configure the layout to be more responsive
    st.markdown("""
        <style>
            /* Responsive layout adjustments */
            @media (max-width: 1200px) {
                .main .block-container {
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                }
                
                [data-testid="stHorizontalBlock"] {
                    flex-direction: column;
                }
                
                [data-testid="stHorizontalBlock"] > div {
                    width: 100%;
                }
            }
            
            /* Improved chart spacing */
            .element-container {
                margin-bottom: 1rem;
            }
            
            /* Better container spacing */
            .row-widget.stButton {
                margin-bottom: 1rem;
            }
            
            /* Fix chart overflow */
            [data-testid="stPlotlyChart"] > div {
                width: 100% !important;
            }
            
            /* Ensure tabs take full width */
            .stTabs [data-baseweb="tab-list"] button {
                flex-grow: 1;
            }
            
            /* Improve readability on smaller screens */
            @media (max-width: 768px) {
                h1 {
                    font-size: 2rem !important;
                }
                
                .content-section {
                    padding: 1rem;
                }
                
                [data-testid="stMetricValue"] {
                    font-size: 1.5rem !important;
                }
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Main content container with gradient background
    st.markdown("""
        <div class="content-section">
            <div style="text-align: center;">
                <h1>🎵 Emotify</h1>
                <p style="color: #666; font-size: 1.3em; font-weight: 300; margin-bottom: 0;">Discover Music That Moves You</p>
                <p style="color: #888; font-size: 1.1em; font-weight: 300;">AI-Powered Emotion-Based Recommendations</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with glassmorphism effect
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-content">
                <h2>🎛️ Control Panel</h2>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container():
           # st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            st.subheader("🎯 Detection Mode")
            mode = st.selectbox(
                "How should we detect your emotion?",
                ["Text Emotion", "Audio Analysis", "Manual Selection"],
                help="Choose how you want to input your emotional state"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
           # st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            st.subheader("🎭 Mood Settings")
            mood_regulation = st.selectbox(
                "Recommendation Strategy",
                ["match", "boost"],
                help="Match: Songs that align with your current mood\nBoost: Songs that can help improve your mood"
            )
            num_recs = st.slider(
                "Number of Songs",
                min_value=5,
                max_value=20,
                value=10,
                help="How many song recommendations would you like?"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["😊 Emotion Detection", "🎵 Recommendations", "📊 Analytics", "💬 ChatBot"])
    
    with tab1:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if mode == "Text Emotion":
                st.markdown("""
                    <div class="emotion-input-card">
                        <h3>💬 Text-Based Emotion Detection</h3>
                        <p>Tell us how you're feeling, and we'll analyze your emotion.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                user_text = st.text_area(
                    "How are you feeling? Describe your mood:",
                    height=150,
                    placeholder="I'm feeling really happy today and want to listen to some upbeat music!"
                )
                
                if st.button("Analyze Emotion", type="primary"):
                    if user_text.strip():
                        with st.spinner("Analyzing your emotion..."):
                            emotion_result = st.session_state.text_detector.predict(user_text)
                            st.session_state.current_emotion = emotion_result

                        # Display results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"""
                            <div class="emotion-card">
                                <h3>Detected Emotion</h3>
                                <h2>{emotion_result['emotion'].title()}</h2>
                                <p>Confidence: {emotion_result['confidence']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            display_emotion_visualization(emotion_result)
                    else:
                        st.warning("Please enter some text to analyze your emotion.")

            elif mode == "Audio Analysis":
                st.subheader("🎤 Audio-Based Emotion Detection")

                uploaded_audio = st.file_uploader(
                    "Upload an audio file to analyze emotion",
                    type=['mp3', 'wav', 'ogg'],
                    help="Upload a song or audio clip to detect its emotional characteristics"
                )

                if uploaded_audio is not None:
                    st.audio(uploaded_audio)

                    if st.button("Analyze Audio Emotion", type="primary"):
                        with st.spinner("Analyzing audio emotion..."):
                            # Save uploaded file to a temporary directory
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_audio.name}") as tmp:
                                tmp.write(uploaded_audio.getbuffer())
                                temp_path = tmp.name

                            try:
                                # Analyze the audio file from the secure temporary path
                                emotion_result = st.session_state.audio_detector.analyze_audio(temp_path)
                                st.session_state.current_emotion = emotion_result

                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

                                # Display results
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(f"""
                                    <div class="emotion-card">
                                        <h3>Audio Emotion</h3>
                                        <h2>{emotion_result['emotion'].title()}</h2>
                                        <p>Confidence: {emotion_result['confidence']:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with col2:
                                    # Display audio characteristics
                                    st.subheader("Audio Characteristics")
                                    metrics_col1, metrics_col2 = st.columns(2)

                                    with metrics_col1:
                                        st.metric("Valence", f"{emotion_result.get('valence', 0): .2f}")
                                        st.metric("Energy", f"{emotion_result.get('energy', 0): .2f}")

                                    with metrics_col2:
                                        st.metric("Arousal", f"{emotion_result.get('arousal', 0): .2f}")
                                        st.metric("Tempo", f"{emotion_result.get('tempo', 0): .0f} BPM")

                            except Exception as e:
                                st.error(f"Error analyzing audio: {str(e)}")
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

            else:  # Manual Selection
                st.subheader("🎯 Manual Emotion Selection")

                selected_emotion = st.selectbox(
                    "Select your current emotion:",
                    ["happy", "sad", "calm", "energetic"],
                    format_func=lambda x: x.title()
                )

                confidence = st.slider("How confident are you about this emotion?", 0.1, 1.0, 0.8)

                if st.button("Set Emotion", type="primary"):
                    st.session_state.current_emotion = {
                        'emotion': selected_emotion,
                        'confidence': confidence
                    }
                    st.success(f"Emotion set to: {selected_emotion.title()}")

    # Recommendations Tab
    with tab2:
        st.header("🎵 Music Recommendations")

        if st.session_state.current_emotion:
            current_emotion = st.session_state.current_emotion['emotion']

            st.info(f"Current emotion: **{current_emotion.title()}** (Confidence: {st.session_state.current_emotion['confidence']:.1%})")

            col1, col2 = st.columns([3, 1])

            with col2:
                if st.button("Get Recommendations", type="primary"):
                    with st.spinner("Finding perfect songs for you..."):
                        recommendations = st.session_state.recommender.hybrid_recommend(
                            user_emotion=current_emotion,
                            user_history=st.session_state.user_history,
                            num_recommendations=num_recs,
                            mood_regulation=mood_regulation
                        )

                        if recommendations:
                            display_song_recommendations(recommendations)
                        else:
                            st.warning("No recommendations found. Try a different emotion or mood regulation strategy.")

            with col1:
                st.markdown(f"""
                **Recommendation Strategy:** {mood_regulation.title()}

                {'Songs that match your current mood' if mood_regulation == 'match' else 'Songs to improve your mood'}
                """)

        else:
            st.warning("Please detect your emotion first using the 'Emotion Detection' tab.")

            # Show sample recommendations
            st.subheader("🎵 Sample Recommendations")
            sample_recs = st.session_state.recommender.recommend_by_emotion("happy", 5)

            st.markdown("*Here are some happy songs to get you started:*")
            for i, song in enumerate(sample_recs, 1):
                st.markdown(f"{i}. **{song['title']}** by *{song['artist']}*")

    # Analytics Tab
    with tab3:
        st.header("📊 Music Database Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Dataset overview
            st.subheader("Dataset Overview")
            total_songs = len(st.session_state.recommender.songs_df)
            unique_artists = st.session_state.recommender.songs_df['artist'].nunique()

            st.metric("Total Songs", total_songs)
            st.metric("Unique Artists", unique_artists)
            st.metric("Emotion Categories", 4)

        with col2:
            # Emotion distribution
            create_emotion_distribution_chart(st.session_state.recommender)

        # Feature analysis
        st.subheader("Audio Feature Analysis")

        feature_cols = ['valence', 'arousal', 'tempo', 'energy']
        if all(col in st.session_state.recommender.songs_df.columns for col in feature_cols):
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Valence Distribution', 'Aroual Distribution',
                              'Tempo Distribution', 'Energy Distribution')
            )

            # Valence
            fig.add_trace(
                go.Histogram(x=st.session_state.recommender.songs_df['valence'], name='Valence'),
                row=1, col=1
            )

            # Arousal
            fig.add_trace(
                go.Histogram(x=st.session_state.recommender.songs_df['arousal'], name='Arousal'),
                row=1, col=2
            )

            # Tempo
            fig.add_trace(
                go.Histogram(x=st.session_state.recommender.songs_df['tempo'], name='Tempo'),
                row=2, col=1
            )

            # Energy
            fig.add_trace(
                go.Histogram(x=st.session_state.recommender.songs_df['energy'], name='Energy'),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=False, title_text="Audio Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Audio feature data (valence, arousal, tempo, energy) not available in the dataset for detailed analysis.")
    with tab4:
        st.header("💬 Talk to Emotify AI")
        st.markdown(
            "Use the chat bot to ask for music or describe your mood. "
            "The bot can auto-detect your emotion from your text!"
        )

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input logic
        if prompt := st.chat_input("Start a conversation about music..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get bot response and potential new emotion
            with st.spinner("Emotify is thinking..."):
                
                # *** KEY CHANGE: CALL THE DYNAMIC get_bot_response FUNCTION ***
                response = get_bot_response(prompt)
                
                # The get_bot_response function already updates st.session_state.current_emotion 
                # and generates a dynamic response based on emotion detection and keywords.

                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(response)

            # Force a rerun to display the latest messages instantly
            st.rerun()
    # Footer
    st.markdown("---")
    st.markdown(
        "🎵 **Emotify** - Made with ❤️ using Streamlit, scikit-learn, and librosa | "
        "Emotion-driven music recommendations powered by AI"
    )

if __name__ == "__main__":
    main()
