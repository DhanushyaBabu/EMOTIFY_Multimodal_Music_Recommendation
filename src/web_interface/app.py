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

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.emotion_detection.text_emotion import TextEmotionDetector
    from src.emotion_detection.audio_emotion import AudioEmotionDetector
    from src.recommendation_engine.recommender import EmotionBasedRecommender
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are installed and accessible.")

# Page configuration
st.set_page_config(
    page_title="Emotify - AI Music Recommendation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .song-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1DB954;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
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

    if 'user_history' not in st.session_state:
        st.session_state.user_history = []

    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None

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
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

def display_song_recommendations(recommendations):
    """Display song recommendations in a nice format."""
    if not recommendations:
        st.warning("No recommendations found.")
        return

    st.subheader("🎵 Recommended Songs")

    for i, song in enumerate(recommendations[:10], 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.markdown(f"**#{i}**")

            with col2:
                st.markdown(f"**{song.get('title', 'Unknown Title')}**")
                st.markdown(f"*by {song.get('artist', 'Unknown Artist')}*")
                st.markdown(f"Emotion: {song.get('emotion', 'Unknown').title()}")

            with col3:
                if st.button(f"👍", key=f"like_{song.get('song_id', i)}"):
                    st.session_state.user_history.append(song.get('song_id', i))
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
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating emotion distribution chart: {e}")

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🎵 Emotify</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Emotion-Driven Music Recommendation System Using Multimodal AI</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # Mode selection
        mode = st.selectbox(
            "Select Detection Mode",
            ["Text Emotion", "Audio Analysis", "Manual Selection"]
        )

        # Mood regulation strategy
        mood_regulation = st.selectbox(
            "Mood Regulation Strategy",
            ["match", "boost"],
            help="Match: Songs that match current mood\nBoost: Songs to improve mood"
        )

        # Number of recommendations
        num_recs = st.slider("Number of Recommendations", 5, 20, 10)

        st.markdown("---")

        # User history
        if st.session_state.user_history:
            st.subheader("👤 Your Preferences")
            st.write(f"Liked songs: {len(st.session_state.user_history)}")
            if st.button("Clear History"):
                st.session_state.user_history = []
                st.rerun()

    # Main content area
    tabs = st.tabs(["🎯 Emotion Detection", "🎵 Recommendations", "📊 Analytics"])

    # Emotion Detection Tab
    with tabs[0]:
        st.header("Detect Your Current Emotion")

        if mode == "Text Emotion":
            st.subheader("💬 Text-Based Emotion Detection")

            user_text = st.text_area(
                "How are you feeling? Describe your mood:",
                placeholder="I'm feeling really happy today and want to listen to some upbeat music!",
                height=100
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
                        # Save uploaded file temporarily
                        temp_path = f"temp_audio_{uploaded_audio.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_audio.getbuffer())

                        try:
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
    with tabs[1]:
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
    with tabs[2]:
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
                subplot_titles=('Valence Distribution', 'Arousal Distribution',
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

    # Footer
    st.markdown("---")
    st.markdown(
        "🎵 **Emotify** - Made with ❤️ using Streamlit, scikit-learn, and librosa | "
        "Emotion-driven music recommendations powered by AI"
    )

if __name__ == "__main__":
    main()
