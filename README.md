# EMOTIFY: Emotion-Driven Music Recommendation System Using Multimodal AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎵 Overview

Emotify is an advanced emotion-driven music recommendation system that uses multimodal AI to detect user emotions and recommend music that aligns with or regulates the listener's mood. The system analyzes emotions through multiple modalities including text sentiment, speech tone, lyrics sentiment, and audio features.

## 🎯 Key Features

### Core Features (MVP)
- **Emotion Detection**: From text input (chat/sentiment analysis)
- **Lyrics Sentiment Analysis**: Mood classification from song lyrics
- **Audio Feature Extraction**: Mood classification using audio characteristics
- **Content-Based Recommendation**: Music suggestions based on emotional state

### Advanced Features
- **Hybrid Recommendation**: Content-based + collaborative filtering
- **Real-time Adaptability**: Dynamic mood change detection
- **Mood Regulation Modes**: Match mood or boost mood options
- **Interactive Web Interface**: Streamlit-based user interface

### Future-Ready Features
- **Reinforcement Learning**: Adaptive recommendations
- **Conversational Interface**: Chatbot integration
- **Mental Health Integration**: Mood regulation assistance
- **Cultural Personalization**: Location and culture-aware recommendations

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   User Input    │    │  Emotion         │    │  Recommendation     │
│  - Text         │───▶│  Detection       │───▶│  Engine             │
│  - Speech       │    │  - NLP Models    │    │  - Content-Based    │
│  - Audio        │    │  - Audio Models  │    │  - Collaborative    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                ▲                         │
                                │                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Song Database │    │  Feature         │    │  Music Playlist     │
│  - Lyrics       │───▶│  Extraction      │    │  - Emotion-matched  │
│  - Audio Files  │    │  - Text Features │    │  - Mood-regulated   │
│  - Metadata     │    │  - Audio Features│    │  - Personalized     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- macOS 10.14+ / Linux / Windows 10+
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   cd EMOTIFY_Multimodal_Music_Recommendation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv emotify_env
   source emotify_env/bin/activate  # On macOS/Linux
   # emotify_env\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements/requirements.txt
   ```

4. **Download required datasets** (Optional - demo data included)
   ```bash
   python scripts/download_datasets.py
   ```

5. **Run the application**
   ```bash
   streamlit run src/web_interface/app.py
   ```

## 📊 Datasets Used

- **DEAM Dataset**: Dynamic Emotion in Music (songs with valence-arousal labels)
- **MoodyLyrics Dataset**: Songs labeled with emotional moods
- **MSD + MusiXmatch**: Large dataset of lyrics and metadata
- **EmoLex**: Word-emotion mapping lexicon
- **Last.fm HetRec 2011**: User listening history and preferences

## 🧠 Emotion Categories

### Basic Classification (MVP)
- **Happy**: Upbeat, joyful, celebratory music
- **Sad**: Melancholic, reflective, somber music
- **Calm/Relaxed**: Peaceful, meditative, ambient music
- **Energetic**: High-tempo, motivational, workout music

### Advanced Classification
- **Valence-Arousal 2D Model**: Continuous emotional space
  - Valence: Positive ↔ Negative
  - Arousal: High Energy ↔ Low Energy

## 🛠️ Technology Stack

- **Languages**: Python 3.8+
- **NLP**: NLTK, VADER, Transformers (BERT)
- **Audio Processing**: librosa, pyAudioAnalysis
- **Machine Learning**: scikit-learn, TensorFlow, PyTorch
- **Recommendation**: Surprise, LightFM
- **Web Interface**: Streamlit
- **APIs**: Spotify API, Last.fm API

## 📁 Project Structure

```
EMOTIFY_Multimodal_Music_Recommendation/
├── src/
│   ├── emotion_detection/      # Emotion detection models
│   ├── recommendation_engine/  # Recommendation algorithms
│   ├── data_processing/       # Data preprocessing utilities
│   ├── models/               # Model definitions
│   └── web_interface/        # Streamlit web app
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Preprocessed data
│   └── datasets/             # Sample datasets
├── models/
│   └── trained/              # Trained model files
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
```

## 🎮 Usage Examples

### 1. Text-Based Emotion Detection
```python
from src.emotion_detection.text_emotion import TextEmotionDetector

detector = TextEmotionDetector()
emotion = detector.predict("I'm feeling really down today")
# Output: {'emotion': 'sad', 'confidence': 0.87}
```

### 2. Audio Feature Analysis
```python
from src.emotion_detection.audio_emotion import AudioEmotionDetector

audio_detector = AudioEmotionDetector()
emotion = audio_detector.analyze_audio("path/to/song.mp3")
# Output: {'valence': 0.3, 'arousal': 0.7, 'emotion': 'energetic'}
```

### 3. Get Recommendations
```python
from src.recommendation_engine.recommender import EmotionBasedRecommender

recommender = EmotionBasedRecommender()
playlist = recommender.recommend(
    user_emotion="happy",
    num_songs=10,
    mood_regulation="match"  # or "boost"
)
```

## 🎯 Development Phases

### Phase 1: MVP (4 Weeks) ✅
- [x] Data collection and preprocessing
- [x] Basic emotion detection models
- [x] Content-based recommendation engine
- [x] Simple web interface

### Phase 2: Advanced Features (4 Weeks)
- [ ] Collaborative filtering integration
- [ ] Hybrid recommendation engine
- [ ] Real-time adaptability
- [ ] Enhanced UI/UX

### Phase 3: Future Enhancements
- [ ] Speech emotion detection
- [ ] Reinforcement learning
- [ ] Conversational chatbot
- [ ] Mental health integration

## 📈 Evaluation Metrics

- **Emotion Detection**: Accuracy, F1-score, Confusion Matrix
- **Recommendation Quality**: Precision@k, Recall@k, NDCG
- **User Satisfaction**: Feedback loop integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For support, email your-email@example.com or create an issue in this repository.

## 🎵 Acknowledgments

- DEAM Dataset contributors
- MusiXmatch for lyrics data
- Spotify for API access
- Open source music emotion research community

---

**Made with ❤️ for better music experiences through AI**
