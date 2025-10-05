# Emotify Documentation

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup Steps

1. **Clone or extract the project**
   ```bash
   cd EMOTIFY_Multimodal_Music_Recommendation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv emotify_env

   # On macOS/Linux:
   source emotify_env/bin/activate

   # On Windows:
   emotify_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements/requirements.txt
   ```

4. **Run setup script**
   ```bash
   python scripts/setup.py
   ```

5. **Configure API keys** (optional)
   - Copy `.env.template` to `.env`
   - Add your Spotify and Last.fm API keys

## Quick Start

### Running the Web Application
```bash
# Option 1: Use the run script
python scripts/run.py

# Option 2: Direct streamlit command
streamlit run src/web_interface/app.py
```

### Using the Python API

#### Text Emotion Detection
```python
from src.emotion_detection.text_emotion import TextEmotionDetector

detector = TextEmotionDetector()
result = detector.predict("I'm feeling really happy today!")

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### Audio Emotion Analysis
```python
from src.emotion_detection.audio_emotion import AudioEmotionDetector

audio_detector = AudioEmotionDetector()
result = audio_detector.analyze_audio("path/to/song.mp3")

print(f"Emotion: {result['emotion']}")
print(f"Valence: {result['valence']:.2f}")
print(f"Arousal: {result['arousal']:.2f}")
```

#### Music Recommendations
```python
from src.recommendation_engine.recommender import EmotionBasedRecommender

recommender = EmotionBasedRecommender()
recommendations = recommender.recommend_by_emotion(
    target_emotion="happy",
    num_recommendations=10,
    mood_regulation="match"
)

for song in recommendations:
    print(f"{song['title']} by {song['artist']}")
```

## API Reference

### TextEmotionDetector

#### Methods
- `predict(text: str) -> Dict`: Analyze emotion from text
- `predict_batch(texts: List[str]) -> List[Dict]`: Batch prediction
- `analyze_lyrics(lyrics: str) -> Dict`: Specialized lyrics analysis

#### Parameters
- `model_type`: 'vader', 'textblob', or 'keyword'

### AudioEmotionDetector

#### Methods
- `analyze_audio(audio_path: str) -> Dict`: Analyze audio file
- `extract_features(audio_path: str) -> np.ndarray`: Extract audio features
- `analyze_batch(audio_paths: List[str]) -> List[Dict]`: Batch analysis

#### Parameters
- `sample_rate`: Audio sample rate (default: 22050)

### EmotionBasedRecommender

#### Methods
- `recommend_by_emotion(emotion, num_recs, mood_regulation)`: Emotion-based recommendations
- `recommend_similar_songs(song_id, num_recs)`: Similarity-based recommendations
- `hybrid_recommend(user_emotion, user_history, num_recs, mood_regulation)`: Hybrid approach

#### Parameters
- `mood_regulation`: 'match' or 'boost'
- `num_recommendations`: Number of songs to recommend

## Datasets

### Built-in Sample Dataset
- 50 sample songs with emotional labels
- Includes audio features (valence, arousal, tempo, energy)
- Located at `data/datasets/sample_songs.csv`

### Supported External Datasets
- **DEAM Dataset**: Dynamic Emotion in Music
- **MoodyLyrics**: Kaggle dataset with mood-labeled lyrics
- **Last.fm HetRec 2011**: User listening history

### Custom Dataset Format
```csv
song_id,title,artist,emotion,valence,arousal,tempo,energy,danceability,lyrics
1,Happy Song,Artist Name,happy,0.8,0.6,120,0.7,0.8,"happy lyrics here"
```

## Model Architecture

### Emotion Classification
```
Text Input → NLP Processing → Sentiment Analysis → Emotion Categories
                ↓
        [VADER/TextBlob/BERT]
                ↓
        [happy, sad, calm, energetic]
```

### Audio Feature Extraction
```
Audio File → librosa → Feature Extraction → Emotion Classification
                         ↓
            [MFCCs, Tempo, Spectral Features]
                         ↓
              [Rule-based Classification]
```

### Recommendation Pipeline
```
User Emotion → Content Filtering → Similarity Computation → Ranking
     ↓              ↓                      ↓                  ↓
Emotion Input → Song Database → Feature Matching → Top-K Songs
```

## File Structure

```
EMOTIFY_Multimodal_Music_Recommendation/
├── src/
│   ├── emotion_detection/         # Emotion detection modules
│   │   ├── text_emotion.py       # Text-based emotion analysis
│   │   └── audio_emotion.py      # Audio-based emotion analysis
│   ├── recommendation_engine/     # Recommendation algorithms
│   │   └── recommender.py        # Main recommendation engine
│   ├── data_processing/          # Data utilities
│   │   └── data_utils.py         # Dataset processing tools
│   └── web_interface/            # Streamlit web application
│       └── app.py                # Main web app
├── data/                         # Data directories
│   ├── datasets/                 # Sample and external datasets
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
├── configs/                      # Configuration files
└── requirements/                 # Dependencies
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions
- Write unit tests for new features

### Adding New Features
1. **New Emotion Categories**: Modify emotion mappings in relevant modules
2. **New Audio Features**: Extend `AudioEmotionDetector.extract_features()`
3. **New Recommendation Algorithms**: Add methods to `EmotionBasedRecommender`
4. **New Data Sources**: Extend `DataProcessor` with new loading methods

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the project directory and virtual environment is activated
cd EMOTIFY_Multimodal_Music_Recommendation
source emotify_env/bin/activate  # macOS/Linux
pip install -r requirements/requirements.txt
```

**Audio Processing Errors**
```bash
# Install additional audio libraries
pip install pyaudio soundfile
# On macOS: brew install portaudio
# On Ubuntu: sudo apt-get install portaudio19-dev
```

**NLTK Data Missing**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
```

**Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear
# Run with specific port
streamlit run src/web_interface/app.py --server.port 8502
```

### Performance Optimization
- Use batch processing for multiple files
- Cache model predictions when possible
- Consider using GPU for deep learning models
- Implement pagination for large datasets

## License
This project is licensed under the MIT License.

## Acknowledgments
- DEAM Dataset contributors
- MusiXmatch for lyrics data
- Spotify for audio features API
- Open source music emotion research community
