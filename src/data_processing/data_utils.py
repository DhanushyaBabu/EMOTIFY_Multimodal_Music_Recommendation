"""
Data processing utilities for Emotify.
Handles dataset loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import json
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    """
    Data processing utilities for music emotion datasets.
    """

    def __init__(self):
        """Initialize the data processor."""
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_moody_lyrics_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess MoodyLyrics dataset.

        Args:
            file_path: Path to the dataset CSV file

        Returns:
            Preprocessed DataFrame
        """
        try:
            df = pd.read_csv(file_path)

            # Standard preprocessing
            df = self._clean_text_data(df)
            df = self._normalize_emotions(df)

            return df

        except Exception as e:
            print(f"Error loading MoodyLyrics dataset: {str(e)}")
            return pd.DataFrame()

    def load_deam_dataset(self, audio_features_path: str, annotations_path: str) -> pd.DataFrame:
        """
        Load and merge DEAM dataset components.

        Args:
            audio_features_path: Path to audio features CSV
            annotations_path: Path to emotion annotations CSV

        Returns:
            Merged DataFrame with audio features and emotion labels
        """
        try:
            # Load audio features
            audio_df = pd.read_csv(audio_features_path)

            # Load emotion annotations
            annotations_df = pd.read_csv(annotations_path)

            # Merge datasets
            merged_df = pd.merge(audio_df, annotations_df, on='song_id', how='inner')

            # Convert valence-arousal to discrete emotions
            merged_df = self._valence_arousal_to_emotions(merged_df)

            return merged_df

        except Exception as e:
            print(f"Error loading DEAM dataset: {str(e)}")
            return pd.DataFrame()

    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text data (lyrics, titles, etc.)."""
        text_columns = ['lyrics', 'title', 'artist']

        for col in text_columns:
            if col in df.columns:
                # Remove extra whitespace
                df[col] = df[col].astype(str).str.strip()

                # Remove special characters (keep basic punctuation)
                df[col] = df[col].str.replace(r'[^\w\s.,!?-]', '', regex=True)

                # Convert to lowercase for consistency
                df[col] = df[col].str.lower()

        return df

    def _normalize_emotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize emotion labels to standard categories."""
        if 'emotion' not in df.columns:
            return df

        # Emotion mapping dictionary
        emotion_mapping = {
            # Happy variants
            'happy': 'happy',
            'joy': 'happy',
            'excited': 'happy',
            'cheerful': 'happy',
            'upbeat': 'happy',
            'positive': 'happy',
            'euphoric': 'happy',

            # Sad variants
            'sad': 'sad',
            'depression': 'sad',
            'melancholy': 'sad',
            'down': 'sad',
            'blue': 'sad',
            'negative': 'sad',
            'gloomy': 'sad',
            'sorrowful': 'sad',

            # Calm variants
            'calm': 'calm',
            'relaxed': 'calm',
            'peaceful': 'calm',
            'serene': 'calm',
            'tranquil': 'calm',
            'mellow': 'calm',
            'chill': 'calm',
            'soothing': 'calm',

            # Energetic variants
            'energetic': 'energetic',
            'energy': 'energetic',
            'pump': 'energetic',
            'intense': 'energetic',
            'vigorous': 'energetic',
            'dynamic': 'energetic',
            'powerful': 'energetic',
            'aggressive': 'energetic'
        }

        # Apply mapping
        df['emotion'] = df['emotion'].str.lower().map(emotion_mapping)

        # Fill unmapped emotions with 'calm' as default
        df['emotion'] = df['emotion'].fillna('calm')

        return df

    def _valence_arousal_to_emotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert valence-arousal values to discrete emotion categories.

        Valence: positive (high) vs negative (low)
        Arousal: high energy vs low energy
        """
        if 'valence' not in df.columns or 'arousal' not in df.columns:
            return df

        def classify_emotion(valence, arousal):
            # Normalize values to 0-1 range if needed
            if valence > 1:
                valence = valence / 9.0  # Assuming 1-9 scale
            if arousal > 1:
                arousal = arousal / 9.0

            # Quadrant-based classification
            if valence >= 0.5 and arousal >= 0.5:
                return 'happy'  # High valence, high arousal
            elif valence < 0.5 and arousal >= 0.5:
                return 'energetic'  # Low valence, high arousal (aggressive/intense)
            elif valence >= 0.5 and arousal < 0.5:
                return 'calm'  # High valence, low arousal (peaceful)
            else:
                return 'sad'  # Low valence, low arousal

        df['emotion'] = df.apply(lambda row: classify_emotion(row['valence'], row['arousal']), axis=1)

        return df

    def create_sample_dataset(self, num_songs: int = 1000) -> pd.DataFrame:
        """
        Create a sample dataset for development and testing.

        Args:
            num_songs: Number of songs to generate

        Returns:
            Sample DataFrame with realistic music data
        """
        np.random.seed(42)  # For reproducibility

        # Generate basic song information
        titles = self._generate_song_titles(num_songs)
        artists = self._generate_artist_names(num_songs)

        # Generate emotions with realistic distribution
        emotions = np.random.choice(
            ['happy', 'sad', 'calm', 'energetic'],
            size=num_songs,
            p=[0.3, 0.2, 0.25, 0.25]  # Slightly more happy songs
        )

        # Generate audio features correlated with emotions
        audio_features = self._generate_audio_features(emotions)

        # Generate lyrics based on emotions
        lyrics = self._generate_lyrics(emotions)

        # Create DataFrame
        df = pd.DataFrame({
            'song_id': range(1, num_songs + 1),
            'title': titles,
            'artist': artists,
            'emotion': emotions,
            'valence': audio_features['valence'],
            'arousal': audio_features['arousal'],
            'tempo': audio_features['tempo'],
            'energy': audio_features['energy'],
            'danceability': audio_features['danceability'],
            'popularity': np.random.uniform(0.1, 1.0, num_songs),
            'lyrics': lyrics
        })

        return df

    def _generate_song_titles(self, num_songs: int) -> List[str]:
        """Generate realistic song titles."""
        title_patterns = {
            'happy': ['Sunshine {}', 'Happy {}', 'Feel Good {}', 'Bright {}', 'Joyful {}'],
            'sad': ['Lonely {}', 'Broken {}', 'Blue {}', 'Tears of {}', 'Empty {}'],
            'calm': ['Peaceful {}', 'Quiet {}', 'Serene {}', 'Gentle {}', 'Soft {}'],
            'energetic': ['Power {}', 'Rush {}', 'High {}', 'Intense {}', 'Dynamic {}']
        }

        words = ['Heart', 'Dreams', 'Night', 'Day', 'Love', 'Time', 'Life', 'Soul', 
                'Mind', 'World', 'Sky', 'Ocean', 'Fire', 'Light', 'Shadow', 'Hope']

        titles = []
        for _ in range(num_songs):
            emotion = np.random.choice(list(title_patterns.keys()))
            pattern = np.random.choice(title_patterns[emotion])
            word = np.random.choice(words)
            titles.append(pattern.format(word))

        return titles

    def _generate_artist_names(self, num_songs: int) -> List[str]:
        """Generate artist names."""
        first_names = ['Alex', 'Jordan', 'Taylor', 'Casey', 'Morgan', 'River', 'Sage', 'Phoenix']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
        band_words = ['Sound', 'Music', 'Band', 'Collective', 'Group', 'Artists', 'Ensemble']

        artists = []
        for _ in range(num_songs):
            if np.random.random() < 0.6:  # 60% individual artists
                first = np.random.choice(first_names)
                last = np.random.choice(last_names)
                artists.append(f"{first} {last}")
            else:  # 40% bands
                first = np.random.choice(first_names)
                band_word = np.random.choice(band_words)
                artists.append(f"{first} {band_word}")

        return artists

    def _generate_audio_features(self, emotions: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate audio features correlated with emotions."""
        num_songs = len(emotions)

        # Base features
        valence = np.zeros(num_songs)
        arousal = np.zeros(num_songs)
        tempo = np.zeros(num_songs)
        energy = np.zeros(num_songs)
        danceability = np.zeros(num_songs)

        for i, emotion in enumerate(emotions):
            if emotion == 'happy':
                valence[i] = np.random.normal(0.7, 0.15)
                arousal[i] = np.random.normal(0.6, 0.15)
                tempo[i] = np.random.normal(120, 20)
                energy[i] = np.random.normal(0.7, 0.15)
                danceability[i] = np.random.normal(0.8, 0.1)
            elif emotion == 'sad':
                valence[i] = np.random.normal(0.3, 0.15)
                arousal[i] = np.random.normal(0.4, 0.15)
                tempo[i] = np.random.normal(80, 15)
                energy[i] = np.random.normal(0.3, 0.15)
                danceability[i] = np.random.normal(0.3, 0.15)
            elif emotion == 'calm':
                valence[i] = np.random.normal(0.5, 0.1)
                arousal[i] = np.random.normal(0.3, 0.1)
                tempo[i] = np.random.normal(70, 10)
                energy[i] = np.random.normal(0.4, 0.1)
                danceability[i] = np.random.normal(0.4, 0.15)
            elif emotion == 'energetic':
                valence[i] = np.random.normal(0.6, 0.15)
                arousal[i] = np.random.normal(0.8, 0.1)
                tempo[i] = np.random.normal(140, 25)
                energy[i] = np.random.normal(0.9, 0.05)
                danceability[i] = np.random.normal(0.9, 0.05)

        # Clip values to valid ranges
        valence = np.clip(valence, 0, 1)
        arousal = np.clip(arousal, 0, 1)
        tempo = np.clip(tempo, 40, 200)
        energy = np.clip(energy, 0, 1)
        danceability = np.clip(danceability, 0, 1)

        return {
            'valence': valence,
            'arousal': arousal,
            'tempo': tempo,
            'energy': energy,
            'danceability': danceability
        }

    def _generate_lyrics(self, emotions: np.ndarray) -> List[str]:
        """Generate sample lyrics based on emotions."""
        lyrics_templates = {
            'happy': [
                "sunshine brings joy to my heart every day",
                "dancing through life with a smile on my face",
                "feeling good vibes all around me now",
                "happiness flows like a river of gold"
            ],
            'sad': [
                "lonely nights and empty dreams fill my soul",
                "tears fall like rain on my broken heart",
                "memories fade but the pain remains",
                "walking alone through the shadows of time"
            ],
            'calm': [
                "peaceful waters wash my worries away",
                "gentle breeze carries me to serenity",
                "quiet moments bring clarity to mind",
                "stillness speaks louder than words"
            ],
            'energetic': [
                "power flows through every beat of my heart",
                "adrenaline rushes through my veins",
                "unstoppable force driving me forward",
                "high energy ignites my inner fire"
            ]
        }

        lyrics = []
        for emotion in emotions:
            template = np.random.choice(lyrics_templates[emotion])
            lyrics.append(template)

        return lyrics

    def save_dataset(self, df: pd.DataFrame, file_path: str):
        """Save dataset to CSV file."""
        try:
            df.to_csv(file_path, index=False)
            print(f"Dataset saved to {file_path}")
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive statistics about the dataset."""
        stats = {
            'total_songs': len(df),
            'unique_artists': df['artist'].nunique() if 'artist' in df.columns else 0,
            'emotion_distribution': df['emotion'].value_counts().to_dict() if 'emotion' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'audio_feature_stats': {}
        }

        # Audio feature statistics
        audio_features = ['valence', 'arousal', 'tempo', 'energy', 'danceability']
        for feature in audio_features:
            if feature in df.columns:
                stats['audio_feature_stats'][feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max()
                }

        return stats

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()

    print("Creating sample dataset...")
    sample_df = processor.create_sample_dataset(100)

    print("Dataset created successfully!")
    print(f"Shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")

    # Get statistics
    stats = processor.get_dataset_statistics(sample_df)
    print("\nDataset Statistics:")
    print(f"Total songs: {stats['total_songs']}")
    print(f"Unique artists: {stats['unique_artists']}")
    print(f"Emotion distribution: {stats['emotion_distribution']}")
