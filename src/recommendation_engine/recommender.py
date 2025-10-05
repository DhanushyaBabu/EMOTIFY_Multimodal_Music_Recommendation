"""
Music recommendation engine for Emotify.
Implements content-based, collaborative, and hybrid recommendation algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import joblib
import os
import sys

# Add root directory to path to import DatasetManager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.datasets import DatasetManager


class EmotionBasedRecommender:
    """
    Music recommendation system based on emotional state and user preferences.
    """

    def __init__(self, songs_data_path: Optional[str] = None):
        """
        Initialize the recommendation engine.

        Args:
            songs_data_path: Path to songs dataset CSV file
        """
        self.dm = DatasetManager()
        self.songs_df = self.dm.load_dataset(songs_data_path) if songs_data_path and os.path.exists(songs_data_path) else self.dm.create_sample_spotify_dataset()
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()

        # Feature matrices for similarity computation
        self.content_features = None
        self.audio_features = None
        self.lyrical_features = None

        self._prepare_features()

    def _create_sample_dataset(self):
        """Create a sample dataset for demonstration."""
        self.songs_df = self.dm.create_sample_spotify_dataset(50)
        self._prepare_features()

    def load_dataset(self, file_path: str):
        """Load songs dataset from CSV file."""
        try:
            self.songs_df = self.dm.load_dataset(file_path)
            self._prepare_features()
            print(f"Loaded {len(self.songs_df)} songs from {file_path}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            self._create_sample_dataset()

    def _prepare_features(self):
        """Prepare feature matrices for similarity computation."""
        if self.songs_df is None:
            return

        # Rename columns to match expected names
        self.songs_df = self.songs_df.rename(columns={'track_name': 'title', 'artists': 'artist', 'track_id': 'song_id'})

        # Prepare audio features
        audio_cols = ['valence', 'arousal', 'tempo', 'energy', 'danceability']
        available_audio_cols = [col for col in audio_cols if col in self.songs_df.columns]

        if available_audio_cols:
            self.audio_features = self.scaler.fit_transform(
                self.songs_df[available_audio_cols].fillna(0)
            )

        # Prepare lyrical features
        if 'lyrics' in self.songs_df.columns:
            lyrics_text = self.songs_df['lyrics'].fillna('').astype(str)
            self.lyrical_features = self.tfidf_vectorizer.fit_transform(lyrics_text)

        # Combine content features
        self._create_content_features()

    def _create_content_features(self):
        """Create combined content feature matrix."""
        features = []

        # Add audio features if available
        if self.audio_features is not None:
            features.append(self.audio_features)

        # Add emotion one-hot encoding
        if 'emotion' in self.songs_df.columns:
            emotions = pd.get_dummies(self.songs_df['emotion'])
            features.append(emotions.values)

        # Combine all features
        if features:
            self.content_features = np.hstack(features)
        else:
            # Fallback to basic features
            self.content_features = np.random.random((len(self.songs_df), 10))

    def recommend_by_emotion(self, 
                           target_emotion: str, 
                           num_recommendations: int = 10,
                           mood_regulation: str = "match") -> List[Dict]:
        """
        Recommend songs based on target emotion.

        Args:
            target_emotion: Target emotion ('happy', 'sad', 'calm', 'energetic')
            num_recommendations: Number of songs to recommend
            mood_regulation: 'match' to match mood or 'boost' to improve mood

        Returns:
            List of recommended songs with metadata
        """
        if self.songs_df is None:
            return []

        # Ensure 'song_id' column exists
        if 'song_id' not in self.songs_df.columns:
            self.songs_df['song_id'] = self.songs_df.index

        # Filter songs by emotion
        if mood_regulation == "match":
            # Recommend songs that match the current emotion
            filtered_songs = self.songs_df[self.songs_df['emotion'] == target_emotion]
        elif mood_regulation == "boost":
            # Recommend songs to improve mood
            mood_boost_map = {
                'sad': ['happy', 'calm'],
                'happy': ['happy', 'energetic'],
                'calm': ['happy', 'calm'],
                'energetic': ['energetic', 'happy']
            }
            target_emotions = mood_boost_map.get(target_emotion, [target_emotion])
            filtered_songs = self.songs_df[self.songs_df['emotion'].isin(target_emotions)]
        else:
            filtered_songs = self.songs_df

        # Sort by popularity or other criteria
        if 'popularity' in filtered_songs.columns:
            filtered_songs = filtered_songs.sort_values('popularity', ascending=False)

        # Get top recommendations
        recommendations = filtered_songs.head(num_recommendations)

        return recommendations.to_dict('records')

    def recommend_similar_songs(self, 
                              song_id: int, 
                              num_recommendations: int = 10) -> List[Dict]:
        """
        Recommend songs similar to a given song.

        Args:
            song_id: ID of the reference song
            num_recommendations: Number of recommendations

        Returns:
            List of similar songs
        """
        if self.songs_df is None or self.content_features is None:
            return []

        try:
            # Find song index
            song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index[0]

            # Compute similarities
            similarities = cosine_similarity(
                [self.content_features[song_idx]], 
                self.content_features
            )[0]

            # Get most similar songs (excluding the song itself)
            similar_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]

            recommendations = self.songs_df.iloc[similar_indices]
            return recommendations.to_dict('records')

        except (IndexError, KeyError):
            return self.recommend_by_emotion('happy', num_recommendations)

    def hybrid_recommend(self, 
                        user_emotion: str,
                        user_history: List[int] = None,
                        num_recommendations: int = 10,
                        mood_regulation: str = "match") -> List[Dict]:
        """
        Hybrid recommendation combining emotion-based and collaborative filtering.

        Args:
            user_emotion: Current user emotion
            user_history: List of song IDs the user has liked
            num_recommendations: Number of recommendations
            mood_regulation: Mood regulation strategy

        Returns:
            List of recommended songs
        """
        # Get emotion-based recommendations
        emotion_recs = self.recommend_by_emotion(
            user_emotion, 
            num_recommendations, 
            mood_regulation
        )

        # If user history is available, incorporate collaborative filtering
        if user_history and len(user_history) > 0:
            # Find similar songs based on user history
            similar_recs = []
            for song_id in user_history[-5:]:  # Use last 5 songs
                similar_songs = self.recommend_similar_songs(song_id, 5)
                similar_recs.extend(similar_songs)

            # Combine and deduplicate recommendations
            all_recs = emotion_recs + similar_recs
            seen_ids = set()
            unique_recs = []

            for rec in all_recs:
                if 'song_id' in rec and rec['song_id'] not in seen_ids:
                    unique_recs.append(rec)
                    seen_ids.add(rec['song_id'])

                if len(unique_recs) >= num_recommendations:
                    break

            return unique_recs[:num_recommendations]

        return emotion_recs

    def get_song_details(self, song_id: int) -> Dict:
        """Get detailed information about a specific song."""
        if self.songs_df is None:
            return {}

        song = self.songs_df[self.songs_df['song_id'] == song_id]
        if not song.empty:
            return song.iloc[0].to_dict()
        return {}

    def add_user_feedback(self, song_id: int, rating: float):
        """Add user feedback for future recommendations (placeholder)."""
        # This would update user preferences in a real system
        print(f"Feedback recorded: Song {song_id} rated {rating}/5")

    def get_emotion_distribution(self) -> Dict:
        """Get distribution of emotions in the dataset."""
        if self.songs_df is None or 'emotion' in self.songs_df.columns:
            return {}

        return self.songs_df['emotion'].value_counts().to_dict()

# Example usage
if __name__ == "__main__":
    recommender = EmotionBasedRecommender()

    print("Emotify Recommendation Engine")
    print("=" * 40)

    # Test emotion-based recommendations
    print("\nHappy mood recommendations:")
    happy_recs = recommender.recommend_by_emotion("happy", 5)
    for i, song in enumerate(happy_recs, 1):
        print(f"{i}. {song['title']} by {song['artist']}")

    print("\nSad mood boost recommendations:")
    sad_boost_recs = recommender.recommend_by_emotion("sad", 5, "boost")
    for i, song in enumerate(sad_boost_recs, 1):
        print(f"{i}. {song['title']} by {song['artist']}")

    # Test similar song recommendations
    print("\nSongs similar to song ID 1:")
    similar_recs = recommender.recommend_similar_songs(1, 3)
    for i, song in enumerate(similar_recs, 1):
        print(f"{i}. {song['title']} by {song['artist']}")
