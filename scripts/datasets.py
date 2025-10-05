"""
Dataset management and processing utilities for Emotify.
Handles dataset loading, preprocessing, and sample dataset creation.
Updated for Spotify 1 Million Tracks dataset structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')


class DatasetManager:
    """
    Manages loading and preprocessing of Spotify 1 Million Tracks dataset.
    Dataset contains ~1M tracks with 19 features covering 2000-2023.
    """

    def __init__(self):
        # Mapping to normalize emotion labels based on audio features
        # These will be derived from valence and energy values
        self.emotion_mapping = {
            'happy': 'happy',
            'joy': 'happy',
            'excited': 'happy',
            'cheerful': 'happy',
            'upbeat': 'happy',
            'positive': 'happy',
            'euphoric': 'happy',
            'elated': 'happy',
            'joyful': 'happy',
            'optimistic': 'happy',
            'sad': 'sad',
            'depression': 'sad',
            'melancholy': 'sad',
            'down': 'sad',
            'blue': 'sad',
            'negative': 'sad',
            'gloomy': 'sad',
            'sorrowful': 'sad',
            'depressed': 'sad',
            'dejected': 'sad',
            'calm': 'calm',
            'relaxed': 'calm',
            'peaceful': 'calm',
            'serene': 'calm',
            'tranquil': 'calm',
            'mellow': 'calm',
            'chill': 'calm',
            'soothing': 'calm',
            'zen': 'calm',
            'meditative': 'calm',
            'energetic': 'energetic',
            'energy': 'energetic',
            'pump': 'energetic',
            'intense': 'energetic',
            'vigorous': 'energetic',
            'dynamic': 'energetic',
            'powerful': 'energetic',
            'aggressive': 'energetic',
            'hyped': 'energetic',
            'pumped': 'energetic',
        }

        # Expected columns for Spotify 1 Million Tracks dataset
        self.expected_columns = [
            'Unnamed: 0',      # Index column (unique id)
            'artist_name',         # Name of the artist
            'track_name',      # Track or record name
            'track_id',        # Unique id from Spotify for a track
            'popularity',      # Score range 0 to 100
            'year',    # Released year (2000 to 2023)
            'genre',           # Genre type of the song
            'danceability',    # Track suitability for dancing (0.0 to 1.0)
            'energy',          # Measure of intensity and activity (0.0 to 1.0)
            'key',             # The key track is in (-1 to 11)
            'loudness',        # Loudness in dB
            'mode',            # Major (1) or minor (0)
            'speechiness',     # Speechiness (0.0 to 1.0)
            'acousticness',    # Acousticness (0.0 to 1.0)
            'instrumentalness', # Instrumentalness (0.0 to 1.0)
            'liveness',        # Liveness (0.0 to 1.0)
            'valence',         # Musical positiveness (0.0 to 1.0)
            'tempo',           # Tempo in BPM
            'duration_ms',     # Duration in milliseconds
            'time_signature'   # Time signature
        ]

        # Audio feature ranges for validation
        self.feature_ranges = {
            'danceability': (0.0, 1.0),
            'energy': (0.0, 1.0),
            'key': (-1, 11),
            'loudness': (-60, 5),
            'mode': (0, 1),
            'speechiness': (0.0, 1.0),
            'acousticness': (0.0, 1.0),
            'instrumentalness': (0.0, 1.0),
            'liveness': (0.0, 1.0),
            'valence': (0.0, 1.0),
            'tempo': (0, 300),
            'popularity': (0, 100),
            'time_signature': (0, 7)
        }

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load the Spotify 1 Million Tracks dataset CSV file and preprocess it.
        Args:
            file_path: Path to the spotify_data.csv dataset file.
        Returns:
            Preprocessed DataFrame with emotion labels derived from audio features.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            print(f"Loading dataset from: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Rename columns to match expected names
            df = df.rename(columns={'artist_name': 'artists', 'year': 'release_date'})

            # Validate dataset structure
            expected_cols_after_rename = [col if col not in ['artist_name', 'year'] else ('artists' if col == 'artist_name' else 'release_date') for col in self.expected_columns]
            missing_cols = set(expected_cols_after_rename) - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing expected columns: {missing_cols}")

            # Clean and preprocess the data
            df = self._clean_text_fields(df)
            df = self._validate_audio_features(df)
            df = self._derive_emotions_from_features(df)
            df = self._handle_missing_values(df)
            df = self._add_additional_features(df)

            print(f"Dataset preprocessing completed. Final shape: {df.shape}")
            return df

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Creating a sample dataset instead...")
            return self.create_sample_spotify_dataset(50)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating a sample dataset instead...")
            return self.create_sample_spotify_dataset(50)

    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean textual fields such as track_name and artists.
        """
        text_fields = ['track_name', 'artists', 'genre']

        for field in text_fields:
            if field in df.columns:
                # Convert to string and strip whitespace
                df[field] = df[field].astype(str).str.strip()

                # Remove excessive whitespace
                df[field] = df[field].str.replace(r'\s+', ' ', regex=True)

                # Handle special characters based on field type
                if field == 'artists':
                    # Keep ampersands, commas, parentheses for artist collaborations
                    df[field] = df[field].str.replace(r'[^\w\s.,!?\-&()]+', '', regex=True)
                elif field == 'track_name':
                    # Keep basic punctuation for song titles
                    df[field] = df[field].str.replace(r'[^\w\s.,!?\-&()\'\"]+', '', regex=True)
                else:
                    # Basic cleaning for genre
                    df[field] = df[field].str.replace(r'[^\w\s\-]+', '', regex=True)

                # Remove rows with invalid text
                df = df[df[field] != '']
                df = df[df[field] != 'nan']

        return df

    def _validate_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean audio feature values.
        """
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                # Convert to numeric
                df[feature] = pd.to_numeric(df[feature], errors='coerce')

                # Clip values to valid ranges
                df[feature] = df[feature].clip(lower=min_val, upper=max_val)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        # Fill missing text fields
        if 'artists' in df.columns:
            df['artists'] = df['artists'].fillna('Unknown Artist')
        if 'track_name' in df.columns:
            df['track_name'] = df['track_name'].fillna('Unknown Track')
        if 'genre' in df.columns:
            df['genre'] = df['genre'].fillna('pop')  # Default to pop genre

        # Fill missing audio features with median values grouped by genre
        audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                         'speechiness', 'acousticness', 'instrumentalness',
                         'liveness', 'valence', 'tempo', 'popularity']

        for feature in audio_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')

                # Fill with genre-specific median, then overall median
                if 'genre' in df.columns:
                    df[feature] = df.groupby('genre')[feature].transform(
                        lambda x: x.fillna(x.median())
                    )

                # Fill remaining NaNs with overall median
                df[feature] = df[feature].fillna(df[feature].median())

        # Handle missing dates
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_numeric(df['release_date'], errors='coerce')
            df['release_date'] = df['release_date'].fillna(df['release_date'].median())
            df['release_date'] = df['release_date'].astype(int)

        # Handle missing duration
        if 'duration_ms' in df.columns:
            df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
            df['duration_ms'] = df['duration_ms'].fillna(df['duration_ms'].median())

        return df

    def _derive_emotions_from_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive emotion labels from valence and energy audio features.
        Uses Russell's Circumplex Model of Affect with refined thresholds.
        """
        if 'valence' in df.columns and 'energy' in df.columns:
            # Convert to numeric if they're strings
            df['valence'] = pd.to_numeric(df['valence'], errors='coerce').fillna(0.5)
            df['energy'] = pd.to_numeric(df['energy'], errors='coerce').fillna(0.5)

            # Refined emotion classification using Russell's model
            conditions = [
                (df['valence'] >= 0.6) & (df['energy'] >= 0.6),   # High valence, high energy -> Happy
                (df['valence'] < 0.4) & (df['energy'] >= 0.6),    # Low valence, high energy -> Energetic/Angry
                (df['valence'] >= 0.6) & (df['energy'] < 0.4),    # High valence, low energy -> Calm/Content
                (df['valence'] < 0.4) & (df['energy'] < 0.4),     # Low valence, low energy -> Sad
            ]

            choices = ['happy', 'energetic', 'calm', 'sad']

            # Default case for middle values (0.4 <= valence < 0.6 or 0.4 <= energy < 0.6)
            df['emotion'] = np.select(conditions, choices, default='calm')

            # Add emotion confidence score
            df['emotion_confidence'] = np.maximum(
                np.abs(df['valence'] - 0.5) * 2,  # Distance from neutral valence
                np.abs(df['energy'] - 0.5) * 2    # Distance from neutral energy
            )

        else:
            # If valence/energy not available, assign based on other features
            if 'danceability' in df.columns and 'tempo' in df.columns:
                # Use danceability and tempo as proxies
                df['valence'] = df.get('danceability', 0.5)
                df['energy'] = (df.get('tempo', 120) - 60) / 140  # Normalize tempo
                df = self._derive_emotions_from_features(df)  # Recursive call
            else:
                # Random assignment as last resort
                df['emotion'] = np.random.choice(['happy', 'sad', 'calm', 'energetic'],
                                               size=len(df), p=[0.3, 0.2, 0.25, 0.25])
                df['emotion_confidence'] = 0.5

        return df

    def _add_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for better analysis.
        """
        # Add decade information
        if 'release_date' in df.columns:
            df['decade'] = (df['release_date'] // 10) * 10

        # Add duration in minutes
        if 'duration_ms' in df.columns:
            df['duration_min'] = df['duration_ms'] / 60000

        # Add mood intensity (combination of energy and valence)
        if 'energy' in df.columns and 'valence' in df.columns:
            df['mood_intensity'] = np.sqrt((df['energy']**2 + df['valence']**2) / 2)

        # Add danceability category
        if 'danceability' in df.columns:
            df['dance_category'] = pd.cut(df['danceability'],
                                        bins=[0, 0.3, 0.6, 1.0],
                                        labels=['Low', 'Medium', 'High'])

        # Add popularity category
        if 'popularity' in df.columns:
            df['popularity_category'] = pd.cut(df['popularity'],
                                             bins=[0, 30, 60, 100],
                                             labels=['Low', 'Medium', 'High'])

        return df

    def create_sample_spotify_dataset(self, num_songs: int = 50) -> pd.DataFrame:
        """
        Generate a sample dataset matching Spotify 1 Million Tracks structure.
        Args:
            num_songs: Number of sample songs to generate.
        Returns:
            DataFrame with sample songs in Spotify dataset format.
        """
        np.random.seed(42)

        # Extended list of genres from Spotify's catalog
        sample_genres = [
            'pop', 'rock', 'hip-hop', 'electronic', 'indie', 'jazz', 'classical',
            'country', 'r&b', 'folk', 'reggae', 'blues', 'punk', 'metal', 'funk',
            'disco', 'house', 'techno', 'ambient', 'latin', 'soul', 'gospel',
            'alternative', 'grunge', 'ska', 'dubstep', 'trap', 'lo-fi', 'indie-pop',
            'synthwave', 'progressive', 'post-rock', 'world', 'new-age', 'downtempo'
        ]

        # Generate realistic artist names
        artist_prefixes = ['The', 'DJ', 'MC', 'Lil', 'Big', 'Young', 'Old', 'Super', 'Electric', '']
        artist_names = ['Storm', 'Wave', 'Fire', 'Moon', 'Star', 'River', 'Cloud',
                       'Thunder', 'Lightning', 'Shadow', 'Phoenix', 'Dragon', 'Wolf',
                       'Ocean', 'Mountain', 'Forest', 'Desert', 'Galaxy', 'Comet', 'Aurora']

        artists = []
        for i in range(num_songs):
            prefix = np.random.choice(artist_prefixes)
            name = np.random.choice(artist_names)
            suffix = np.random.choice(['', 'Band', 'Collective', 'Project', ''])
            number = np.random.randint(1, 999) if np.random.random() < 0.2 else ''

            full_name = f"{prefix} {name} {suffix}{number}".strip()
            full_name = ' '.join(full_name.split())  # Remove extra spaces
            artists.append(full_name)

        # Generate diverse track names
        track_adjectives = ['Lost', 'Broken', 'Electric', 'Midnight', 'Golden', 'Silent',
                           'Burning', 'Frozen', 'Dancing', 'Falling', 'Rising', 'Shining']
        track_nouns = ['Love', 'Night', 'Day', 'Dream', 'Heart', 'Soul', 'Life',
                      'Time', 'World', 'Fire', 'Water', 'Sky', 'Road', 'Home',
                      'Memory', 'Story', 'Song', 'Light', 'Shadow', 'Rain']

        track_names = []
        for _ in range(num_songs):
            if np.random.random() < 0.4:  # Single word titles
                track_names.append(np.random.choice(track_nouns))
            elif np.random.random() < 0.7:  # Adjective + Noun
                track_names.append(f"{np.random.choice(track_adjectives)} {np.random.choice(track_nouns)}")
            else:  # More complex titles
                track_names.append(f"{np.random.choice(track_nouns)} of {np.random.choice(track_nouns)}")

        # Generate correlated audio features based on genre
        df_data = []

        for i in range(num_songs):
            genre = np.random.choice(sample_genres)

            # Genre-specific feature generation
            if genre in ['electronic', 'house', 'techno', 'dubstep', 'trap']:
                energy = np.random.uniform(0.6, 1.0)
                danceability = np.random.uniform(0.5, 1.0)
                valence = np.random.uniform(0.4, 0.9)
                acousticness = np.random.uniform(0.0, 0.3)
                instrumentalness = np.random.uniform(0.3, 1.0)
                tempo = np.random.uniform(120, 180)
            elif genre in ['jazz', 'blues', 'soul']:
                energy = np.random.uniform(0.3, 0.8)
                danceability = np.random.uniform(0.4, 0.8)
                valence = np.random.uniform(0.3, 0.7)
                acousticness = np.random.uniform(0.4, 0.9)
                instrumentalness = np.random.uniform(0.1, 0.7)
                tempo = np.random.uniform(80, 140)
            elif genre in ['classical', 'ambient', 'new-age']:
                energy = np.random.uniform(0.1, 0.5)
                danceability = np.random.uniform(0.1, 0.4)
                valence = np.random.uniform(0.2, 0.8)
                acousticness = np.random.uniform(0.6, 1.0)
                instrumentalness = np.random.uniform(0.7, 1.0)
                tempo = np.random.uniform(60, 120)
            elif genre in ['rock', 'metal', 'punk']:
                energy = np.random.uniform(0.7, 1.0)
                danceability = np.random.uniform(0.3, 0.7)
                valence = np.random.uniform(0.2, 0.8)
                acousticness = np.random.uniform(0.0, 0.4)
                instrumentalness = np.random.uniform(0.0, 0.5)
                tempo = np.random.uniform(120, 200)
            else:  # pop, indie, country, etc.
                energy = np.random.uniform(0.4, 0.8)
                danceability = np.random.uniform(0.4, 0.8)
                valence = np.random.uniform(0.3, 0.8)
                acousticness = np.random.uniform(0.2, 0.7)
                instrumentalness = np.random.uniform(0.0, 0.3)
                tempo = np.random.uniform(90, 150)

            song_data = {
                'Unnamed: 0': i,
                'artists': artists[i],
                'track_name': track_names[i],
                'track_id': f"spotify:track:{np.random.randint(100000, 999999)}",
                'popularity': np.random.randint(0, 101),
                'release_date': np.random.randint(2000, 2024),
                'genre': genre,
                'danceability': danceability,
                'energy': energy,
                'key': np.random.randint(-1, 12),
                'loudness': np.random.uniform(-60, 5),
                'mode': np.random.randint(0, 2),
                'speechiness': np.random.uniform(0.0, 0.5),
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': np.random.uniform(0.0, 0.5),
                'valence': valence,
                'tempo': tempo,
                'duration_ms': np.random.randint(30000, 600000),
                'time_signature': np.random.choice([3, 4, 5], p=[0.05, 0.9, 0.05])
            }

            df_data.append(song_data)

        df = pd.DataFrame(df_data)

        # Derive emotions and add additional features
        df = self._derive_emotions_from_features(df)
        df = self._add_additional_features(df)

        return df

    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about the loaded dataset.
        """
        info = {
            'total_tracks': len(df),
            'unique_artists': df['artists'].nunique() if 'artists' in df.columns else 0,
            'unique_genres': df['genre'].nunique() if 'genre' in df.columns else 0,
            'year_range': (int(df['release_date'].min()), int(df['release_date'].max())) if 'release_date' in df.columns else None,
            'decade_distribution': df['decade'].value_counts().to_dict() if 'decade' in df.columns else {},
            'emotion_distribution': df['emotion'].value_counts().to_dict() if 'emotion' in df.columns else {},
            'genre_distribution': df['genre'].value_counts().head(10).to_dict() if 'genre' in df.columns else {},
            'avg_popularity': round(df['popularity'].mean(), 2) if 'popularity' in df.columns else 0,
            'avg_duration_min': round(df['duration_min'].mean(), 2) if 'duration_min' in df.columns else 0,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_quality_score': self._calculate_data_quality_score(df)
        }
        return info

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a data quality score based on completeness and validity.
        """
        # Check completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))

        # Check feature validity (values within expected ranges)
        validity_scores = []
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                valid_count = ((df[feature] >= min_val) & (df[feature] <= max_val)).sum()
                validity_scores.append(valid_count / len(df))

        validity = np.mean(validity_scores) if validity_scores else 1.0

        # Overall quality score
        quality_score = (completeness * 0.6 + validity * 0.4)
        return round(quality_score, 3)

    def filter_by_emotion(self, df: pd.DataFrame, emotion: str) -> pd.DataFrame:
        """
        Filter dataset by emotion label.
        """
        if 'emotion' in df.columns:
            return df[df['emotion'] == emotion].copy()
        return pd.DataFrame()

    def filter_by_genre(self, df: pd.DataFrame, genre: str) -> pd.DataFrame:
        """
        Filter dataset by genre (case-insensitive, partial matching).
        """
        if 'genre' in df.columns:
            return df[df['genre'].str.contains(genre, case=False, na=False)].copy()
        return pd.DataFrame()

    def filter_by_year_range(self, df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Filter dataset by year range.
        """
        if 'release_date' in df.columns:
            return df[(df['release_date'] >= start_year) & (df['release_date'] <= end_year)].copy()
        return pd.DataFrame()

    def filter_by_popularity(self, df: pd.DataFrame, min_popularity: int = 50) -> pd.DataFrame:
        """
        Filter dataset by minimum popularity score.
        """
        if 'popularity' in df.columns:
            return df[df['popularity'] >= min_popularity].copy()
        return pd.DataFrame()

    def get_recommendations(self, df: pd.DataFrame, target_emotion: str,
                          min_confidence: float = 0.6, limit: int = 10) -> pd.DataFrame:
        """
        Get song recommendations based on emotion and confidence.
        """
        if 'emotion' not in df.columns or 'emotion_confidence' not in df.columns:
            return pd.DataFrame()

        filtered = df[
            (df['emotion'] == target_emotion) &
            (df['emotion_confidence'] >= min_confidence)
        ].copy()

        # Sort by confidence and popularity
        if not filtered.empty:
            filtered = filtered.sort_values(
                ['emotion_confidence', 'popularity'],
                ascending=[False, False]
            ).head(limit)

        return filtered[['track_name', 'artists', 'genre', 'emotion',
                        'emotion_confidence', 'popularity']].copy()

    def export_processed_dataset(self, df: pd.DataFrame, file_path: str) -> bool:
        """
        Export the processed dataset to CSV.
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Dataset exported successfully to: {file_path}")
            return True
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            return False

    def get_statistics_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get detailed statistical summary of audio features.
        """
        if df.empty:
            return {}

        audio_features = ['danceability', 'energy', 'valence', 'acousticness',
                         'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']

        stats = {}
        for feature in audio_features:
            if feature in df.columns:
                stats[feature] = {
                    'mean': round(df[feature].mean(), 3),
                    'median': round(df[feature].median(), 3),
                    'std': round(df[feature].std(), 3),
                    'min': round(df[feature].min(), 3),
                    'max': round(df[feature].max(), 3),
                    'q25': round(df[feature].quantile(0.25), 3),
                    'q75': round(df[feature].quantile(0.75), 3)
                }

        return stats


if __name__ == "__main__":
    # Test the DatasetManager
    manager = DatasetManager()

    # Create sample dataset for testing
    print("Creating sample dataset...")
    sample_df = manager.load_dataset("nonexistent_file.csv")  # Will create sample

    # Display dataset information
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    info = manager.get_dataset_info(sample_df)
    for key, value in info.items():
        print(f"{key}: {value}")

    # Display sample data
    print("\n" + "="*50)
    print("SAMPLE DATA (First 5 rows)")
    print("="*50)
    print(sample_df.head())

    # Test filtering functions
    print("\n" + "="*50)
    print("FILTERING EXAMPLES")
    print("="*50)

    happy_songs = manager.filter_by_emotion(sample_df, 'happy')
    print(f"Happy songs: {len(happy_songs)}")

    pop_songs = manager.filter_by_genre(sample_df, 'pop')
    print(f"Pop songs: {len(pop_songs)}")

    recent_songs = manager.filter_by_year_range(sample_df, 2020, 2023)
    print(f"Recent songs (2020-2023): {len(recent_songs)}")

    # Test recommendations
    print("\n" + "="*50)
    print("RECOMMENDATION EXAMPLE")
    print("="*50)
    recommendations = manager.get_recommendations(sample_df, 'happy', min_confidence=0.5, limit=5)
    print(recommendations)

    # Display statistics
    print("\n" + "="*50)
    print("AUDIO FEATURES STATISTICS")
    print("="*50)
    stats = manager.get_statistics_summary(sample_df)
    for feature, feature_stats in list(stats.items())[:3]:  # Show first 3 features
        print(f"\n{feature.upper()}:")
        for stat_name, stat_value in feature_stats.items():
            print(f"  {stat_name}: {stat_value}")
