
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import uuid
from datetime import datetime, timedelta

def create_large_dataset(file_path: str, num_records: int = 1_000_000, chunk_size: int = 100_000):
    """
    Generate a large dataset of songs and save it to a CSV file in chunks.
    Args:
        file_path: Path to save the CSV dataset file.
        num_records: Total number of records to generate.
        chunk_size: Number of records to generate per chunk.
    """
    print(f"Generating large dataset with {num_records} records at {file_path}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If the file already exists, remove it to start fresh
    if os.path.exists(file_path):
        os.remove(file_path)

    header_written = False
    for i in tqdm(range(0, num_records, chunk_size), desc="Generating dataset chunks"):
        num_to_generate = min(chunk_size, num_records - i)
        
        genres = ['brazilian', 'turkish', 'hip-hop', 'ambient', 'pop', 'rock', 'latin', 'edm', 'r&b', 'chill', 'academic', 'funk', 'meditative']
        
        # Generate track IDs
        track_ids = [str(uuid.uuid4().hex)[:22] for _ in range(num_to_generate)]
        
        # Generate release dates
        start_date = datetime(1990, 1, 1)
        end_date = datetime.now()
        release_dates = [start_date + timedelta(days=np.random.randint((end_date - start_date).days)) for _ in range(num_to_generate)]

        df = pd.DataFrame({
            'danceability': np.random.uniform(0, 1, num_to_generate),
            'tempo': np.random.uniform(50, 200, num_to_generate),
            'energy': np.random.uniform(0, 1, num_to_generate),
            'playlist_genre': np.random.choice(genres, num_to_generate),
            'loudness': np.random.uniform(-60, 0, num_to_generate),
            'speechiness': np.random.uniform(0, 1, num_to_generate),
            'valence': np.random.uniform(0, 1, num_to_generate),
            'track_artist': [f'Artist {j}' for j in range(i, i + num_to_generate)],
            'time_signature': np.random.choice([1, 3, 4, 5], num_to_generate, p=[0.05, 0.1, 0.8, 0.05]),
            'liveness': np.random.uniform(0, 1, num_to_generate),
            'track_popularity': np.random.randint(0, 101, num_to_generate),
            'track_uri': [f'spotify:track:{tid}' for tid in track_ids],
            'track_album_name': [f'Album {j}' for j in range(i, i + num_to_generate)],
            'playlist_subgenre': np.random.choice(['subgenre_a', 'subgenre_b', 'subgenre_c'], num_to_generate),
            'track_album_release_date': [d.strftime('%Y-%m-%d') for d in release_dates],
            'instrumentalness': np.random.uniform(0, 1, num_to_generate),
            'track_album_id': [str(uuid.uuid4().hex)[:22] for _ in range(num_to_generate)],
            'key': np.random.randint(0, 12, num_to_generate),
            'mode': np.random.randint(0, 2, num_to_generate),
            'duration_ms': np.random.randint(60000, 600000, num_to_generate),
            'acousticness': np.random.uniform(0, 1, num_to_generate),
            'track_id': track_ids,
            'subgenre_type': 'audio_features',
            'artist_id': [str(uuid.uuid4().hex)[:22] for _ in range(num_to_generate)],
        })

        # Reorder columns to better match the original file, though not perfectly
        column_order = [
            'danceability', 'tempo', 'energy', 'playlist_genre', 'loudness', 'speechiness', 'valence',
            'track_artist', 'time_signature', 'liveness', 'track_popularity', 'track_uri',
            'track_album_name', 'playlist_subgenre', 'track_album_release_date', 'instrumentalness',
            'track_album_id', 'key', 'mode', 'duration_ms', 'acousticness', 'track_id', 'subgenre_type', 'artist_id'
        ]
        df = df[column_order]

        # Append to CSV
        df.to_csv(file_path, mode='a', header=not header_written, index=False)
        header_written = True

    print("Large dataset generation complete.")

if __name__ == "__main__":
    output_path = "/Users/test/Downloads/EMOTIFY_Multimodal_Music_Recommendation_macOS/data/datasets/million_songs_dataset.csv"
    create_large_dataset(output_path, num_records=1_000_000)
