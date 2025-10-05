"""
Dataset generation for Emotify.
Handles creation of a large-scale synthetic dataset.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

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

    header_written = False
    for i in tqdm(range(0, num_records, chunk_size), desc="Generating dataset chunks"):
        num_to_generate = min(chunk_size, num_records - i)
        
        genres = ['pop', 'rock', 'hip-hop', 'latin', 'edm', 'r&b']
        
        df = pd.DataFrame({
            'energy': np.random.uniform(0, 1, num_to_generate),
            'tempo': np.random.uniform(50, 200, num_to_generate),
            'danceability': np.random.uniform(0, 1, num_to_generate),
            'playlist_genre': np.random.choice(genres, num_to_generate),
            'loudness': np.random.uniform(-60, 0, num_to_generate),
            'liveness': np.random.uniform(0, 1, num_to_generate),
            'valence': np.random.uniform(0, 1, num_to_generate),
            'track_artist': [f'Artist {j}' for j in range(i, i + num_to_generate)],
            'time_signature': np.random.choice([3, 4, 5], num_to_generate, p=[0.1, 0.8, 0.1]),
            'speechiness': np.random.uniform(0, 1, num_to_generate),
            'track_popularity': np.random.randint(0, 101, num_to_generate),
            'track_name': [f'Song {j}' for j in range(i, i + num_to_generate)],
            'duration_ms': np.random.randint(60000, 600000, num_to_generate),
            'acousticness': np.random.uniform(0, 1, num_to_generate),
            'instrumentalness': np.random.uniform(0, 1, num_to_generate),
            'track_id': [f'track_{j}' for j in range(i, i + num_to_generate)],
        })

        # Append to CSV
        df.to_csv(file_path, mode='a', header=not header_written, index=False)
        header_written = True

    print("Large dataset generation complete.")

if __name__ == "__main__":
    output_path = "/Users/test/Downloads/EMOTIFY_Multimodal_Music_Recommendation_macOS/data/datasets/large_spotify_dataset.csv"
    create_large_dataset(output_path, num_records=1_000_000)
