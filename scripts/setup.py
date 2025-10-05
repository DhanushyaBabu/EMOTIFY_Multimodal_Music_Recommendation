#!/usr/bin/env python3
"""
Setup script for Emotify project.
Downloads required datasets and initializes the project environment.
"""

import os
import sys
from pathlib import Path
import requests
import zipfile
from urllib.parse import urlparse

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/datasets",
        "models/trained",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        print("Downloading NLTK data...")

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('wordnet', quiet=True)

        print("✓ NLTK data downloaded successfully")
    except ImportError:
        print("⚠ NLTK not installed. Please install requirements first.")
    except Exception as e:
        print(f"⚠ Error downloading NLTK data: {e}")

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_path = Path(".env")
    template_path = Path(".env.template")

    if not env_path.exists() and template_path.exists():
        with open(template_path, 'r') as template:
            with open(env_path, 'w') as env_file:
                env_file.write(template.read())
        print(f"✓ Created .env file from template")
    else:
        print("ℹ .env file already exists or template not found")

def setup_sample_data():
    """Set up sample data for development."""
    try:
        # Import after ensuring the path is correct
        sys.path.append(str(Path("src")))
        from data_processing.data_utils import DataProcessor

        processor = DataProcessor()

        # Create larger sample dataset
        print("Creating sample music dataset...")
        sample_df = processor.create_sample_dataset(200)

        # Save dataset
        dataset_path = Path("data/datasets/sample_songs.csv")
        processor.save_dataset(sample_df, str(dataset_path))

        print(f"✓ Created sample dataset with {len(sample_df)} songs")

    except Exception as e:
        print(f"⚠ Error creating sample data: {e}")

def main():
    """Main setup function."""
    print("🎵 Emotify Setup Script")
    print("=" * 40)

    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Setup steps
    print("\n1. Creating project directories...")
    create_directories()

    print("\n2. Setting up environment file...")
    create_env_file()

    print("\n3. Downloading NLTK data...")
    download_nltk_data()

    print("\n4. Creating sample dataset...")
    setup_sample_data()

    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements/requirements.txt")
    print("2. Configure API keys in .env file")
    print("3. Run the app: streamlit run src/web_interface/app.py")

if __name__ == "__main__":
    main()
