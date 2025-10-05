#!/usr/bin/env python3
"""
Run script for Emotify Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main run function."""
    print("🎵 Starting Emotify Application...")

    # Change to project directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    # Check if requirements are installed
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import nltk
        import sklearn
        print("✓ All requirements satisfied")
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please install requirements: pip install -r requirements/requirements.txt")
        return

    # Run Streamlit app
    app_path = "src/web_interface/app.py"
    if Path(app_path).exists():
        print(f"🚀 Launching Emotify at http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    else:
        print(f"❌ App file not found: {app_path}")

if __name__ == "__main__":
    main()
