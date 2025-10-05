"""
Unit tests for Emotify modules.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from emotion_detection.text_emotion import TextEmotionDetector
    from recommendation_engine.recommender import EmotionBasedRecommender
    from data_processing.data_utils import DataProcessor
except ImportError as e:
    print(f"Import error in tests: {e}")

class TestTextEmotionDetector(unittest.TestCase):
    """Test cases for text emotion detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = TextEmotionDetector()

    def test_happy_emotion(self):
        """Test detection of happy emotion."""
        result = self.detector.predict("I'm feeling really happy and excited!")
        self.assertEqual(result['emotion'], 'happy')
        self.assertGreater(result['confidence'], 0.5)

    def test_sad_emotion(self):
        """Test detection of sad emotion."""
        result = self.detector.predict("I'm feeling really sad and depressed")
        self.assertEqual(result['emotion'], 'sad')
        self.assertGreater(result['confidence'], 0.3)

    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.detector.predict("")
        self.assertIn('error', result)

class TestRecommendationEngine(unittest.TestCase):
    """Test cases for recommendation engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.recommender = EmotionBasedRecommender()

    def test_emotion_based_recommendations(self):
        """Test emotion-based recommendations."""
        recommendations = self.recommender.recommend_by_emotion("happy", 5)
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)

    def test_mood_regulation_strategies(self):
        """Test different mood regulation strategies."""
        match_recs = self.recommender.recommend_by_emotion("sad", 3, "match")
        boost_recs = self.recommender.recommend_by_emotion("sad", 3, "boost")

        self.assertIsInstance(match_recs, list)
        self.assertIsInstance(boost_recs, list)

class TestDataProcessor(unittest.TestCase):
    """Test cases for data processing utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()

    def test_sample_dataset_creation(self):
        """Test creation of sample dataset."""
        df = self.processor.create_sample_dataset(10)
        self.assertEqual(len(df), 10)
        self.assertIn('emotion', df.columns)
        self.assertIn('title', df.columns)

    def test_emotion_normalization(self):
        """Test emotion label normalization."""
        import pandas as pd

        test_df = pd.DataFrame({
            'emotion': ['joy', 'depression', 'peaceful', 'energy']
        })

        normalized_df = self.processor._normalize_emotions(test_df)

        expected_emotions = {'happy', 'sad', 'calm', 'energetic'}
        actual_emotions = set(normalized_df['emotion'].unique())

        self.assertTrue(actual_emotions.issubset(expected_emotions))

if __name__ == '__main__':
    unittest.main()
