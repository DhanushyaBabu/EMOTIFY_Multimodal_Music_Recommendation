"""
Text-based emotion detection module for Emotify.
Supports multiple NLP models for sentiment and emotion analysis.
"""

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import joblib
from typing import Dict, List, Tuple
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextEmotionDetector:
    """
    Multimodal text emotion detection using VADER, TextBlob, and custom models.
    """

    def __init__(self, model_type: str = "vader"):
        """
        Initialize the text emotion detector.

        Args:
            model_type: Type of model to use ("vader", "textblob", or "custom")
        """
        self.model_type = model_type
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Emotion mapping for basic 4-class system
        self.emotion_mapping = {
            "happy": ["joy", "happiness", "excited", "cheerful", "upbeat", "positive"],
            "sad": ["sadness", "depression", "melancholy", "down", "blue", "negative"],
            "calm": ["relaxed", "peaceful", "serene", "tranquil", "mellow", "chill"],
            "energetic": ["energy", "pump", "workout", "intense", "vigorous", "dynamic"]
        }

    def _vader_predict(self, text: str) -> Dict:
        """Predict emotion using VADER sentiment analyzer."""
        scores = self.vader_analyzer.polarity_scores(text)

        # Convert VADER scores to emotion categories
        compound = scores['compound']

        if compound >= 0.5:
            emotion = "happy"
            confidence = min(compound, 1.0)
        elif compound <= -0.5:
            emotion = "sad"
            confidence = min(abs(compound), 1.0)
        elif -0.5 < compound < 0.1:
            emotion = "calm"
            confidence = 1.0 - abs(compound)
        else:  # 0.1 <= compound < 0.5
            emotion = "energetic"
            confidence = compound

        return {
            "emotion": emotion,
            "confidence": confidence,
            "raw_scores": scores
        }

    def _textblob_predict(self, text: str) -> Dict:
        """Predict emotion using TextBlob sentiment analysis."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Convert polarity to emotion categories
        if polarity >= 0.3:
            emotion = "happy"
            confidence = polarity
        elif polarity <= -0.3:
            emotion = "sad"
            confidence = abs(polarity)
        elif subjectivity < 0.3:
            emotion = "calm"
            confidence = 1.0 - subjectivity
        else:
            emotion = "energetic"
            confidence = subjectivity

        return {
            "emotion": emotion,
            "confidence": confidence,
            "polarity": polarity,
            "subjectivity": subjectivity
        }

    def _keyword_based_predict(self, text: str) -> Dict:
        """Simple keyword-based emotion detection."""
        text_lower = text.lower()
        emotion_scores = {emotion: 0 for emotion in self.emotion_mapping.keys()}

        for emotion, keywords in self.emotion_mapping.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1

        # Find dominant emotion
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]

        confidence = max_score / len(text_lower.split()) if max_score > 0 else 0.1

        return {
            "emotion": max_emotion,
            "confidence": min(confidence, 1.0),
            "emotion_scores": emotion_scores
        }

    def predict(self, text: str) -> Dict:
        """
        Predict emotion from input text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with emotion prediction and confidence
        """
        if not text or not text.strip():
            return {"emotion": "calm", "confidence": 0.1, "error": "Empty text input"}

        try:
            if self.model_type == "vader":
                return self._vader_predict(text)
            elif self.model_type == "textblob":
                return self._textblob_predict(text)
            elif self.model_type == "keyword":
                return self._keyword_based_predict(text)
            else:
                # Default to VADER
                return self._vader_predict(text)

        except Exception as e:
            return {
                "emotion": "calm",
                "confidence": 0.1,
                "error": f"Prediction error: {str(e)}"
            }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict emotions for multiple texts."""
        return [self.predict(text) for text in texts]

    def analyze_lyrics(self, lyrics: str) -> Dict:
        """
        Specialized method for analyzing song lyrics.

        Args:
            lyrics: Song lyrics text

        Returns:
            Detailed emotion analysis for lyrics
        """
        # Split lyrics into verses for granular analysis
        verses = [verse.strip() for verse in lyrics.split('\n\n') if verse.strip()]

        verse_emotions = []
        for verse in verses:
            emotion_result = self.predict(verse)
            verse_emotions.append(emotion_result)

        # Aggregate emotions across verses
        if verse_emotions:
            # Get most common emotion
            emotions = [result['emotion'] for result in verse_emotions]
            emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)

            # Average confidence
            avg_confidence = np.mean([result['confidence'] for result in verse_emotions])

            return {
                "overall_emotion": dominant_emotion,
                "confidence": avg_confidence,
                "verse_emotions": verse_emotions,
                "emotion_distribution": emotion_counts
            }
        else:
            return self.predict(lyrics)

# Example usage and testing
if __name__ == "__main__":
    detector = TextEmotionDetector("vader")

    # Test cases
    test_texts = [
        "I'm feeling really happy and excited today!",
        "This is such a sad and depressing song",
        "I need some calm and relaxing music",
        "Let's get pumped up with some energetic beats!"
    ]

    print("Text Emotion Detection Results:")
    print("=" * 40)

    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Emotion: {result['emotion']} (Confidence: {result['confidence']:.2f})")
        print("-" * 40)
