"""
Audio-based emotion detection module for Emotify.
Extracts audio features and classifies emotions from music audio.
"""

import librosa
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AudioEmotionDetector:
    """
    Audio emotion detection using librosa for feature extraction
    and machine learning models for classification.
    """

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio emotion detector.

        Args:
            sample_rate: Audio sample rate for processing
        """
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        self.model = None

        # Audio feature configuration
        self.feature_config = {
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'n_chroma': 12,
            'n_contrast': 6  # Reduced from 7 to 6
        }

        # Initialize with a basic model (can be replaced with trained model)
        self._initialize_basic_model()

    def _initialize_basic_model(self):
        """Initialize a basic model for demo purposes."""
        # This would normally load a pre-trained model
        # For demo, we'll create a simple rule-based classifier
        self.model = "rule_based"

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract comprehensive audio features from an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Feature vector as numpy array
        """
        try:
            # Load audio file at its native sample rate to avoid resampling errors
            y, sr = librosa.load(audio_path, sr=None)

            # Extract various audio features
            features = []

            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.feature_config['n_mfcc'])
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            features.extend(mfccs_mean)
            features.extend(mfccs_std)

            # 2. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.feature_config['n_chroma'])
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)

            # 3. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=self.feature_config['n_contrast'])
            contrast_mean = np.mean(contrast, axis=1)
            features.extend(contrast_mean)

            # 4. Tempo and rhythm features
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # Ensure tempo is a single value, not an array
            if isinstance(tempo, np.ndarray):
                tempo = np.mean(tempo) if tempo.size > 0 else 0.0

            features.append(tempo)

            # 5. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)

            features.append(np.mean(spectral_centroid))
            features.append(np.mean(spectral_rolloff))
            features.append(np.mean(spectral_bandwidth))
            features.append(np.mean(zero_crossing_rate))
            features.append(np.std(zero_crossing_rate))  # New feature for laughter detection
            features.append(np.mean(spectral_flatness))

            # 6. Energy and dynamics
            rms_energy = librosa.feature.rms(y=y)
            features.append(np.mean(rms_energy))
            features.append(np.std(rms_energy))

            return np.array(features)

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            # Return zero vector if extraction fails
            return np.zeros(50)  # Approximate expected feature vector size

    def extract_features_from_array(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract features from audio array (useful for streaming audio).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Feature vector as numpy array
        """
        if sr is None:
            sr = self.sample_rate

        try:
            features = []

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.feature_config['n_mfcc'])
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))

            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)

            # Spectral features
            features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
            features.append(np.mean(librosa.feature.rms(y=y)))

            return np.array(features)

        except Exception as e:
            print(f"Error extracting features from audio array: {str(e)}")
            return np.zeros(50)

    def _rule_based_classify(self, features: np.ndarray) -> Dict:
        """
        Advanced rule-based classification using multiple audio features.
        """
        # Add a check for a zero vector, which indicates feature extraction failed
        if np.all(features == 0):
            return {
                "emotion": "unknown",
                "confidence": 0.1,
                "valence": 0.5,
                "arousal": 0.5,
                "tempo": 0,
                "energy": 0,
                "brightness": 0,
                "error": "Feature extraction failed"
            }
            
        # Check for valid feature vector
        if len(features) < 50:
            return {"emotion": "calm", "confidence": 0.5, "valence": 0.5, "arousal": 0.5}

        # Feature indices (approximate based on extraction order)
        tempo_idx = 40
        energy_idx = -2
        energy_std_idx = -1
        brightness_idx = -8
        zcr_idx = -6
        zcr_std_idx = -5
        flatness_idx = -4

        # Extract features
        tempo = features[tempo_idx]
        energy = features[energy_idx]
        energy_std = features[energy_std_idx]
        brightness = features[brightness_idx]
        zcr = features[zcr_idx]
        zcr_std = features[zcr_std_idx]
        flatness = features[flatness_idx]

        # Normalize features
        tempo_norm = min(tempo / 200.0, 1.0)
        energy_norm = min(energy * 150, 1.0)
        brightness_norm = min(brightness / 8000.0, 1.0)
        energy_variation_norm = min(energy_std * 10, 1.0)

        # Speechiness metric: high ZCR and high flatness suggest non-tonal, speech-like audio
        speechiness = (zcr * 0.6 + flatness * 0.4)
        
        # Laughter metric: high energy variation and high zcr variation
        laughter_score = (energy_variation_norm * 0.6 + zcr_std * 0.4)

        if laughter_score > 0.3:  # Strong indicator of a non-musical vocalization (laughing or crying)
            # Differentiate between laughing (less bright) and crying (brighter) based on user's samples
            if brightness_norm > 0.35: # Crying is "brighter" in user's sample
                emotion = "sad"
                valence = 0.3 - (energy_variation_norm * 0.2)
                arousal = 0.5 + (energy_norm * 0.3)
                confidence = min(1 - ((valence + arousal) / 2), 0.99)
            else: # Laughter has less high-frequency energy in user's sample
                emotion = "happy"
                valence = 0.5 + (brightness_norm * 0.3) + (laughter_score * 0.2)
                arousal = 0.6 + (energy_norm * 0.4)
                confidence = min((valence + arousal) / 2, 0.99)

        elif speechiness > 0.1:  # Threshold for other speech-like audio
            # Logic for speech
            valence = brightness_norm * 0.7 + energy_norm * 0.3
            arousal = energy_norm * 0.8 + brightness_norm * 0.2

            if arousal > 0.6 and valence > 0.5:
                emotion = "happy"
                confidence = (arousal + valence) / 2
            elif arousal > 0.6:
                emotion = "energetic"  # Could be angry speech
                confidence = arousal
            else:
                emotion = "calm"  # Neutral speech
                confidence = 1 - arousal
        else:
            # Original logic for music
            valence = (brightness_norm * 0.5 + tempo_norm * 0.5)
            arousal = (energy_norm * 0.6 + tempo_norm * 0.4)

            if arousal > 0.6 and valence > 0.6:
                emotion = "happy"
                confidence = (arousal + valence) / 2
            elif arousal > 0.6 and valence <= 0.4:
                emotion = "energetic"
                confidence = arousal
            elif arousal < 0.4 and valence < 0.4:
                emotion = "sad"
                confidence = 1 - (arousal + valence) / 2
            else:
                emotion = "calm"
                confidence = 1 - arousal

        confidence = min(max(confidence, 0.1), 0.99)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "valence": valence,
            "arousal": arousal,
            "tempo": tempo,
            "energy": energy_norm,
            "brightness": brightness_norm
        }

    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze emotion from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with emotion prediction and audio characteristics
        """
        if not os.path.exists(audio_path):
            return {
                "emotion": "calm",
                "confidence": 0.1,
                "error": f"Audio file not found: {audio_path}"
            }

        try:
            # Extract features
            features = self.extract_features(audio_path)

            # Classify emotion
            result = self._rule_based_classify(features)
            result["audio_file"] = audio_path

            return result

        except Exception as e:
            return {
                "emotion": "calm",
                "confidence": 0.1,
                "error": f"Analysis error: {str(e)}"
            }

    def analyze_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Analyze emotions for multiple audio files."""
        return [self.analyze_audio(path) for path in audio_paths]

    def analyze_directory(self, audio_dir: str) -> List[Dict]:
        """
        Analyze all audio files in a directory.
        
        Args:
            audio_dir: Path to directory containing audio files
            
        Returns:
            List of analysis results for each audio file
        """
        results = []
        try:
            for file in os.listdir(audio_dir):
                if file.lower().endswith(('.mp3', '.wav', '.ogg')):
                    file_path = os.path.join(audio_dir, file)
                    result = self.analyze_audio(file_path)
                    result['file_name'] = file
                    results.append(result)
            return results
        except Exception as e:
            print(f"Error analyzing directory {audio_dir}: {str(e)}")
            return []

    def train_model(self, feature_vectors: List[np.ndarray], labels: List[str]):
        """
        Train a machine learning model for emotion classification.

        Args:
            feature_vectors: List of extracted feature vectors
            labels: Corresponding emotion labels
        """
        if len(feature_vectors) != len(labels):
            raise ValueError("Number of feature vectors must match number of labels")

        # Stack features
        X = np.vstack(feature_vectors)
        y = np.array(labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        return self.model

    def save_model(self, model_path: str):
        """Save trained model and scaler."""
        if self.model is not None and self.model != "rule_based":
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.feature_config
            }, model_path)

    def load_model(self, model_path: str):
        """Load trained model and scaler."""
        if os.path.exists(model_path):
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_config = saved_data.get('config', self.feature_config)

# Example usage
if __name__ == "__main__":
    detector = AudioEmotionDetector()

    print("Audio Emotion Detection Module")
    print("=" * 40)
    print("This module analyzes audio features to detect emotions.")
    print("Features extracted:")
    print("- MFCCs (Mel-frequency cepstral coefficients)")
    print("- Chroma features")
    print("- Spectral contrast")
    print("- Tempo and rhythm")
    print("- Spectral features (centroid, rolloff, bandwidth)")
    print("- Energy and dynamics")
    print()
    print("Usage:")
    print("detector = AudioEmotionDetector()")
    print("result = detector.analyze_audio('path/to/song.mp3')")
