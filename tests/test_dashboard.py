"""
Unit test untuk modul Web Dashboard (Streamlit).
"""
import unittest
from src.dashboard import (
    SENTIMENT_COLORS,
    SENTIMENT_BADGES,
    PRESET_EXAMPLES,
    load_predictor,
    load_aspect_analyzer,
)
from src.aspect_analyzer import AspectSentimentAnalyzer
from src.predictor import SentimentPredictor


class TestDashboard(unittest.TestCase):
    def test_sentiment_constants(self):
        for label in ["positif", "netral", "negatif"]:
            self.assertIn(label, SENTIMENT_COLORS)
            self.assertIn(label, SENTIMENT_BADGES)

    def test_preset_examples(self):
        self.assertGreater(len(PRESET_EXAMPLES), 1)
        self.assertIn("Pilih Contoh Ulasan...", PRESET_EXAMPLES)

    def test_load_predictor_and_aspect(self):
        pred = load_predictor()
        self.assertIsNotNone(pred)
        self.assertIsInstance(pred, SentimentPredictor)

        absa = load_aspect_analyzer(pred)
        self.assertIsNotNone(absa)
        self.assertIsInstance(absa, AspectSentimentAnalyzer)


if __name__ == "__main__":
    unittest.main()
