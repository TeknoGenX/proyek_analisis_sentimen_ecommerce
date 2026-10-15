"""
Unit test untuk modul model trainer dan validasi K-Fold.
"""
import unittest
from pathlib import Path
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_trainer import SentimentModelTrainer


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = SentimentModelTrainer()

    def test_single_class_raises_error(self):
        # Data latih hanya memiliki 1 variasi label sentimen
        X_train = ["bagus", "mantap", "hebat"]
        y_train = ["positif", "positif", "positif"]
        X_test = ["bagus"]
        y_test = ["positif"]

        vec = TfidfVectorizer()
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        with self.assertRaises(ValueError) as ctx:
            self.trainer.train_and_benchmark(X_train_vec, y_train, X_test_vec, y_test)
        self.assertIn("minimal 2 kelas", str(ctx.exception))

    def test_dynamic_kfold_with_small_sample(self):
        # Data latih dengan jumlah sampel sedikit per kelas (misal: 2 per kelas)
        X_train = ["bagus", "keren", "jelek", "rusak"]
        y_train = ["positif", "positif", "negatif", "negatif"]
        X_test = ["mantap", "parah"]
        y_test = ["positif", "negatif"]

        vec = TfidfVectorizer()
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        # Harus berhasil tanpa crash ValueError n_splits > n_samples
        results = self.trainer.train_and_benchmark(X_train_vec, y_train, X_test_vec, y_test)
        self.assertIsNotNone(self.trainer.best_model)
        self.assertIn("Logistic Regression (Balanced)", results)


if __name__ == "__main__":
    unittest.main()
