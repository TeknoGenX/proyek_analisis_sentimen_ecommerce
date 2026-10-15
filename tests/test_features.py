"""
Unit test untuk modul ekstraksi fitur TF-IDF (FeatureExtractor).
"""
import unittest
from pathlib import Path
import sys
import pandas as pd

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.features import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.test_dir = config.MODELS_DIR / "test_features"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.test_dir / "test_tfidf.pkl"
        self.extractor = FeatureExtractor(vectorizer_path=self.vec_path)

        self.corpus = pd.Series([
            "barang bagus packing aman kurir cepat",
            "aplikasi error tidak bisa checkout kecewa",
            "harga murah kualitas lumayan oke",
            "pelayanan buruk respon lambat"
        ])

    def tearDown(self):
        if self.vec_path.exists():
            self.vec_path.unlink()
        if self.test_dir.exists():
            try:
                self.test_dir.rmdir()
            except OSError:
                pass

    def test_fit_transform_and_transform(self):
        """Memverifikasi ekstraksi fitur TF-IDF dan transformasi data baru."""
        X_train = self.extractor.fit_transform(self.corpus)
        self.assertEqual(X_train.shape[0], len(self.corpus))
        self.assertGreater(X_train.shape[1], 0)

        # Uji transformasi data baru
        new_text = pd.Series(["barang bagus sekali"])
        X_new = self.extractor.transform(new_text)
        self.assertEqual(X_new.shape[0], 1)
        self.assertEqual(X_new.shape[1], X_train.shape[1])

    def test_save_and_load_vectorizer(self):
        """Memverifikasi serialisasi dan pemuatan ulang TF-IDF vectorizer."""
        self.extractor.fit_transform(self.corpus)
        self.extractor.save_vectorizer()
        self.assertTrue(self.vec_path.exists())

        new_extractor = FeatureExtractor(vectorizer_path=self.vec_path)
        new_extractor.load_vectorizer()
        self.assertIsNotNone(new_extractor.vectorizer)

        X_new = new_extractor.transform(pd.Series(["barang bagus"]))
        self.assertGreater(X_new.nnz, 0)

    def test_transform_before_fit_raises_error(self):
        """Memverifikasi error saat transform dipanggil sebelum fit."""
        fresh_extractor = FeatureExtractor(vectorizer_path=self.vec_path)
        with self.assertRaises(ValueError):
            fresh_extractor.transform(pd.Series(["tes teks"]))


if __name__ == "__main__":
    unittest.main()
