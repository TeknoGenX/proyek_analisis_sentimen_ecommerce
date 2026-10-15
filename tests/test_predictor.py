"""
Unit test untuk modul inferensi SentimentPredictor.
Mencakup pengujian model LinearSVC (biner & multiclass), LogisticRegression,
penanganan input kosong, dan batch inference.
"""
import unittest
from pathlib import Path
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.predictor import SentimentPredictor


class TestSentimentPredictor(unittest.TestCase):
    def setUp(self):
        self.test_dir = config.MODELS_DIR / "test_models"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.test_dir / "test_vec.pkl"
        self.model_path = self.test_dir / "test_model.pkl"

        # Buat vectorizer dummy
        self.texts = [
            "barang sangat bagus mantap pengiriman cepat",
            "produk jelek rusak parah mengecewakan",
            "lumayan standar biasa saja"
        ]
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.texts)
        joblib.dump(self.vectorizer, self.vec_path)

    def tearDown(self):
        if self.vec_path.exists():
            self.vec_path.unlink()
        if self.model_path.exists():
            self.model_path.unlink()
        if self.test_dir.exists():
            try:
                self.test_dir.rmdir()
            except OSError:
                pass

    def test_predict_binary_linearsvc(self):
        """Memverifikasi perbaikan bug binary decision_function pada LinearSVC."""
        y_binary = np.array(["positif", "negatif", "positif"])
        svc_binary = LinearSVC(random_state=42).fit(self.X, y_binary)
        joblib.dump(svc_binary, self.model_path)

        predictor = SentimentPredictor(
            model_path=self.model_path,
            vectorizer_path=self.vec_path
        )

        res_pos = predictor.predict_one("barang bagus mantap")
        self.assertIn(res_pos["sentimen"], ["positif", "negatif"])
        self.assertGreaterEqual(res_pos["keyakinan"], 0.0)
        self.assertLessEqual(res_pos["keyakinan"], 100.0)
        self.assertIn("positif", res_pos["probabilitas"])
        self.assertIn("negatif", res_pos["probabilitas"])

        res_neg = predictor.predict_one("produk rusak jelek parah")
        self.assertIn(res_neg["sentimen"], ["positif", "negatif"])
        self.assertGreaterEqual(res_neg["keyakinan"], 0.0)

    def test_predict_multiclass_linearsvc(self):
        """Memverifikasi LinearSVC untuk multiclass (3 kelas)."""
        y_multi = np.array(["positif", "negatif", "netral"])
        svc_multi = LinearSVC(random_state=42).fit(self.X, y_multi)
        joblib.dump(svc_multi, self.model_path)

        predictor = SentimentPredictor(
            model_path=self.model_path,
            vectorizer_path=self.vec_path
        )

        res = predictor.predict_one("pelayanan sangat bagus")
        self.assertIn(res["sentimen"], ["positif", "negatif", "netral"])
        self.assertGreaterEqual(res["keyakinan"], 0.0)
        self.assertEqual(len(res["probabilitas"]), 3)

    def test_predict_proba_model(self):
        """Memverifikasi model yang memiliki predict_proba (LogisticRegression)."""
        y_multi = np.array(["positif", "negatif", "netral"])
        lr = LogisticRegression(random_state=42).fit(self.X, y_multi)
        joblib.dump(lr, self.model_path)

        predictor = SentimentPredictor(
            model_path=self.model_path,
            vectorizer_path=self.vec_path
        )

        res = predictor.predict_one("bagus sekali")
        self.assertIn(res["sentimen"], ["positif", "negatif", "netral"])
        self.assertGreaterEqual(res["keyakinan"], 0.0)
        self.assertIn("positif", res["probabilitas"])

    def test_predict_empty_input(self):
        """Memverifikasi penanganan string kosong atau None."""
        y_multi = np.array(["positif", "negatif", "netral"])
        svc = LinearSVC(random_state=42).fit(self.X, y_multi)
        joblib.dump(svc, self.model_path)

        predictor = SentimentPredictor(
            model_path=self.model_path,
            vectorizer_path=self.vec_path
        )

        res_empty = predictor.predict_one("")
        self.assertEqual(res_empty["sentimen"], "netral")
        self.assertEqual(res_empty["keyakinan"], 0.0)

        res_spaces = predictor.predict_one("   ")
        self.assertEqual(res_spaces["sentimen"], "netral")
        self.assertEqual(res_spaces["keyakinan"], 0.0)

    def test_predict_batch(self):
        """Memverifikasi inferensi sekumpulan teks (batch)."""
        y_multi = np.array(["positif", "negatif", "netral"])
        svc = LinearSVC(random_state=42).fit(self.X, y_multi)
        joblib.dump(svc, self.model_path)

        predictor = SentimentPredictor(
            model_path=self.model_path,
            vectorizer_path=self.vec_path
        )

        batch_results = predictor.predict_batch([
            "barang mantap sekali",
            "rusak parah kecewa"
        ])
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(batch_results[0]["sentimen"], "positif")
        self.assertEqual(batch_results[1]["sentimen"], "negatif")

        # Uji juga input string tunggal ke predict_batch
        single_batch = predictor.predict_batch("barang mantap")
        self.assertEqual(len(single_batch), 1)


if __name__ == "__main__":
    unittest.main()
