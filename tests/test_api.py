"""
Unit test untuk modul REST API FastAPI (src.api).
Menguji endpoint /health, /predict, /predict-batch, /metrics, dan /platforms.
"""
import unittest
from pathlib import Path
import sys

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api import app
import config


class TestSentimentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Memverifikasi respons root endpoint /."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("docs_url", data)
        self.assertIn("endpoints", data)

    def test_health_check(self):
        """Memverifikasi health check endpoint /health."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("target_platforms", data)

    def test_predict_single_endpoint(self):
        """Memverifikasi prediksi sentimen tunggal via POST /predict."""
        payload = {"text": "Barang sangat bagus, pengiriman super cepat dan packing rapi!"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["teks_asli"], payload["text"])
        self.assertIn("sentimen", data)
        self.assertIn("keyakinan", data)
        self.assertIn("probabilitas", data)
        self.assertIn(data["sentimen"], ["positif", "negatif", "netral"])

    def test_predict_batch_endpoint(self):
        """Memverifikasi prediksi sentimen batch via POST /predict-batch."""
        payload = {
            "texts": [
                "Barang bagus mantap original",
                "Aplikasi rusak parah error terus kecewa",
                "Standar biasa saja"
            ]
        }
        response = self.client.post("/predict-batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 3)
        self.assertIn("summary", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 3)
        self.assertIn("rata_rata_keyakinan", data)

    def test_predict_batch_empty_list(self):
        """Memverifikasi penanganan daftar teks kosong pada POST /predict-batch."""
        response = self.client.post("/predict-batch", json={"texts": []})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 0)
        self.assertEqual(len(data["results"]), 0)

    def test_metrics_endpoint(self):
        """Memverifikasi pengambilan metrik evaluasi model via GET /metrics."""
        if config.METRICS_PATH.exists():
            response = self.client.get("/metrics")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("best_model_name", data)

    def test_platforms_endpoint(self):
        """Memverifikasi pengambilan metrik benchmark NSS via GET /platforms."""
        if config.COMPARISON_METRICS_PATH.exists():
            response = self.client.get("/platforms")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsInstance(data, list)
            if data:
                self.assertIn("nama_platform", data[0])
                self.assertIn("nss", data[0])

    def test_report_html_endpoint(self):
        """Memverifikasi pengambilan laporan HTML via GET /report/html."""
        response = self.client.get("/report/html")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("<!DOCTYPE html>", response.text)

    def test_report_generate_endpoint(self):
        """Memverifikasi pembuatan laporan via POST /report/generate."""
        response = self.client.post("/report/generate")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("report_path", data)


if __name__ == "__main__":
    unittest.main()
