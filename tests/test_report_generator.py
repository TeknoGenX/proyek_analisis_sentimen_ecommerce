"""
Unit test untuk modul BusinessReportGenerator.
"""
import tempfile
import unittest
from pathlib import Path
from src.report_generator import BusinessReportGenerator, generate_business_report
import config


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = BusinessReportGenerator()

    def test_image_to_base64(self):
        # Image that exists
        if config.CONFUSION_MATRIX_PATH.exists():
            b64 = self.generator._image_to_base64(config.CONFUSION_MATRIX_PATH)
            self.assertIsNotNone(b64)
            self.assertTrue(b64.startswith("data:image/"))

        # Image that doesn't exist
        b64_none = self.generator._image_to_base64(Path("non_existent_image.png"))
        self.assertIsNone(b64_none)

    def test_load_metrics_and_stats(self):
        model_m = self.generator._load_model_metrics()
        self.assertIsInstance(model_m, dict)

        comp_m = self.generator._load_comparison_metrics()
        self.assertIsInstance(comp_m, list)

        stats = self.generator._load_dataset_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total", stats)

    def test_generate_html_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_report.html"
            result_path = self.generator.generate_html_report(output_path=out_file)

            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 1000)

            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verifikasi komponen penting laporan
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Laporan Analisis Sentimen E-Commerce Indonesia", content)
            self.assertIn("Ringkasan Eksekutif", content)
            self.assertIn("Benchmark Komparatif 7 Platform", content)
            self.assertIn("Net Sentiment Score (NSS)", content)
            self.assertIn("Rekomendasi Strategis", content)
            self.assertIn("@media print", content)

    def test_generate_html_report_with_none_and_nan_metrics(self):
        """Memverifikasi bahwa nilai None dan NaN pada NSS / rating tidak memicu TypeError saat sort/format."""
        import json
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_f = Path(tmp_dir) / "test_comp_metrics.json"
            dummy_comp = [
                {"nama_platform": "Platform A", "nss": 25.5, "total_ulasan": 100, "pct_positif": 60.0, "pct_netral": 10.0, "pct_negatif": 30.0, "rata_rata_rating": 4.5},
                {"nama_platform": "Platform B", "nss": None, "total_ulasan": 50, "pct_positif": 20.0, "pct_netral": 30.0, "pct_negatif": 50.0, "rata_rata_rating": None},
                {"nama_platform": "Platform C", "nss": float("nan"), "total_ulasan": 20, "pct_positif": 10.0, "pct_netral": 10.0, "pct_negatif": 80.0, "rata_rata_rating": float("nan")}
            ]
            with open(metrics_f, "w", encoding="utf-8") as f:
                json.dump(dummy_comp, f)

            gen = BusinessReportGenerator(comparison_metrics_path=metrics_f)
            out_file = Path(tmp_dir) / "test_report_none.html"
            res_path = gen.generate_html_report(output_path=out_file)
            self.assertTrue(res_path.exists())


if __name__ == "__main__":
    unittest.main()
