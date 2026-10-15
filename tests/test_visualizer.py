"""
Unit test untuk modul visualizer dan analisis komparatif 7 platform.
"""
import unittest
from pathlib import Path
import sys
import pandas as pd
import shutil

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.visualizer import SentimentVisualizer


class TestSentimentVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_output_dir = config.OUTPUTS_DIR / "test_outputs"
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        cls.visualizer = SentimentVisualizer(output_dir=cls.test_output_dir)

        # Dataset sintetis untuk pengujian 7 platform
        cls.sample_df = pd.DataFrame({
            "teks_ulasan": [
                "aplikasi shopee lemot banget pas flash sale",
                "shopee gratis ongkir mantap!",
                "tokopedia biaya admin mahal",
                "tokopedia cepat sampai",
                "lazada kurir lex ramah barang aman",
                "blibli barang original customer service terbaik",
                "tiktok shop keranjang kuning error pas live",
                "bukalapak mitra sangat membantu usaha warung",
                "zalora retur gampang size chart pas kualitas baju bagus"
            ],
            "clean_review": [
                "aplikasi shopee lambat diskon kilat",
                "shopee gratis ongkos kirim mantap",
                "tokopedia biaya administrasi mahal",
                "tokopedia cepat sampai",
                "lazada kurir ekspedisi lazada ramah barang aman",
                "blibli barang asli layanan pelanggan baik",
                "tiktok shop keranjang belanja error siaran langsung",
                "bukalapak mitra bantu usaha warung",
                "zalora pengembalian barang mudah panduan ukuran pas kualitas baju bagus"
            ],
            "rating": [1, 5, 2, 5, 5, 5, 1, 5, 5],
            "label": [
                "negatif", "positif",
                "negatif", "positif",
                "positif",
                "positif",
                "negatif",
                "positif",
                "positif"
            ],
            "platform": [
                "shopee", "shopee",
                "tokopedia", "tokopedia",
                "lazada",
                "blibli",
                "tiktok_shop",
                "bukalapak",
                "zalora"
            ]
        })

    @classmethod
    def tearDownClass(cls):
        # Bersihkan direktori pengujian
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir, ignore_errors=True)

    def test_calculate_platform_metrics(self):
        metrics_df = self.visualizer.calculate_platform_metrics(self.sample_df)
        self.assertIn("platform", metrics_df.columns)
        self.assertIn("nss", metrics_df.columns)
        self.assertIn("pct_positif", metrics_df.columns)
        self.assertIn("pct_negatif", metrics_df.columns)
        
        # Periksa perhitungan NSS Blibli (100% positif - 0% negatif = 100)
        blibli_row = metrics_df[metrics_df["platform"] == "blibli"].iloc[0]
        self.assertEqual(blibli_row["nss"], 100.0)

        # Periksa perhitungan NSS TikTok Shop (0% positif - 100% negatif = -100)
        tiktok_row = metrics_df[metrics_df["platform"] == "tiktok_shop"].iloc[0]
        self.assertEqual(tiktok_row["nss"], -100.0)

        # Periksa perhitungan NSS Shopee (50% positif - 50% negatif = 0)
        shopee_row = metrics_df[metrics_df["platform"] == "shopee"].iloc[0]
        self.assertEqual(shopee_row["nss"], 0.0)

    def test_plot_comparison_sentiment_distribution(self):
        save_file = self.test_output_dir / "test_dist.png"
        res_path = self.visualizer.plot_comparison_sentiment_distribution(self.sample_df, save_path=save_file)
        self.assertTrue(Path(res_path).exists())
        self.assertGreater(Path(res_path).stat().st_size, 0)

    def test_plot_comparison_nss(self):
        save_file = self.test_output_dir / "test_nss.png"
        res_path = self.visualizer.plot_comparison_nss(self.sample_df, save_path=save_file)
        self.assertTrue(Path(res_path).exists())
        self.assertGreater(Path(res_path).stat().st_size, 0)

    def test_plot_platform_pain_points_grid(self):
        save_file = self.test_output_dir / "test_grid.png"
        res_path = self.visualizer.plot_platform_pain_points_grid(self.sample_df, save_path=save_file)
        self.assertTrue(Path(res_path).exists())
        self.assertGreater(Path(res_path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
