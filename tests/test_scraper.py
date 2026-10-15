"""
Unit test untuk modul scraper dan filtering ulasan multi-platform.
"""
import unittest
from pathlib import Path
import sys
import pandas as pd

# Tambahkan root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.scraper import (
    rating_to_sentiment,
    resolve_platform_info,
    filter_tiktok_commerce_reviews,
    load_dataset
)


class TestScraperModule(unittest.TestCase):
    def test_rating_to_sentiment(self):
        self.assertEqual(rating_to_sentiment(1), "negatif")
        self.assertEqual(rating_to_sentiment(2), "negatif")
        self.assertEqual(rating_to_sentiment(3), "netral")
        self.assertEqual(rating_to_sentiment(4), "positif")
        self.assertEqual(rating_to_sentiment(5), "positif")

    def test_resolve_platform_info_known_platforms(self):
        for key in ["shopee", "tokopedia", "lazada", "blibli", "tiktok_shop", "bukalapak", "zalora"]:
            k, name, app_id = resolve_platform_info(platform_key=key)
            self.assertEqual(k, key)
            self.assertEqual(name, config.ECOMMERCE_PLATFORMS[key]["name"])
            self.assertEqual(app_id, config.ECOMMERCE_PLATFORMS[key]["app_id"])

    def test_resolve_platform_info_by_app_id(self):
        k, name, app_id = resolve_platform_info(app_id="blibli.mobile.commerce")
        self.assertEqual(k, "blibli")
        self.assertEqual(name, "Blibli")

    def test_filter_tiktok_commerce_reviews(self):
        sample_data = {
            "teks_ulasan": [
                "kenapa keranjang kuning ku hilang pas mau checkout pesanan?",
                "video fyp joget nya seru banget filternya keren",
                "beli baju disini cepat sampai tapi kain tipis",
                "akun saya kena banned tolong kembalikan",
                "ongkir mahal banget barang tidak dikirim seller penipu",
                "update versi baru malah lag pas nonton siaran langsung video"
            ],
            "rating": [1, 5, 3, 1, 1, 2]
        }
        df = pd.DataFrame(sample_data)
        filtered = filter_tiktok_commerce_reviews(df)
        
        # Ulasan belanja harus lolos (indeks 0, 2, 4)
        self.assertEqual(len(filtered), 3)
        self.assertIn("keranjang kuning", filtered["teks_ulasan"].iloc[0])
        self.assertIn("beli baju", filtered["teks_ulasan"].iloc[1])
        self.assertIn("ongkir mahal", filtered["teks_ulasan"].iloc[2])

    def test_load_dataset_backward_compatibility(self):
        df = load_dataset()
        self.assertIn("teks_ulasan", df.columns)
        self.assertIn("rating", df.columns)
        self.assertIn("label", df.columns)
        self.assertIn("platform", df.columns)


if __name__ == "__main__":
    unittest.main()
