"""
Unit test untuk modul preprocessor dan normalisasi slang e-commerce.
"""
import unittest
from pathlib import Path
import sys

# Tambahkan path root proyek ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import IndonesianTextPreprocessor
import config


class TestIndonesianTextPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = IndonesianTextPreprocessor()

    def test_slang_normalization(self):
        text = "brg bgs bgt seller ramah bgtu mantul"
        cleaned = self.preprocessor.clean(text, do_stemming=False)
        self.assertIn("barang", cleaned)
        self.assertIn("bagus", cleaned)
        self.assertIn("penjual", cleaned)

    def test_ecommerce_specific_slang(self):
        # Menguji istilah e-commerce spesifik yang diminta
        text = "beli di lazmall pake cod gratisongkir liat size chart mau retur spaylater error di bmart dan keranjang kuning"
        
        # 1. Uji tahap normalisasi slang langsung
        normalized = self.preprocessor.normalize_slang(text)
        self.assertIn("toko resmi lazada", normalized)      # lazmall
        self.assertIn("bayar di tempat", normalized)        # cod
        self.assertIn("gratis ongkos kirim", normalized)    # gratisongkir
        self.assertIn("panduan ukuran", normalized)         # size chart
        self.assertIn("pengembalian barang", normalized)    # retur
        self.assertIn("paylater shopee", normalized)        # spaylater
        self.assertIn("blibli mart", normalized)            # bmart
        self.assertIn("keranjang belanja", normalized)      # keranjang kuning

        # 2. Uji hasil akhir clean (setelah stopword removal)
        cleaned = self.preprocessor.clean(text, do_stemming=False)
        self.assertIn("toko resmi lazada", cleaned)
        self.assertIn("bayar tempat", cleaned)              # 'di' tereliminasi oleh stopword removal
        self.assertIn("gratis ongkos kirim", cleaned)
        self.assertIn("panduan ukuran", cleaned)
        self.assertIn("pengembalian barang", cleaned)
        self.assertIn("paylater shopee", cleaned)
        self.assertIn("blibli mart", cleaned)
        self.assertIn("keranjang belanja", cleaned)

    def test_preserve_negations(self):
        text = "aplikasi tidak bagus dan sangat mengecewakan"
        cleaned = self.preprocessor.clean(text, do_stemming=False)
        # Pastikan kata negasi 'tidak' tidak terbuang
        self.assertIn("tidak", cleaned)

    def test_repeated_characters(self):
        text = "mantaaaaap parahhhh"
        cleaned = self.preprocessor.reduce_repeated_characters(text)
        self.assertEqual(cleaned, "mantap parah")

    def test_emoji_mapping(self):
        text = "Barang mantap 👍 kurir ramah"
        cleaned = self.preprocessor.clean(text, do_stemming=False)
        self.assertIn("bagus", cleaned)


if __name__ == "__main__":
    unittest.main()
