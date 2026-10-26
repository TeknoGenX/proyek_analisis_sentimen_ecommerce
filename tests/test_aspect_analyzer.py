"""
Unit test untuk modul Aspect-Based Sentiment Analysis (ABSA).
"""
import unittest
import pandas as pd
from src.aspect_analyzer import AspectSentimentAnalyzer, ASPECT_TAXONOMY


class TestAspectSentimentAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = AspectSentimentAnalyzer()

    def test_clause_segmentation(self):
        text = "Barangnya bagus banget, tapi pengirimannya lelet parah dan kurirnya judes."
        clauses = self.analyzer.segment_clauses(text)
        self.assertIsInstance(clauses, list)
        self.assertGreater(len(clauses), 1)

    def test_empty_clause_segmentation(self):
        self.assertEqual(self.analyzer.segment_clauses(""), [])
        self.assertEqual(self.analyzer.segment_clauses("   "), [])

    def test_identify_aspects_in_clause(self):
        clause_shipping = "kurirnya lambat banget kirim paket"
        aspects = self.analyzer.identify_aspects_in_clause(clause_shipping)
        self.assertIn("pengiriman", aspects)

        clause_product = "barangnya bagus ori asli tidak rusak"
        aspects = self.analyzer.identify_aspects_in_clause(clause_product)
        self.assertIn("kualitas_produk", aspects)

        clause_price = "harga murah gratis ongkir dapat diskon promo"
        aspects = self.analyzer.identify_aspects_in_clause(clause_price)
        self.assertIn("harga_promo", aspects)

        clause_cs = "seller ramah fast respon balas chat"
        aspects = self.analyzer.identify_aspects_in_clause(clause_cs)
        self.assertIn("pelayanan_cs", aspects)

        clause_app = "aplikasi error lag pas mau checkout pembayaran"
        aspects = self.analyzer.identify_aspects_in_clause(clause_app)
        self.assertIn("aplikasi_sistem", aspects)

    def test_analyze_aspects_multiaspect(self):
        text = "Barangnya bagus original mantap, tapi pengiriman lelet kurir lambat banget paket telat tiba."
        res = self.analyzer.analyze_aspects(text)

        self.assertIn("teks_asli", res)
        self.assertIn("aspek_terdeteksi", res)
        self.assertIn("ringkasan_sentimen", res)

        aspects = res["aspek_terdeteksi"]
        self.assertIn("kualitas_produk", aspects)
        self.assertIn("pengiriman", aspects)
        self.assertEqual(aspects["kualitas_produk"]["sentimen"], "positif")
        self.assertEqual(aspects["pengiriman"]["sentimen"], "negatif")

    def test_analyze_aspects_empty(self):
        res = self.analyzer.analyze_aspects("")
        self.assertEqual(res["total_aspek_terdeteksi"], 0)
        self.assertEqual(res["aspek_terdeteksi"], {})

    def test_analyze_aspects_nan_and_none(self):
        """Memverifikasi penanganan nilai None, np.nan, dan tipe non-string tanpa TypeError."""
        import numpy as np
        res_none = self.analyzer.analyze_aspects(None)
        self.assertEqual(res_none["total_aspek_terdeteksi"], 0)
        self.assertEqual(res_none["teks_asli"], "")

        res_nan = self.analyzer.analyze_aspects(np.nan)
        self.assertEqual(res_nan["total_aspek_terdeteksi"], 0)
        self.assertEqual(res_nan["teks_asli"], "")

        res_num = self.analyzer.analyze_aspects(12345)
        self.assertEqual(res_num["total_aspek_terdeteksi"], 0)

        # Uji juga segment_clauses langsung
        self.assertEqual(self.analyzer.segment_clauses(None), [])
        self.assertEqual(self.analyzer.segment_clauses(np.nan), [])
        self.assertEqual(self.analyzer.segment_clauses(999), ["999"])

    def test_analyze_dataframe(self):
        df = pd.DataFrame({
            "ulasan": [
                "Barang sangat bagus original!",
                "Pengiriman super lambat mengecewakan."
            ]
        })
        res_df = self.analyzer.analyze_dataframe(df)

        for asp in ASPECT_TAXONOMY.keys():
            col_name = f"aspek_{asp}"
            self.assertIn(col_name, res_df.columns)

        self.assertEqual(res_df.loc[0, "aspek_kualitas_produk"], "positif")
        self.assertEqual(res_df.loc[1, "aspek_pengiriman"], "negatif")

    def test_analyze_dataframe_invalid_column(self):
        df = pd.DataFrame({"ulasan": ["bagus"]})
        with self.assertRaises(KeyError):
            self.analyzer.analyze_dataframe(df, text_column="kolom_tidak_ada")


if __name__ == "__main__":
    unittest.main()
