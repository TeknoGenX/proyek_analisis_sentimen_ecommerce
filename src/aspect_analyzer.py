"""
Modul Aspect-Based Sentiment Analysis (ABSA) untuk E-Commerce Indonesia.
Menganalisis sentimen spesifik pada 5 aspek utama:
1. pengiriman (Pengiriman & Logistik)
2. kualitas_produk (Kualitas & Kondisi Produk)
3. harga_promo (Harga, Promo & Biaya Layanan)
4. pelayanan_cs (Pelayanan Penjual & Layanan Pelanggan / CS)
5. aplikasi_sistem (Aplikasi, Fitur, & Sistem Transaksi)
"""
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

import config
from src.predictor import SentimentPredictor

logger = logging.getLogger(__name__)

# Definisi Aspek E-Commerce dan Kata Kunci Relevan
ASPECT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "pengiriman": {
        "display_name": "Pengiriman & Logistik",
        "keywords": {
            "kurir", "kirim", "pengiriman", "dikirim", "mengirim", "antar", "diantar",
            "paket", "sampai", "pesanan sampai", "ongkir", "ongkos kirim", "cepat sampai",
            "lama sampai", "lambat", "lelet", "lama", "cepat", "kilat", "ekspedisi",
            "resi", "lex", "sicepat", "jnt", "jne", "anteraja", "ninja", "pos", "kargo",
            "pengantaran", "sampainya", "nyampe", "nyampe nya"
        }
    },
    "kualitas_produk": {
        "display_name": "Kualitas & Kondisi Produk",
        "keywords": {
            "kualitas", "barang", "produk", "ori", "asli", "original", "kw", "palsu",
            "rusak", "pecah", "cacat", "bagus", "jelek", "mantap", "mantul", "keren",
            "bahan", "kain", "tebal", "tipis", "awet", "tahan", "sesuai", "kemasan",
            "packing", "bubble wrap", "bocor", "patah", "sobek", "kotor", "warna",
            "ukuran", "size", "pas", "nyaman", "adem", "rapi", "mulus", "hancur"
        }
    },
    "harga_promo": {
        "display_name": "Harga, Promo & Biaya",
        "keywords": {
            "harga", "murah", "mahal", "promo", "diskon", "voucher", "cashback",
            "gratis ongkir", "gratisongkir", "free ongkir", "bebas ongkir", "biaya",
            "biaya admin", "biaya layanan", "jasa aplikasi", "potongan", "flash sale",
            "sepadan", "worthit", "worth it", "terjangkau", "ekonomis", "kemahalan",
            "ongkir mahal", "potongan harga", "kupon"
        }
    },
    "pelayanan_cs": {
        "display_name": "Pelayanan & Customer Service",
        "keywords": {
            "seller", "penjual", "toko", "admin", "cs", "customer service", "layanan pelanggan",
            "pelayanan", "respon", "response", "ramah", "sopan", "balas", "chat",
            "komplain", "retur", "refund", "kembalikan", "pengembalian", "tukar",
            "batal", "klaim", "tanggung jawab", "solutif", "lambat respon", "fast respon",
            "slow respon", "kecewa seller", "penjual ramah", "penjual amanah", "penipu"
        }
    },
    "aplikasi_sistem": {
        "display_name": "Aplikasi & Sistem Transaksi",
        "keywords": {
            "aplikasi", "app", "apk", "sistem", "fitur", "error", "bug", "lemot",
            "lag", "update", "crash", "checkout", "cekout", "co", "bayar", "pembayaran",
            "saldo", "spay", "spaylater", "paylater", "gopay", "dana", "cod",
            "keranjang", "keranjang kuning", "login", "banned", "notifikasi", "iklan",
            "otp", "akun", "terblokir", "batal otomatis", "gangguan"
        }
    }
}

# Pola regex pemisah klausa berdasarkan tanda baca dan kata hubung kontras/koordinatif
CLAUSE_SPLIT_REGEX = re.compile(
    r'[,.;!?\n]+|\b(?:tetapi|tapi|namun|padahal|sedangkan|meskipun|walaupun|meski|cuma|hanya|sayang|sayangnya|dan)\b',
    re.IGNORECASE
)


class AspectSentimentAnalyzer:
    def __init__(self, predictor: Optional[SentimentPredictor] = None):
        """
        Inisialisasi AspectSentimentAnalyzer dengan SentimentPredictor.
        """
        self.predictor = predictor or SentimentPredictor()

    @staticmethod
    def segment_clauses(text: Any) -> List[str]:
        """
        Memecah teks ulasan menjadi klausa-klausa independen berdasarkan tanda baca
        dan kata hubung pembalik konteks (misal 'tapi', 'namun', 'dan').
        """
        if text is None or pd.isna(text) or not str(text).strip():
            return []
        text_str = str(text)
        clauses = [c.strip() for c in CLAUSE_SPLIT_REGEX.split(text_str) if c and len(c.strip()) > 1]
        return clauses if clauses else [text_str.strip()]

    def identify_aspects_in_clause(self, clause: Any) -> Set[str]:
        """
        Mendeteksi aspek apa saja yang dibahas dalam suatu klausa teks.
        """
        clause_lower = str(clause).lower()
        detected = set()

        for aspect_key, aspect_data in ASPECT_TAXONOMY.items():
            for kw in aspect_data["keywords"]:
                # Cocokkan frasa multi-kata atau kata tunggal dengan word boundary
                if " " in kw:
                    if kw in clause_lower:
                        detected.add(aspect_key)
                        break
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', clause_lower):
                        detected.add(aspect_key)
                        break

        return detected

    def analyze_aspects(self, text: Any) -> Dict[str, Any]:
        """
        Melakukan analisis sentimen berbasis aspek pada satu teks ulasan.
        
        Returns:
            Dict dengan rincian aspek yang terdeteksi, sentimen per aspek,
            skor keyakinan, dan klausa bukti pendukung.
        """
        if text is None or pd.isna(text) or not str(text).strip():
            return {
                "teks_asli": "" if (text is None or pd.isna(text)) else str(text),
                "total_aspek_terdeteksi": 0,
                "aspek_terdeteksi": {},
                "ringkasan_sentimen": {}
            }

        text_str = str(text)
        clauses = self.segment_clauses(text_str)
        aspect_clauses: Dict[str, List[str]] = {asp: [] for asp in ASPECT_TAXONOMY.keys()}

        # 1. Kelompokkan klausa berdasarkan aspek yang terkandung
        for clause in clauses:
            detected_aspects = self.identify_aspects_in_clause(clause)
            for asp in detected_aspects:
                aspect_clauses[asp].append(clause)

        # 2. Jika ada aspek yang terdeteksi di teks utuh namun klausa pendek tidak tertangkap
        for aspect_key, aspect_data in ASPECT_TAXONOMY.items():
            if not aspect_clauses[aspect_key]:
                # Cek apakah teks keseluruhan menyebutkan aspek
                if self.identify_aspects_in_clause(text_str).intersection({aspect_key}):
                    aspect_clauses[aspect_key].append(text_str)

        # 3. Hitung sentimen untuk setiap aspek yang memiliki klausa
        aspect_results: Dict[str, Dict[str, Any]] = {}
        aspect_summary: Dict[str, str] = {}

        for aspect_key, matched_clauses in aspect_clauses.items():
            if not matched_clauses:
                continue

            combined_clause_text = ". ".join(matched_clauses)
            pred = self.predictor.predict_one(combined_clause_text)

            aspect_info = ASPECT_TAXONOMY[aspect_key]
            aspect_results[aspect_key] = {
                "nama_aspek": aspect_info["display_name"],
                "sentimen": pred["sentimen"],
                "keyakinan": pred["keyakinan"],
                "probabilitas": pred.get("probabilitas", {}),
                "klausa_pendukung": matched_clauses
            }
            aspect_summary[aspect_key] = pred["sentimen"]

        return {
            "teks_asli": text,
            "total_aspek_terdeteksi": len(aspect_results),
            "aspek_terdeteksi": aspect_results,
            "ringkasan_sentimen": aspect_summary
        }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Melakukan Aspect-Based Sentiment Analysis pada seluruh baris DataFrame.
        Menambahkan kolom sentimen per aspek:
        - 'aspek_pengiriman'
        - 'aspek_kualitas_produk'
        - 'aspek_harga_promo'
        - 'aspek_pelayanan_cs'
        - 'aspek_aplikasi_sistem'
        """
        if df.empty:
            result_df = df.copy()
            for asp in ASPECT_TAXONOMY.keys():
                result_df[f"aspek_{asp}"] = []
            return result_df

        col = text_column or SentimentPredictor.detect_text_column(df)
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan dalam DataFrame. Kolom tersedia: {list(df.columns)}")
        result_df = df.copy()

        aspect_columns = {f"aspek_{asp}": [] for asp in ASPECT_TAXONOMY.keys()}

        for text in result_df[col].fillna("").astype(str):
            res = self.analyze_aspects(text)
            summary = res.get("ringkasan_sentimen", {})
            for asp in ASPECT_TAXONOMY.keys():
                # Nilai: 'positif', 'negatif', 'netral', atau 'tidak_disebut'
                aspect_columns[f"aspek_{asp}"].append(summary.get(asp, "tidak_disebut"))

        for col_name, values in aspect_columns.items():
            result_df[col_name] = values

        return result_df
