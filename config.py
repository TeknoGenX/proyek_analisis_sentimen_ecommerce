"""
Modul Konfigurasi Terpusat untuk Proyek Analisis Sentimen E-Commerce Indonesia.
Mendukung analisis komparasi 7 platform: Shopee, Tokopedia, Lazada, Blibli,
TikTok Shop (Shop | Tokopedia), Bukalapak, dan Zalora.
"""
from pathlib import Path

# Direktori Utama
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# File Paths
RAW_DATASET_PATH = RAW_DATA_DIR / "dataset_ulasan.csv"
COMPARISON_RAW_DATASET_PATH = RAW_DATA_DIR / "dataset_komparasi_ecommerce.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "dataset_cleaned.csv"
COMPARISON_PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "dataset_komparasi_cleaned.csv"

SLANG_DICT_PATH = DATA_DIR / "slang_dict.json"
MODEL_PATH = MODELS_DIR / "model_sentimen.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
STEM_CACHE_PATH = MODELS_DIR / "stem_cache.json"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Visualisasi Paths
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "ConfusionMatrix.png"
WORDCLOUD_NEGATIF_PATH = OUTPUTS_DIR / "WordCloud_Keluhan.png"
WORDCLOUD_POSITIF_PATH = OUTPUTS_DIR / "WordCloud_Positif.png"
DISTRIBUTION_CHART_PATH = OUTPUTS_DIR / "distribusi_sentimen.png"
COMPARISON_SENTIMENT_PATH = OUTPUTS_DIR / "komparasi_sentimen_platform.png"
COMPARISON_NSS_PATH = OUTPUTS_DIR / "komparasi_nss_platform.png"
COMPARISON_WORDCLOUD_GRID_PATH = OUTPUTS_DIR / "komparasi_wordcloud_keluhan.png"
PLATFORM_WORDCLOUDS_DIR = OUTPUTS_DIR / "platform_wordclouds"
COMPARISON_METRICS_PATH = OUTPUTS_DIR / "komparasi_metrics.json"

# Konfigurasi Platform E-Commerce Target
ECOMMERCE_PLATFORMS = {
    "shopee": {
        "name": "Shopee",
        "app_id": "com.shopee.id",
        "category": "general_marketplace",
        "description": "Marketplace massal, promo gratis ongkir, live stream"
    },
    "tokopedia": {
        "name": "Tokopedia",
        "app_id": "com.tokopedia.tkpd",
        "category": "general_marketplace",
        "description": "Ekosistem belanja lengkap, produk elektronik & official store"
    },
    "lazada": {
        "name": "Lazada",
        "app_id": "com.lazada.android",
        "category": "general_marketplace",
        "description": "LazMall, kampanye diskon mega, logistik in-house LEX"
    },
    "blibli": {
        "name": "Blibli",
        "app_id": "blibli.mobile.commerce",
        "category": "general_marketplace",
        "description": "Customer service 24/7 unggulan, jaminan 100% orisinil, Blibli Tiket"
    },
    "tiktok_shop": {
        "name": "TikTok Shop (Shop | Tokopedia)",
        "app_id": "com.zhiliaoapp.musically",
        "category": "social_commerce",
        "description": "Social commerce, video-driven shopping, live streaming"
    },
    "bukalapak": {
        "name": "Bukalapak",
        "app_id": "com.bukalapak.android",
        "category": "general_marketplace",
        "description": "Mitra O2O, produk hobi/gaming, voucher & marketplace komunitas"
    },
    "zalora": {
        "name": "Zalora",
        "app_id": "com.zalora.android",
        "category": "fashion_beauty",
        "description": "Spesialis fashion & kecantikan, 100% original, retur 30 hari"
    }
}

# Parameter Scraping
DEFAULT_PLATFORM = "tokopedia"
DEFAULT_APP_ID = ECOMMERCE_PLATFORMS[DEFAULT_PLATFORM]["app_id"]
DEFAULT_SCRAPE_COUNT = 800
DEFAULT_COMPARISON_SCRAPE_COUNT = 300
SCRAPE_LANG = "id"
SCRAPE_COUNTRY = "id"
SCRAPE_MAX_WORKERS = 4

# Kata Kunci Filter Khusus Transaksi / Belanja TikTok Shop
TIKTOK_COMMERCE_KEYWORDS = [
    "shop", "toko", "beli", "membeli", "pembelian", "belanja", "pesan", "pesanan",
    "barang", "paket", "kurir", "kirim", "dikirim", "pengiriman", "ongkir",
    "seller", "penjual", "checkout", "cekout", "keranjang", "kuning", "keranjang kuning",
    "order", "orderan", "resi", "cod", "bayar", "pembayaran", "refund", "retur",
    "pengembalian", "produk", "jual", "jualan", "diskon", "promo", "voucher",
    "batal", "rusak", "cacat", "harga", "kualitas", "baju", "celana", "size",
    "admin", "dana", "saldo", "uang", "ori", "asli", "palsu", "kaos", "sepatu",
    "tas", "skincare", "makeup", "live shopping"
]

# Label Sentimen
LABELS = ["negatif", "netral", "positif"]

# Parameter Ekstraksi Fitur (TF-IDF)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigram & Bigram untuk menangkap konteks frasa seperti 'tidak bagus'
TFIDF_SUBLINEAR_TF = True

# Parameter Pemodelan
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Negation words yang TIDAK BOLEH dihapus oleh Stopword Remover
PRESERVED_NEGATIONS = {
    "tidak", "bukan", "kurang", "jangan", "belum", "tanpa", 
    "tidaklah", "bukanlah", "tak", "tiada"
}
