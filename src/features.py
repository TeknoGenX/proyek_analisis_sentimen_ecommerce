"""
Modul Ekstraksi Fitur Teks menggunakan TF-IDF.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(
        self,
        max_features: int = config.TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = config.TFIDF_NGRAM_RANGE,
        sublinear_tf: bool = config.TFIDF_SUBLINEAR_TF,
        vectorizer_path: Optional[Path] = None
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
        self.vectorizer: Optional[TfidfVectorizer] = None

    def fit_transform(self, texts: pd.Series):
        """Membangun kosakata TF-IDF dan mengubah teks pelatihan menjadi matriks fitur."""
        logger.info(
            f"Fitting TF-IDF Vectorizer (ngram_range={self.ngram_range}, "
            f"max_features={self.max_features}, sublinear_tf={self.sublinear_tf})..."
        )
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf
        )
        features = self.vectorizer.fit_transform(texts)
        logger.info(f"Ekstraksi fitur selesai. Bentuk matriks: {features.shape}")
        return features

    def transform(self, texts: pd.Series):
        """Mentransformasikan teks baru menggunakan kosakata yang sudah dilatih."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer belum dilatih atau dimuat! Panggil fit_transform atau load_vectorizer terlebih dahulu.")
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, path: Optional[Path] = None) -> None:
        """Menyimpan vectorizer ke file joblib."""
        save_path = path or self.vectorizer_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, save_path)
        logger.info(f"Vectorizer disimpan ke: {save_path}")

    def load_vectorizer(self, path: Optional[Path] = None) -> TfidfVectorizer:
        """Memuat vectorizer dari file joblib."""
        load_path = path or self.vectorizer_path
        if not load_path.exists():
            raise FileNotFoundError(f"File vectorizer tidak ditemukan di: {load_path}")
        self.vectorizer = joblib.load(load_path)
        logger.info(f"Vectorizer berhasil dimuat dari: {load_path}")
        return self.vectorizer
