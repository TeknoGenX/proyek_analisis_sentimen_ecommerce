"""
Modul Pra-Pemrosesan Teks Bahasa Indonesia Khusus E-Commerce.
Mencakup:
1. Normalisasi Simbol, Emoji, URL, & Angka
2. Reduksi Karakter Berulang (e.g., 'baguuus' -> 'bagus')
3. Kamus Slang / Bahasa Gaul E-Commerce
4. Stopword Removal dengan Preservasi Kata Negasi (e.g., 'tidak', 'kurang')
5. Stemming Sastrawi dengan Persistent Disk Cache (JSON)
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import config

logger = logging.getLogger(__name__)

# Pemetaan emoji ke kata sentimen sebelum penghapusan karakter khusus
EMOJI_SENTIMENT_MAP = {
    "👍": " bagus ",
    "⭐": " bagus ",
    "❤️": " suka ",
    "🥰": " suka ",
    "🔥": " mantap ",
    "👎": " buruk ",
    "😭": " kecewa ",
    "😡": " marah ",
    "🤬": " marah ",
    "🥺": " kecewa ",
    "🤮": " jelek "
}


class IndonesianTextPreprocessor:
    def __init__(
        self,
        slang_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        preserve_negations: bool = True
    ):
        self.slang_path = slang_path or config.SLANG_DICT_PATH
        self.cache_path = cache_path or config.STEM_CACHE_PATH
        self.preserve_negations = preserve_negations

        # Memuat kamus slang dan memisahkan frasa multi-kata dan kata tunggal
        self.slang_dict: Dict[str, str] = self._load_slang_dict()
        self.phrase_slang: Dict[str, str] = {
            k: v for k, v in sorted(self.slang_dict.items(), key=lambda x: len(x[0]), reverse=True)
            if " " in k
        }
        self.single_slang: Dict[str, str] = {
            k: v for k, v in self.slang_dict.items() if " " not in k
        }

        # Inisialisasi Sastrawi
        stem_factory = StemmerFactory()
        self.stemmer = stem_factory.create_stemmer()

        # Inisialisasi Stopword Remover
        stop_factory = StopWordRemoverFactory()
        default_stopwords: List[str] = stop_factory.get_stop_words()
        
        if self.preserve_negations:
            # Hapus kata negasi dari daftar stopword agar tetap dipertahankan
            self.stopwords: Set[str] = {
                word for word in default_stopwords 
                if word not in config.PRESERVED_NEGATIONS
            }
        else:
            self.stopwords = set(default_stopwords)

        # Memuat cache stemming dari disk jika tersedia
        self.stem_cache: Dict[str, str] = self._load_cache()
        self.new_cache_entries = 0

    def _load_slang_dict(self) -> Dict[str, str]:
        """Memuat kamus slang dari file JSON."""
        if self.slang_path.exists():
            try:
                with open(self.slang_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Gagal membaca slang_dict.json: {e}")
        return {}

    def _load_cache(self) -> Dict[str, str]:
        """Memuat persistent cache stemming dari file JSON."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    logger.info(f"Loaded {len(cache)} kata dari persistent stem cache.")
                    return cache
            except Exception as e:
                logger.warning(f"Gagal membaca cache stemming: {e}")
        return {}

    def save_cache(self) -> None:
        """Menyimpan pembaruan cache stemming ke file JSON di disk."""
        if self.new_cache_entries > 0 or not self.cache_path.exists():
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.stem_cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Persistent stem cache tersimpan ({len(self.stem_cache)} kata).")
                self.new_cache_entries = 0
            except Exception as e:
                logger.error(f"Gagal menyimpan cache stemming: {e}")

    @staticmethod
    def reduce_repeated_characters(text: str) -> str:
        """
        Mereduksi karakter berulang lebih dari 2 kali menjadi 1 karakter.
        Contoh: 'mantaaaaap' -> 'mantap', 'baguuuus' -> 'bagus'.
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def normalize_slang(self, text: str) -> str:
        """Mengganti kata slang, singkatan, dan frasa e-commerce dengan kata baku."""
        # 1. Normalisasi frasa (multi-word slang yang mengandung spasi)
        for phrase, replacement in self.phrase_slang.items():
            if phrase in text:
                text = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text)

        # 2. Normalisasi kata tunggal
        words = text.split()
        normalized_words = [self.single_slang.get(word, word) for word in words]
        return " ".join(normalized_words)

    def remove_stopwords(self, text: str) -> str:
        """Menghapus stopword tetapi mempertahankan kata-kata negasi."""
        words = text.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        return " ".join(filtered_words)

    def stem_text(self, text: str) -> str:
        """
        Melakukan stemming Sastrawi yang dioptimalkan dengan cache persistensi.
        """
        words = text.split()
        stemmed_tokens = []
        for word in words:
            if word in self.stem_cache:
                stemmed_tokens.append(self.stem_cache[word])
            else:
                stemmed = self.stemmer.stem(word)
                self.stem_cache[word] = stemmed
                self.new_cache_entries += 1
                stemmed_tokens.append(stemmed)
        return " ".join(stemmed_tokens)

    def clean(self, text: str, do_stemming: bool = True) -> str:
        """
        Pipeline lengkap pra-pemrosesan teks ulasan bahasa Indonesia.
        
        Tahapan:
        1. Konversi huruf kecil (case folding)
        2. Pemetaan emoji ke kata sentimen
        3. Pembersihan URL, mention, hashtag, dan karakter non-alfabet
        4. Reduksi huruf berulang
        5. Normalisasi slang / singkatan
        6. Penghapusan stopword (menjaga negasi)
        7. Stemming Sastrawi (dengan caching)
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Pemetaan emoji
        for emoji, word in EMOJI_SENTIMENT_MAP.items():
            if emoji in text:
                text = text.replace(emoji, word)

        # 3. Hapus URL, Mention, Hashtags, HTML Tags
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+|#\w+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)

        # Hapus karakter selain a-z dan spasi
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Normalisasi spasi
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. Reduksi karakter berulang (e.g. mantaaaap -> mantap)
        text = self.reduce_repeated_characters(text)

        # 5. Normalisasi slang e-commerce
        text = self.normalize_slang(text)

        # 6. Stopword removal (dengan menjaga kata negasi)
        text = self.remove_stopwords(text)

        # 7. Stemming
        if do_stemming:
            text = self.stem_text(text)

        # Final whitespace trim
        text = re.sub(r'\s+', ' ', text).strip()
        return text
