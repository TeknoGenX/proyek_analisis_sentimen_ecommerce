"""
Modul Inferensi Sentimen untuk Prediksi Teks Real-Time / Batch.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import config
from src.preprocessor import IndonesianTextPreprocessor
from src.features import FeatureExtractor
from src.model_trainer import SentimentModelTrainer

logger = logging.getLogger(__name__)


class SentimentPredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        slang_path: Optional[Path] = None,
        neutral_threshold: float = 0.0,
        binary_mode: bool = False
    ):
        self.preprocessor = IndonesianTextPreprocessor(slang_path=slang_path)
        
        self.feature_extractor = FeatureExtractor(vectorizer_path=vectorizer_path)
        self.vectorizer = self.feature_extractor.load_vectorizer()

        trainer = SentimentModelTrainer(model_save_path=model_path)
        self.model = trainer.load_model()
        self.classes_ = getattr(self.model, "classes_", np.array(config.LABELS))
        self.neutral_threshold = neutral_threshold
        self.binary_mode = binary_mode

    def predict_one(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Melakukan inferensi sentimen pada satu teks ulasan.
        Mendukung opsi klasifikasi biner dan ambang batas keyakinan (confidence threshold) untuk netral.
        
        Returns:
            Dict dengan kunci:
            - teks_asli: Ulasan awal
            - teks_bersih: Hasil pra-pemrosesan
            - sentimen: Label hasil prediksi ('positif' / 'netral' / 'negatif')
            - keyakinan: Persentase probabilitas model terhadap kelas terpilih
            - probabilitas: Rincian probabilitas tiap kelas (jika didukung)
        """
        if text is None or pd.isna(text) or not str(text).strip():
            return {
                "teks_asli": "" if (text is None or pd.isna(text)) else str(text),
                "teks_bersih": "",
                "sentimen": "netral",
                "keyakinan": 0.0,
                "probabilitas": {}
            }

        # 1. Pra-pemrosesan teks
        cleaned_text = self.preprocessor.clean(text)
        if not cleaned_text or not any(c.isalnum() for c in cleaned_text):
            return {
                "teks_asli": text,
                "teks_bersih": cleaned_text,
                "sentimen": "netral",
                "keyakinan": 0.0,
                "probabilitas": {}
            }

        # 2. Vektorisasi TF-IDF
        vec = self.vectorizer.transform([cleaned_text])

        # 3. Prediksi model
        prediction = self.model.predict(vec)[0]

        # 4. Hitung probabilitas jika model mendukung
        probs_dict = {}
        confidence = 1.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            for cls_name, prob_val in zip(self.classes_, probs):
                probs_dict[cls_name] = round(float(prob_val), 4)
            confidence = probs_dict.get(prediction, float(max(probs)))
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(vec)
            # Kasus biner pada scikit-learn (LinearSVC): decision_function memiliki shape (n_samples,)
            if raw_scores.ndim == 1 or (raw_scores.ndim == 2 and raw_scores.shape[1] == 1):
                score = float(raw_scores.ravel()[0])
                # Fungsi Sigmoid untuk probabilitas 2 kelas
                p1 = 1.0 / (1.0 + np.exp(-score))
                p0 = 1.0 - p1
                probs = [p0, p1]
                classes = list(self.classes_[:2]) if len(self.classes_) >= 2 else ["negatif", "positif"]
                for cls_name, prob_val in zip(classes, probs):
                    probs_dict[cls_name] = round(float(prob_val), 4)
                confidence = probs_dict.get(prediction, float(max(probs)))
            else:
                # Kasus multiclass: shape (n_samples, n_classes)
                scores = raw_scores[0]
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
                for cls_name, prob_val in zip(self.classes_, probs):
                    probs_dict[cls_name] = round(float(prob_val), 4)
                confidence = probs_dict.get(prediction, float(max(probs)))

        # Opsi 1: Mode Biner (fokus murni Keluhan vs Pujian)
        if self.binary_mode and "positif" in probs_dict and "negatif" in probs_dict:
            pos_p = probs_dict.get("positif", 0.0)
            neg_p = probs_dict.get("negatif", 0.0)
            total_bin = pos_p + neg_p
            if total_bin > 0:
                p_pos_norm = pos_p / total_bin
                p_neg_norm = neg_p / total_bin
            else:
                p_pos_norm, p_neg_norm = 0.5, 0.5

            if p_pos_norm >= p_neg_norm:
                prediction = "positif"
                confidence = p_pos_norm
            else:
                prediction = "negatif"
                confidence = p_neg_norm

        # Opsi 2: Algorithmic Neutral Thresholding (keyakinan rendah < threshold -> netral)
        conf_pct = round(float(confidence) * 100, 2)
        if not self.binary_mode and self.neutral_threshold > 0.0:
            if conf_pct < self.neutral_threshold:
                prediction = "netral"

        return {
            "teks_asli": text,
            "teks_bersih": cleaned_text,
            "sentimen": prediction,
            "keyakinan": conf_pct,
            "probabilitas": probs_dict
        }

    def predict_batch(self, texts: Union[List[str], pd.Series, str]) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """Melakukan inferensi pada sekumpulan teks sekaligus."""
        if isinstance(texts, str):
            texts = [texts]
        return [self.predict_one(t) for t in texts]

    @staticmethod
    def detect_text_column(df: pd.DataFrame) -> str:
        """Mendeteksi secara otomatis nama kolom teks ulasan dalam DataFrame."""
        candidate_cols = [
            "teks_ulasan", "clean_review", "content", "review", "ulasan",
            "text", "komentar", "comment", "feedback"
        ]
        for col in candidate_cols:
            if col in df.columns:
                return col
        for col in df.columns:
            if any(cand == col.lower() or cand in col.lower() for cand in candidate_cols):
                return col
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(str_cols) > 0:
            return str_cols[0]
        raise KeyError(
            f"Tidak dapat menemukan kolom teks ulasan dalam file. "
            f"Kolom tersedia: {list(df.columns)}. "
            f"Gunakan parameter '--text-column' untuk menentukan nama kolom."
        )

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Melakukan prediksi sentimen pada seluruh baris dalam DataFrame.
        Menambahkan kolom: 'clean_review', 'prediksi_sentimen', 'keyakinan', dan 'prob_*'.
        """
        if df.empty:
            result_df = df.copy()
            result_df["clean_review"] = []
            result_df["prediksi_sentimen"] = []
            result_df["keyakinan"] = []
            for cls_name in self.classes_:
                result_df[f"prob_{cls_name}"] = []
            return result_df

        col = text_column or self.detect_text_column(df)
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan dalam DataFrame. Kolom tersedia: {list(df.columns)}")

        result_df = df.copy()
        raw_texts = result_df[col].fillna("").astype(str).tolist()

        clean_texts = []
        predictions = []
        confidences = []
        prob_details = {cls_name: [] for cls_name in self.classes_}

        for text in raw_texts:
            res = self.predict_one(text)
            clean_texts.append(res["teks_bersih"])
            predictions.append(res["sentimen"])
            confidences.append(res["keyakinan"])
            probs = res.get("probabilitas", {})
            for cls_name in self.classes_:
                prob_details[cls_name].append(probs.get(cls_name, 0.0))

        result_df["clean_review"] = clean_texts
        result_df["prediksi_sentimen"] = predictions
        result_df["keyakinan"] = confidences
        for cls_name in self.classes_:
            result_df[f"prob_{cls_name}"] = prob_details[cls_name]

        return result_df

    def predict_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        text_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Membaca ulasan dari file CSV atau Excel, melakukan prediksi sentimen batch,
        dan mengekspor hasilnya ke file output (CSV atau Excel).
        """
        in_p = Path(input_path)
        if not in_p.exists():
            raise FileNotFoundError(f"File input tidak ditemukan: {in_p}")

        suffix = in_p.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(in_p)
        else:
            try:
                df = pd.read_csv(in_p)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(in_p, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(in_p, encoding="latin-1")

        result_df = self.predict_dataframe(df, text_column=text_column)

        if output_path:
            out_p = Path(output_path)
        else:
            out_p = config.OUTPUTS_DIR / f"prediksi_{in_p.stem}.csv"

        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() in [".xlsx", ".xls"]:
            result_df.to_excel(out_p, index=False)
        else:
            result_df.to_csv(out_p, index=False)

        logger.info(f"Hasil prediksi batch ({len(result_df)} baris) berhasil disimpan ke: {out_p}")
        return result_df, out_p

    def analyze_aspects(self, text: str) -> Dict[str, Any]:
        """
        Menganalisis sentimen per aspek (ABSA) pada satu teks ulasan.
        Mengembalikan sentimen untuk pengiriman, kualitas produk, harga/promo, CS, dan aplikasi.
        """
        from src.aspect_analyzer import AspectSentimentAnalyzer
        analyzer = AspectSentimentAnalyzer(predictor=self)
        return analyzer.analyze_aspects(text)

