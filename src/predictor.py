"""
Modul Inferensi Sentimen untuk Prediksi Teks Real-Time / Batch.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

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
        slang_path: Optional[Path] = None
    ):
        self.preprocessor = IndonesianTextPreprocessor(slang_path=slang_path)
        
        self.feature_extractor = FeatureExtractor(vectorizer_path=vectorizer_path)
        self.vectorizer = self.feature_extractor.load_vectorizer()

        trainer = SentimentModelTrainer(model_save_path=model_path)
        self.model = trainer.load_model()
        self.classes_ = getattr(self.model, "classes_", np.array(config.LABELS))

    def predict_one(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Melakukan inferensi sentimen pada satu teks ulasan.
        
        Returns:
            Dict dengan kunci:
            - teks_asli: Ulasan awal
            - teks_bersih: Hasil pra-pemrosesan
            - sentimen: Label hasil prediksi ('positif' / 'netral' / 'negatif')
            - keyakinan: Persentase probabilitas model terhadap kelas terpilih
            - probabilitas: Rincian probabilitas tiap kelas (jika didukung)
        """
        if not text or not str(text).strip():
            return {
                "teks_asli": text,
                "teks_bersih": "",
                "sentimen": "netral",
                "keyakinan": 0.0,
                "probabilitas": {}
            }

        # 1. Pra-pemrosesan teks
        cleaned_text = self.preprocessor.clean(text)
        if not cleaned_text:
            cleaned_text = text.lower().strip()

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
            confidence = probs_dict.get(prediction, max(probs))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(vec)[0]
            # Softmax sederhana untuk mengubah skor margin ke probabilitas
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            for cls_name, prob_val in zip(self.classes_, probs):
                probs_dict[cls_name] = round(float(prob_val), 4)
            confidence = probs_dict.get(prediction, max(probs))

        return {
            "teks_asli": text,
            "teks_bersih": cleaned_text,
            "sentimen": prediction,
            "keyakinan": round(float(confidence) * 100, 2),
            "probabilitas": probs_dict
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """Melakukan inferensi pada sekumpulan teks sekaligus."""
        return [self.predict_one(t) for t in texts]
