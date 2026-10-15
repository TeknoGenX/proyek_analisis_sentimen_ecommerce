"""
Modul Pelatihan dan Evaluasi Model Machine Learning.
Mendukung multi-model benchmarking:
- Logistic Regression (Balanced)
- Complement Naive Bayes (Cocok untuk text imbalanced)
- Linear Support Vector Classifier (LinearSVC Balanced)
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import config

logger = logging.getLogger(__name__)


class SentimentModelTrainer:
    def __init__(self, model_save_path: Optional[Path] = None):
        self.model_save_path = model_save_path or config.MODEL_PATH
        self.best_model = None
        self.best_model_name = ""
        self.evaluation_results = {}

    def get_candidate_models(self) -> Dict[str, Any]:
        """Mengembalikan daftar model kandidat untuk benchmarking."""
        return {
            "Logistic Regression (Balanced)": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_STATE
            ),
            "Complement Naive Bayes": ComplementNB(
                alpha=0.5
            ),
            "Linear SVC (Balanced)": LinearSVC(
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
                max_iter=2000
            )
        }

    def train_and_benchmark(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        labels=config.LABELS
    ) -> Dict[str, Any]:
        """
        Melatih beberapa model kandidat, mengevaluasi dengan cross-validation dan test set,
        lalu memilih model dengan performa terbaik.
        """
        candidates = self.get_candidate_models()
        best_f1 = -1.0

        # Validasi variasi kelas data latih
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Jumlah kelas unik pada data latih terlalu sedikit ({len(unique_classes)} kelas: {list(unique_classes)}). "
                f"Model klasifikasi memerlukan minimal 2 kelas berbeda untuk dilatih."
            )

        # Tentukan jumlah split K-Fold secara dinamis dan aman
        class_counts = pd.Series(y_train).value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        n_splits = min(5, min_class_count, len(y_train))
        use_cv = n_splits >= 2

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE) if use_cv else None

        print("\n" + "="*60)
        print(" MEMULAI BENCHMARKING MODEL MACHINE LEARNING ".center(60, "="))
        print("="*60)

        for name, model in candidates.items():
            # Cross Validation jika data sampel mencukupi
            if use_cv:
                cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")
                cv_mean = float(np.mean(cv_scores))
            else:
                logger.warning(
                    f"Jumlah sampel per kelas ({min_class_count}) tidak cukup untuk K-Fold CV. Cross-validation dilewati."
                )
                cv_scores = []
                cv_mean = 0.0

            # Fit on training data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            report = classification_report(y_test, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

            self.evaluation_results[name] = {
                "cv_f1_weighted_mean": round(cv_mean, 4),
                "test_accuracy": round(acc, 4),
                "test_f1_weighted": round(f1_weighted, 4),
                "test_f1_macro": round(f1_macro, 4),
                "classification_report": report,
                "confusion_matrix": cm
            }

            cv_text = f"{cv_mean * 100:.2f}%" if use_cv else "N/A (sampel terlalu sedikit)"
            fold_label = f"{n_splits}-Fold" if use_cv else "Skipped"
            print(f"\n[Model] {name}")
            print(f"  ├─ CV F1-Weighted ({fold_label}) : {cv_text}")
            print(f"  ├─ Test Accuracy           : {acc * 100:.2f}%")
            print(f"  └─ Test F1-Score (Weighted): {f1_weighted * 100:.2f}%")

            # Pilih model terbaik berdasarkan F1-Score Weighted pada data uji
            if f1_weighted > best_f1:
                best_f1 = f1_weighted
                self.best_model = model
                self.best_model_name = name

        print("\n" + "="*60)
        print(f" MODEL TERBAIK TERPILIH: {self.best_model_name} (F1: {best_f1*100:.2f}%) ")
        print("="*60)

        # Print classification report detail untuk model terbaik
        best_preds = self.best_model.predict(X_test)
        print("\nLaporan Klasifikasi Lengkap Model Terbaik:")
        print(classification_report(y_test, best_preds, labels=labels, target_names=labels, zero_division=0))

        return self.evaluation_results

    def save_best_model(self, path: Optional[Path] = None) -> None:
        """Menyimpan model terbaik dan ringkasan metrik."""
        if self.best_model is None:
            raise ValueError("Belum ada model yang dilatih. Jalankan train_and_benchmark terlebih dahulu.")

        save_path = path or self.model_save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, save_path)
        logger.info(f"Model terbaik ({self.best_model_name}) berhasil disimpan ke: {save_path}")

        # Simpan evaluasi metrik ke JSON
        metrics_payload = {
            "best_model_name": self.best_model_name,
            "benchmark_results": self.evaluation_results
        }
        with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrik evaluasi disimpan ke: {config.METRICS_PATH}")

    def load_model(self, path: Optional[Path] = None):
        """Memuat model tersimpan."""
        load_path = path or self.model_save_path
        if not load_path.exists():
            raise FileNotFoundError(f"File model tidak ditemukan di: {load_path}")
        self.best_model = joblib.load(load_path)
        logger.info(f"Model berhasil dimuat dari: {load_path}")
        return self.best_model
