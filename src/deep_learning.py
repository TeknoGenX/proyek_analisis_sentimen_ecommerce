"""
Modul Deep Learning (Opsional / Penelitian Lanjutan) untuk Analisis Sentimen E-Commerce.
Menggunakan TensorFlow / tf_keras dan Transfer Learning NNLM ID dari TF Hub.
"""
import logging
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)


def train_deep_learning_model(
    epochs: int = 15,
    batch_size: int = 16,
    output_model_path: Optional[Path] = None,
    plot_path: Optional[Path] = None
):
    """
    Melatih model Deep Learning menggunakan Transfer Learning NNLM Bahasa Indonesia.
    """
    try:
        import tensorflow as tf
        import tf_keras as keras
        import tensorflow_hub as hub
    except ImportError as e:
        logger.error(f"Pustaka Deep Learning (tensorflow / tf_keras / tensorflow_hub) belum terpasang: {e}")
        return

    # Muat dataset bersih
    if not config.PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset belum siap di: {config.PROCESSED_DATASET_PATH}. Jalankan preprocess terlebih dahulu.")

    df = pd.read_csv(config.PROCESSED_DATASET_PATH).dropna(subset=["clean_review", "label"])
    
    # Filter biner (positif vs negatif) untuk model NNLM biner
    df_binary = df[df["label"].isin(["positif", "negatif"])].copy()
    if len(df_binary) < 4 or df_binary["label"].nunique() < 2:
        logger.warning("Jumlah data biner tidak mencukupi untuk melatih model Deep Learning.")
        return

    df_binary["target"] = df_binary["label"].apply(lambda x: 1 if x == "positif" else 0)

    class_counts = df_binary["target"].value_counts()
    can_stratify = class_counts.min() >= 2
    stratify_arg = df_binary["target"].values if can_stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        df_binary["clean_review"].values,
        df_binary["target"].values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_arg
    )

    print("\n" + "=" * 60)
    print(" MEMULAI PELATIHAN DEEP LEARNING (NNLM BAHASA INDONESIA) ")
    print("=" * 60)
    print(f"Total data latih: {len(X_train)} | Total data validasi: {len(X_val)}")

    module_url = "https://tfhub.dev/google/nnlm-id-dim128/2"
    logger.info(f"Memuat embedding NNLM dari: {module_url}")
    
    hub_layer = hub.KerasLayer(
        module_url,
        input_shape=[],
        dtype=tf.string,
        trainable=False,
        name="NNLM_ID_Embedding"
    )

    model = keras.models.Sequential([
        hub_layer,
        keras.layers.Dense(128, activation="relu", name="Dense_128"),
        keras.layers.Dropout(0.4, name="Dropout_1"),
        keras.layers.Dense(64, activation="relu", name="Dense_64"),
        keras.layers.Dropout(0.3, name="Dropout_2"),
        keras.layers.Dense(1, activation="sigmoid", name="Output_Layer")
    ], name="Sentimen_ECommerce_DeepLearning")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="akurasi")]
    )

    model.summary()

    save_model_file = output_model_path or (config.MODELS_DIR / "deep_learning_sentiment.keras")
    save_model_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(save_model_file),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Simpan kurva pelatihan
    target_plot = plot_path or (config.OUTPUTS_DIR / "grafik_pelatihan_deep_learning.png")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["akurasi"], label="Train Akurasi")
    plt.plot(history.history["val_akurasi"], label="Val Akurasi")
    plt.title("Akurasi Model Deep Learning")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Model Deep Learning")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(target_plot, dpi=300)
    plt.close()
    print(f"\n[✓] Model Deep Learning tersimpan di: {save_model_file}")
    print(f"[✓] Grafik evaluasi Deep Learning tersimpan di: {target_plot}")


if __name__ == "__main__":
    train_deep_learning_model()
