import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras  
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PERSIAPAN DATASET PENELITIAN
# ==========================================
csv_path = "dataset_ulasan1.csv"

# Blok ini akan mencoba membaca dataset CSV asli.
# Jika belum ada, script akan otomatis membuat data simulasi yang lebih besar.
if os.path.exists(csv_path):
    print(f"Membaca dataset dari {csv_path}...")
    df = pd.read_csv(csv_path)
    # Asumsi CSV memiliki kolom 'ulasan' dan 'sentimen' (0 = Negatif, 1 = Positif)
else:
    print(f"File {csv_path} tidak ditemukan. Membuat dataset simulasi...")
    # Dataset simulasi diperbesar agar model bisa melihat variasi
    ulasan_positif = [
        "Barangnya bagus banget, pengiriman super cepat. Mantap!",
        "Produk original, garansi resmi. Puas belanja di toko ini.",
        "Kurir ramah, packing aman pakai bubble wrap tebal.",
        "Bintang lima pokoknya, kualitas sesuai dengan harga.",
        "Kualitas produk sangat baik, original dan murah.",
        "Suka banget sama barangnya, sesuai ekspektasi.",
        "Packing rapi dan aman, seller juga ramah dan fast respon.",
        "Mantap, berfungsi dengan baik tanpa cacat."
    ] * 15  # Digandakan untuk simulasi

    ulasan_negatif = [
        "Seller tidak responsif, barang cacat pas sampai.",
        "Sangat kecewa, ukurannya tidak sesuai dengan deskripsi.",
        "Aplikasi sering error saat mau checkout, tolong diperbaiki.",
        "Pengiriman lama banget, nyesel beli pakai bebas ongkir.",
        "Barang palsu, tidak sesuai gambar di etalase.",
        "Barang rusak tidak bisa dipakai, rugi beli di sini.",
        "Pelayanan buruk, chat tidak pernah dibalas.",
        "Packing asal-asalan, kardusnya penyok semua."
    ] * 15  # Digandakan untuk simulasi

    data = {
        'ulasan': ulasan_positif + ulasan_negatif,
        'sentimen': [1]*len(ulasan_positif) + [0]*len(ulasan_negatif)
    }
    df = pd.DataFrame(data)
    # df.to_csv(csv_path, index=False) # Opsional: simpan simulasi ini ke CSV

X_train, X_val, y_train, y_val = train_test_split(
    df['ulasan'], df['sentimen'], test_size=0.2, random_state=42, stratify=df['sentimen']
)
X_train, X_val, y_train, y_val = map(np.array, [X_train, X_val, y_train, y_val])

# ==========================================
# 2. ARSITEKTUR MODEL (NNLM BAHASA INDONESIA)
# ==========================================
def build_sentiment_model():
    print("\nMemuat NNLM Bahasa Indonesia dari TF Hub...")
    module_url = "https://tfhub.dev/google/nnlm-id-dim128/2"
    hub_layer = hub.KerasLayer(module_url, input_shape=[], dtype=tf.string, trainable=False)

    model = keras.models.Sequential([
        hub_layer,
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4), # Ditingkatkan dari 0.3 untuk mencegah overfitting
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Learning rate diturunkan sedikit agar model belajar lebih stabil
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name='akurasi')]
    )
    return model

# ==========================================
# 3. PELATIHAN DENGAN CALLBACKS (RESEARCH MODE)
# ==========================================
if __name__ == "__main__":
    model = build_sentiment_model()
    
    # CALLBACK 1: Early Stopping (Berhenti jika val_loss tidak membaik setelah 5 epoch)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    # CALLBACK 2: Model Checkpoint (Otomatis menyimpan bobot model terbaik)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='best_sentiment_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    print("\n=== MEMULAI PELATIHAN MODEL ===")
    history = model.fit(
        X_train, y_train, 
        epochs=30,           # Max epoch diperbesar, EarlyStopping yang akan menghentikan
        batch_size=16,       # Batch size disesuaikan untuk data yang lebih besar
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # ==========================================
    # 4. EVALUASI DAN VISUALISASI
    # ==========================================
    print("\n=== MENYIMPAN GRAFIK PELATIHAN ===")
    plt.figure(figsize=(12, 4))

    # Grafik Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['akurasi'], label='Akurasi Training')
    plt.plot(history.history['val_akurasi'], label='Akurasi Validasi')
    plt.title('Metrik Akurasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()

    # Grafik Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss Training')
    plt.plot(history.history['val_loss'], label='Loss Validasi')
    plt.title('Metrik Loss (Kerugian)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('grafik_pelatihan.png')
    print("Grafik disimpan sebagai 'grafik_pelatihan.png'. Silakan buka file tersebut untuk menganalisis performa model.")

    # Model siap diuji atau dimuat ulang (tf_keras.models.load_model) di API Django/Backend.