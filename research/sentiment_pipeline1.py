import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras  # Wajib menggunakan tf_keras untuk Keras 2
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PERSIAPAN DATASET E-COMMERCE INDONESIA
# ==========================================
data = {
    'ulasan': [
        "Barangnya bagus banget, pengiriman super cepat. Mantap!",
        "Seller tidak responsif, barang cacat pas sampai.",
        "Kurir ramah, packing aman pakai bubble wrap tebal.",
        "Sangat kecewa, ukurannya tidak sesuai dengan deskripsi produk.",
        "Harga murah tapi kualitas lumayan oke lah buat sehari-hari.",
        "Aplikasi sering error saat mau checkout, tolong diperbaiki.",
        "Produk original, garansi resmi. Puas belanja di toko ini.",
        "Pengiriman lama banget, nyesel beli pakai bebas ongkir."
    ],
    'sentimen': [1, 0, 1, 0, 1, 0, 1, 0] # 1 = Positif, 0 = Negatif
}
df = pd.DataFrame(data)

# Memisahkan data pelatihan (80%) dan validasi (20%)
X_train, X_val, y_train, y_val = train_test_split(
    df['ulasan'],
    df['sentimen'],
    test_size=0.2,  
    random_state=42
)

X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# ==========================================
# 2. PEMBANGUNAN MODEL (NNLM BAHASA INDONESIA)
# ==========================================
def build_sentiment_model():
    print("Memuat NNLM Bahasa Indonesia dari TF Hub...")
    # Menggunakan NNLM ID yang menggunakan operasi bawaan TensorFlow (tanpa tensorflow-text)
    module_url = "https://tfhub.dev/google/nnlm-id-dim128/2"
    
    # Layer ekstraksi fitur teks otomatis
    hub_layer = hub.KerasLayer(
        module_url,
        input_shape=[], 
        dtype=tf.string,
        trainable=False, 
        name="NNLM_ID_Embedding"
    )

    # PERHATIAN: Semua arsitektur layer di bawah ini MUTLAK menggunakan `keras` dari `tf_keras`
    model = keras.models.Sequential([
        hub_layer,
        keras.layers.Dense(128, activation='relu', name='Dense_128'),
        keras.layers.Dropout(0.3, name='Dropout_1'),
        keras.layers.Dense(64, activation='relu', name='Dense_64'),
        keras.layers.Dropout(0.2, name='Dropout_2'),
        keras.layers.Dense(1, activation='sigmoid', name='Output_Layer')
    ], name="Sentimen_ECommerce_Model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name='akurasi')]
    )
    
    return model

# ==========================================
# 3. EKSEKUSI PELATIHAN & PREDIKSI
# ==========================================
if __name__ == "__main__":
    print("=== MEMULAI MODE PELATIHAN ===")
    model = build_sentiment_model()
    model.summary()

    # Memulai pelatihan
    history = model.fit(
        X_train,
        y_train,
        epochs=10, 
        batch_size=2, 
        validation_data=(X_val, y_val),
        verbose=1
    )

    def prediksi_sentimen(model_terlatih, daftar_ulasan_baru):
        print("\n=== MEMULAI MODE PREDIKSI ===")
        
        prediksi_probabilitas = model_terlatih.predict(daftar_ulasan_baru)
        
        hasil_prediksi = []
        for teks, prob in zip(daftar_ulasan_baru, prediksi_probabilitas):
            kelas = "Positif" if prob[0] >= 0.5 else "Negatif"
            skor_keyakinan = prob[0] if kelas == "Positif" else (1 - prob[0])
            
            hasil_prediksi.append({
                "Ulasan": teks,
                "Sentimen": kelas,
                "Keyakinan": f"{skor_keyakinan * 100:.2f}%"
            })
        
        df_hasil = pd.DataFrame(hasil_prediksi)
        print(df_hasil.to_string(index=False))

    # Menguji model dengan data ulasan baru
    ulasan_baru = [
        "Barang sampai dengan selamat, kurirnya juga cepat banget. Recommended seller!",
        "Beli pas flash sale tapi barang yang dikirim salah warna, retur ribet.",
        "Bintang lima pokoknya, kualitas sesuai dengan harga."
    ]

    prediksi_sentimen(model, ulasan_baru)