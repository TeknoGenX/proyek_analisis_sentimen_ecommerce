# Proyek Analisis Sentimen E-Commerce Indonesia

Proyek ini adalah implementasi *End-to-End Machine Learning Pipeline* (NLP) untuk menganalisis sentimen pengguna aplikasi E-Commerce di Indonesia (Studi Kasus: Tokopedia/Shopee) berdasarkan ulasan di Google Play Store.

## 📌 Fitur Utama
1. **Automated Data Scraper**: Mengambil ulasan asli pengguna secara langsung dari Play Store.
2. **Optimized Preprocessing**: Menggunakan Sastrawi dengan sistem *Caching* untuk mempercepat proses *stemming* bahasa Indonesia hingga 10x lipat.
3. **Imbalanced Data Handling**: Menggunakan bobot kelas otomatis (`class_weight='balanced'`) pada Logistic Regression untuk memastikan ulasan negatif (keluhan) tetap terdeteksi dengan baik.
4. **Automated Business Visualizations**: Menghasilkan *WordCloud* keluhan (*Pain Points*) dan *Confusion Matrix*.
5. **Model Export**: Menyimpan model AI (`model.pkl`) dan *vectorizer* (`vectorizer.pkl`) agar bisa diintegrasikan ke API atau aplikasi web di masa depan.
6. **Live Testing Mode**: Anda bisa mengetikkan kalimat ulasan baru di terminal untuk melihat tebakan AI secara *real-time*.

## 🚀 Cara Menjalankan Proyek (Linux / Kali Linux / Ubuntu)

**1. Persiapan Lingkungan**
```bash
# Buat folder proyek
mkdir sentimen_ecommerce && cd sentimen_ecommerce

# Buat dan aktifkan Virtual Environment
python3 -m venv venv
source venv/bin/activate
```

**2. Instalasi Dependensi**
```bash
pip install -r requirements.txt
```

**3. Jalankan Pipeline Utama**
```bash
python sentiment_pipeline.py
```

## 📊 Output Proyek
Setelah skrip selesai dijalankan, Anda akan mendapatkan file berikut di dalam folder Anda:
- `dataset_ulasan.csv` (Dataset mentah hasil scraping)
- `WordCloud_Keluhan.png` (Visualisasi kata terbanyak pada sentimen negatif)
- `ConfusionMatrix.png` (Visualisasi evaluasi model)
- `model_sentimen.pkl` & `tfidf_vectorizer.pkl` (File model yang siap di-deploy)