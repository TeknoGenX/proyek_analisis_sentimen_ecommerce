# Proyek Analisis Sentimen E-Commerce Indonesia (End-to-End NLP)

Proyek ini adalah sistem **Analisis Sentimen Natural Language Processing (NLP) tingkat produksi** yang dirancang khusus untuk memproses dan menganalisis ulasan pengguna aplikasi E-Commerce di Indonesia (seperti Tokopedia, Shopee, Blibli, Lazada, dll.) dari Google Play Store.

Proyek ini telah direfaktor dan dioptimalkan dari kode dasar sebelumnya menjadi arsitektur modular, scalable, dan siap pakai (*production-ready*).

---

## 🌟 Keunggulan & Fitur Utama

1. **Automated Play Store Scraper (`src/scraper.py`)**:
   - Pengambilan ulasan pengguna secara langsung (*real-time*) menggunakan Google Play Scraper.
   - Penanganan error jaringan cerdas dengan fallback otomatis ke dataset lokal cadangan jika offline.
   - Pelabelan sentimen otomatis berdasarkan rating bintang (1-2: Negatif, 3: Netral, 4-5: Positif).

2. **Advanced Indonesian Text Preprocessor (`src/preprocessor.py`)**:
   - **Emoji to Sentiment Mapping**: Mengubah emotikon umum (misal: 👍, 😭, 😡) menjadi representasi sentimen tekstual sebelum pembersihan simbol.
   - **Slang & Typo Normalizer (`data/slang_dict.json`)**: Normalisasi bahasa gaul, singkatan e-commerce, dan typo khas ulasan Indonesia (contoh: *bgt* $\rightarrow$ *banget*, *bapuk* $\rightarrow$ *buruk*, *pdhl* $\rightarrow$ *padahal*, *gabisa* $\rightarrow$ *tidak bisa*).
   - **Repeated Character Reducer**: Mengurangi pemanjangan karakter (contoh: *mantaaaaap* $\rightarrow$ *mantap*, *rusakkk* $\rightarrow$ *rusak*).
   - **Negation Preservation Stopword Removal**: Mempertahankan kata-kata negasi krusial seperti *tidak*, *kurang*, *jangan*, *bukan* agar konteks sentimen tidak terbalik.
   - **Persistent Disk Caching (`models/stem_cache.json`)**: Cache hasil stemming Sastrawi langsung disimpan ke file JSON di disk, sehingga eksekusi kedua dan seterusnya berjalan dalam hitungan detik (10x-50x lebih cepat).

3. **TF-IDF N-Gram Feature Extraction (`src/features.py`)**:
   - Mendukung Unigram dan Bigram `(1, 2)` untuk mengenali frasa negasi seperti *"tidak bagus"*, *"kurang ramah"*.
   - Sublinear Term Frequency untuk mereduksi dominasi frekuensi kata ekstrem.

4. **Multi-Model Benchmarking & Auto-Selection (`src/model_trainer.py`)**:
   - Membandingkan model secara objektif:
     - **Logistic Regression (Class-Weighted Balanced)**
     - **Complement Naive Bayes (Optimasi untuk Imbalanced Text)**
     - **Linear Support Vector Classifier (LinearSVC)**
   - Menggunakan 5-Fold Stratified Cross-Validation dan pengujian pada hold-out test set.
   - Otomatis memilih dan mengekspor model dengan performa F1-Score terbaik.

5. **Business Insights & Visualisasi Otomatis (`src/visualizer.py`)**:
   - **Confusion Matrix Heatmap**: Mengukur ketepatan klasifikasi per kategori.
   - **Customer Pain Points WordCloud**: Menganalisis kata kunci keluhan pada ulasan negatif untuk kebutuhan tim produk/CS.
   - **Customer Delight WordCloud**: Menganalisis hal-hal yang disukai pengguna pada ulasan positif.
   - **Sentiment Distribution Chart**: Visualisasi proporsi sentimen.

6. **Inference Engine & Interactive CLI (`src/predictor.py` & `main.py`)**:
   - Menyediakan API inferensi tunggal dan batch dengan skor keyakinan (*confidence score*).
   - Mode CLI interaktif dengan tampilan antarmuka visual terminal menggunakan `Rich`.

7. **Deep Learning Transfer Learning (`src/deep_learning.py`)**:
   - Modul eksperimen lanjutan menggunakan TensorFlow Hub `NNLM Bahasa Indonesia (128 dimensi)` untuk klasifikasi sentimen berbasis deep learning.

---

## 📁 Struktur Direktori

```text
proyek_analisis_sentimen_ecommerce/
│
├── config.py                     # Konfigurasi terpusat (path, hyperparameter, label)
├── main.py                       # CLI utama aplikasi
├── requirements.txt              # Daftar dependensi library
├── README.md                     # Dokumentasi lengkap proyek
│
├── data/
│   ├── raw/
│   │   └── dataset_ulasan.csv    # Dataset mentah hasil scraping Play Store
│   ├── processed/
│   │   └── dataset_cleaned.csv   # Dataset hasil pembersihan & normalisasi
│   └── slang_dict.json           # Kamus normalisasi kata gaul & slang e-commerce
│
├── src/
│   ├── __init__.py
│   ├── scraper.py                # Modul scraper Google Play Store
│   ├── preprocessor.py           # Preprocessing teks Indonesia + disk cache
│   ├── features.py               # Ekstraksi fitur TF-IDF (Unigram & Bigram)
│   ├── model_trainer.py          # Benchmarking & pelatihan model ML
│   ├── visualizer.py             # Visualisasi Confusion Matrix & WordCloud
│   ├── predictor.py              # Engine inferensi & prediksi real-time
│   └── deep_learning.py          # Modul Deep Learning NNLM ID (TensorFlow)
│
├── models/
│   ├── model_sentimen.pkl        # File bobot model ML terbaik
│   ├── tfidf_vectorizer.pkl      # File kosakata TF-IDF
│   ├── stem_cache.json           # File cache persistent kata Sastrawi
│   └── evaluation_metrics.json   # Ringkasan metrik akurasi & F1-Score
│
├── outputs/
│   ├── ConfusionMatrix.png       # Grafik Confusion Matrix
│   ├── WordCloud_Keluhan.png     # WordCloud ulasan negatif (Pain Points)
│   ├── WordCloud_Positif.png     # WordCloud ulasan positif
│   └── distribusi_sentimen.png   # Diagram proporsi sentimen
│
└── tests/
    └── test_preprocessor.py      # Unit test pengujian modul preprocessing
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Persiapan Environment
Gunakan virtual environment yang sudah ada atau buat baru:
```bash
# Masuk ke folder proyek
cd /home/andiliani/code/nlp_sentimen_app/proyek_analisis_sentimen_ecommerce

# Aktifkan virtual environment
source ../venv/bin/activate

# Install dependensi (jika belum)
pip install -r requirements.txt
```

### 2. Menjalankan Seluruh Pipeline Sekaligus
Cukup jalankan perintah berikut untuk menjalankan *scraping* (atau load dataset), *preprocessing*, *benchmarking*, *visualisasi*, dan uji prediksi:
```bash
python main.py run-all
```

### 3. Menjalankan Modul Per Tahap (Sesuai Kebutuhan)

- **Scraping Ulasan Baru dari Play Store**:
  ```bash
  python main.py scrape --app com.tokopedia.tkpd --count 800
  ```
  *(Dapat diganti dengan `com.shopee.id` atau aplikasi lain)*

- **Pra-Pemrosesan Teks Saja**:
  ```bash
  python main.py preprocess
  ```

- **Pelatihan Model & Benchmarking Saja**:
  ```bash
  python main.py train
  ```

- **Menghasilkan Visualisasi Saja**:
  ```bash
  python main.py evaluate
  ```

- **Prediksi Cepat Satu Teks Ulasan**:
  ```bash
  python main.py predict -t "Barangnya rusak parah, seller dichat gak balas sama sekali!"
  ```

- **Mode Interaktif Terminal (Live Chatbot Testing)**:
  ```bash
  python main.py interactive
  ```

---

## 🧪 Menjalankan Unit Tests
Uji keandalan modul pembersihan teks dan kamus slang:
```bash
python -m unittest tests/test_preprocessor.py
```

---

## 📊 Integrasi API / Kode Python Mandiri
Anda dapat memanggil predictor model langsung di script Python lain:
```python
from src.predictor import SentimentPredictor

predictor = SentimentPredictor()
hasil = predictor.predict_one("Bagus banget pengiriman cepat dan barang original!")

print("Sentimen:", hasil["sentimen"])
print("Keyakinan:", hasil["keyakinan"], "%")
print("Rincian Probabilitas:", hasil["probabilitas"])
```
