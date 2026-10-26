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

7. **Aspect-Based Sentiment Analysis / ABSA (`src/aspect_analyzer.py`)**:
   - Pemecahan klausa cerdas berbasis kata hubung (*tapi, namun, padahal, dan*).
   - Klasifikasi sentimen tersendiri untuk 5 aspek e-commerce utama: *Pengiriman, Kualitas Produk, Harga & Promo, Pelayanan CS, Aplikasi & Sistem*.

8. **Production REST API Server (`src/api.py`)**:
   - Dibangun dengan FastAPI dan Uvicorn untuk melayani endpoint inferensi real-time, prediksi batch, ABSA, dan OpenAPI Swagger UI interaktif di browser.

9. **Interactive Web Dashboard (`app.py` & `src/dashboard.py`)**:
   - Dashboard analitik berbasis Streamlit untuk pengujian ulasan real-time, ABSA, batch file uploader (CSV/Excel), visualisasi distribusi sentimen, benchmark komparasi 7 platform, pratinjau laporan eksekutif, dan metrik model ML.

10. **Automated Executive Business Report Generator (`src/report_generator.py`)**:
    - Generator laporan bisnis analitik berformat HTML interaktif yang mandiri (*self-contained* dengan grafik Base64 tersemat).
    - Dilengkapi *executive summary*, perbandingan KPI, peringkat NSS, analisis 5 aspek ABSA, dan *print-to-PDF friendly CSS* (`@media print`).

11. **Deep Learning Transfer Learning (`src/deep_learning.py`)**:
    - Modul eksperimen lanjutan menggunakan TensorFlow Hub `NNLM Bahasa Indonesia (128 dimensi)` untuk klasifikasi sentimen berbasis deep learning.

---

## 📁 Struktur Direktori

```text
proyek_analisis_sentimen_ecommerce/
│
├── app.py                        # Entrypoint Web Dashboard (Streamlit)
├── config.py                     # Konfigurasi terpusat (path, hyperparameter, label)
├── main.py                       # CLI utama aplikasi (11 subcommands)
├── requirements.txt              # Daftar dependensi library
├── README.md                     # Dokumentasi lengkap proyek
│
├── data/
│   ├── raw/
│   │   ├── dataset_ulasan.csv    # Dataset mentah ulasan Play Store
│   │   └── dataset_komparasi_ecommerce.csv # Dataset 7 platform
│   ├── processed/
│   │   ├── dataset_cleaned.csv   # Dataset hasil pembersihan & normalisasi
│   │   └── dataset_komparasi_cleaned.csv
│   └── slang_dict.json           # Kamus normalisasi kata gaul & slang e-commerce
│
├── src/
│   ├── __init__.py
│   ├── api.py                    # REST API Server FastAPI (Swagger UI)
│   ├── aspect_analyzer.py        # Aspect-Based Sentiment Analysis (ABSA)
│   ├── dashboard.py              # Logika UI Web Dashboard (Streamlit)
│   ├── deep_learning.py          # Modul Deep Learning NNLM ID (TensorFlow)
│   ├── features.py               # Ekstraksi fitur TF-IDF (Unigram & Bigram)
│   ├── model_trainer.py          # Benchmarking & pelatihan model ML
│   ├── predictor.py              # Engine inferensi & prediksi real-time
│   ├── preprocessor.py           # Preprocessing teks Indonesia + disk cache
│   ├── report_generator.py       # Generator Laporan Bisnis Eksekutif (HTML/PDF)
│   ├── scraper.py                # Scraper Play Store (multi-platform & filter TikTok)
│   └── visualizer.py             # Visualisasi Confusion Matrix, WordCloud, NSS
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
│   ├── distribusi_sentimen.png   # Diagram proporsi sentimen
│   ├── komparasi_nss_platform.png # Grafik komparasi NSS 7 platform
│   ├── komparasi_sentimen_platform.png # Komparasi proporsi sentimen
│   ├── komparasi_wordcloud_keluhan.png # Grid WordCloud komparatif
│   └── laporan_eksekutif_sentimen.html # Laporan Analitik Bisnis Eksekutif
│
└── tests/
    ├── test_api.py               # Unit test REST API FastAPI
    ├── test_aspect_analyzer.py   # Unit test Aspect-Based Sentiment Analysis
    ├── test_dashboard.py         # Unit test Web Dashboard Streamlit
    ├── test_features.py          # Unit test ekstraksi fitur TF-IDF
    ├── test_model_trainer.py     # Unit test pelatihan model ML
    ├── test_predictor.py         # Unit test inference engine
    ├── test_preprocessor.py      # Unit test pembersihan & slang normalizer
    ├── test_report_generator.py  # Unit test generator laporan bisnis
    ├── test_scraper.py           # Unit test scraping & fallback
    └── test_visualizer.py        # Unit test grafik & WordCloud
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Persiapan Environment
Gunakan virtual environment yang sudah ada atau buat baru:
```bash
# Masuk ke folder proyek
cd /home/andiliani/code/proyek_analisis_sentimen_ecommerce

# Aktifkan virtual environment
source venv/bin/activate

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

- **Analisis Sentimen Berbasis Aspek (ABSA)**:
  ```bash
  python main.py aspect -t "Barang original bagus, tapi kurirnya lambat pengiriman telat."
  ```

- **Prediksi Massal / Batch Inference (CSV / Excel)**:
  ```bash
  # Prediksi massal dan ekspor hasil ke file CSV
  python main.py predict-batch -i data/raw/dataset_ulasan.csv -o outputs/hasil_prediksi.csv

  # Dapat juga menggunakan file Excel (.xlsx) dan menentukan kolom ulasan secara spesifik
  python main.py predict-batch -i ulasan.xlsx -o outputs/hasil_prediksi.xlsx -c ulasan_customer
  ```

- **Mode Interaktif Terminal (Live Chatbot Testing)**:
  ```bash
  python main.py interactive
  ```

- **Menjalankan Web Dashboard Interaktif (Streamlit)**:
  ```bash
  python main.py dashboard
  # atau langsung dengan:
  streamlit run app.py
  ```
  Akses di browser: [http://localhost:8501](http://localhost:8501)
  - Fitur: Uji sentimen real-time, ABSA, batch uploader (CSV/Excel), visualisasi distribusi grafik, benchmark 7 platform, pratinjau & download laporan eksekutif, dan ringkasan metrik model ML.

- **Menghasilkan Laporan Analitik Bisnis Eksekutif (HTML / PDF)**:
  ```bash
  # Generate laporan HTML mandiri (dengan visualisasi grafik tersemat)
  python main.py report

  # Generate dan langsung buka otomatis di browser
  python main.py report --open
  ```
  *File laporan tersimpan di: `outputs/laporan_eksekutif_sentimen.html` (Dapat dicetak ke PDF dengan tombol cetak atau `Ctrl + P`).*

- **Menjalankan REST API Server (FastAPI + Swagger UI)**:
  ```bash
  python main.py serve --port 8000
  ```
  Akses di browser:
  - **Swagger UI (Interactive Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)
  - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
  - **Healthcheck**: `GET http://localhost:8000/health`
  - **Single Prediction**: `POST http://localhost:8000/predict` (Body: `{"text": "..."}`)
  - **Batch Prediction**: `POST http://localhost:8000/predict-batch` (Body: `{"texts": ["...", "..."]}`)
  - **Aspect Sentiment**: `POST http://localhost:8000/analyze-aspects` (Body: `{"text": "..."}`)
  - **Model Metrics**: `GET http://localhost:8000/metrics`
  - **Platform Benchmarks**: `GET http://localhost:8000/platforms`
  - **Executive Report HTML**: `GET http://localhost:8000/report/html`
  - **Generate Report**: `POST http://localhost:8000/report/generate`

---

## 🧪 Menjalankan Unit Tests
Jalankan seluruh test suite otomatis:
```bash
python -m unittest discover tests
```
Atau jalankan modul pengujian spesifik:
```bash
python -m unittest tests/test_preprocessor.py
python -m unittest tests/test_predictor.py
python -m unittest tests/test_aspect_analyzer.py
python -m unittest tests/test_dashboard.py
python -m unittest tests/test_report_generator.py
python -m unittest tests/test_api.py
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

# Atau untuk prediksi massal satu file sekaligus:
hasil_df, saved_path = predictor.predict_file("input_ulasan.csv", "outputs/hasil.csv")
print(f"Prediksi selesai! Disimpan ke: {saved_path}")
```

