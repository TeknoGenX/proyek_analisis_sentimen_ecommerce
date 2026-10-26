# 🏛️ Arsitektur Modular & Laporan Analisis Bug Proyek E-Commerce NLP

Dokumen ini menjelaskan struktur arsitektur modular proyek Analisis Sentimen E-Commerce Indonesia, pemetaan modul terpadu dari skrip legacy, serta laporan komprehensif audit dan perbaikan bug (*bug analysis*).

---

## 📂 1. Peta Struktur Folder & Modularitas Proyek

Semua komponen yang sebelumnya tersebar di berbagai folder dan skrip monolitik kini dikonsolidasikan ke dalam struktur proyek yang rapi, terisolasi, dan mudah dianalisis:

```
proyek_analisis_sentimen_ecommerce/
├── app.py                            # Entry point Web Dashboard (Streamlit)
├── main.py                           # CLI Unified Controller (11 Subcommands)
├── config.py                         # Konfigurasi terpusat (paths, hiperparameter, 7 platform)
├── requirements.txt                  # Daftar dependensi pustaka Python
├── src/                              # Modul Inti Produksi (Modular Pipeline)
│   ├── __init__.py
│   ├── scraper.py                    # 1. Scraper Play Store (7 platform + filter transaksi TikTok Shop)
│   ├── preprocessor.py               # 2. Text Normalizer, Slang Dictionary, Caching Sastrawi
│   ├── features.py                   # 3. TF-IDF Extractor (Unigram + Bigram, Sublinear TF)
│   ├── model_trainer.py              # 4. Multi-Model ML Benchmarking (LR, CNB, LinearSVC)
│   ├── predictor.py                  # 5. Inference Engine (Single, Dataframe, Batch CSV/Excel)
│   ├── aspect_analyzer.py            # 6. Aspect-Based Sentiment Analysis (ABSA 5 aspek utama)
│   ├── visualizer.py                 # 7. Chart Generator (Confusion Matrix, NSS, WordClouds)
│   ├── deep_learning.py              # 8. Transfer Learning & Neural Network NNLM ID
│   ├── report_generator.py           # 9. Executive HTML & Print-Ready PDF Report Generator
│   ├── api.py                        # 10. Production REST API (FastAPI + Swagger UI)
│   └── dashboard.py                  # 11. Interactive Web Dashboard UI (Streamlit)
├── data/
│   ├── raw/                          # Dataset ulasan mentah hasil scraping
│   │   ├── dataset_ulasan.csv
│   │   └── dataset_komparasi_ecommerce.csv
│   ├── processed/                    # Dataset bersih siap latih
│   │   ├── dataset_cleaned.csv
│   │   └── dataset_komparasi_cleaned.csv
│   └── slang_dict.json               # Kamus normalisasi bahasa gaul & singkatan e-commerce
├── models/                           # Serialisasi model ML, vectorizer, dan cache
│   ├── model_sentimen.pkl            # Model klasifikasi terbaik (LinearSVC Balanced)
│   ├── tfidf_vectorizer.pkl          # Kosakata TF-IDF terlatih
│   ├── stem_cache.json               # Persistent cache Sastrawi (2.460+ entri)
│   ├── evaluation_metrics.json       # Metrik benchmark model
│   └── best_sentiment_model.keras    # Model deep learning biner (NNLM ID)
├── outputs/                          # Visualisasi dan laporan output pipeline
│   ├── platform_wordclouds/          # WordCloud ulasan positif & keluhan per platform
│   ├── komparasi_metrics.json        # Metrik komparatif 7 platform
│   ├── komparasi_nss_platform.png    # Grafik benchmark NSS
│   ├── komparasi_sentimen_platform.png# Diagram distribusi komparatif
│   ├── komparasi_wordcloud_keluhan.png# Panel grid perbandingan keluhan
│   └── laporan_eksekutif_sentimen.html# Laporan eksekutif siap cetak ke PDF
├── tests/                            # Unit test suite lengkap (60 tests terverifikasi)
│   ├── test_preprocessor.py          # Pengujian pembersihan, slang, dan stemming
│   ├── test_features.py              # Pengujian ekstraksi fitur TF-IDF
│   ├── test_model_trainer.py         # Pengujian benchmarking model klasifikasi
│   ├── test_predictor.py             # Pengujian inferensi tunggal, dataframe, file batch
│   ├── test_aspect_analyzer.py       # Pengujian segmentasi klausa dan ABSA
│   ├── test_scraper.py               # Pengujian scraping, filter TikTok, error handling
│   ├── test_visualizer.py            # Pengujian diagram, matriks, wordcloud
│   ├── test_report_generator.py      # Pengujian generator laporan eksekutif HTML
│   ├── test_api.py                   # Pengujian endpoint FastAPI
│   └── test_dashboard.py             # Pengujian modul antarmuka Streamlit
├── research/                         # Arsip skrip eksperimen monolitik awal
│   ├── script_utama_nlp.py
│   ├── sentiment_pipeline.py
│   ├── sentiment_pipeline1.py
│   └── sentiment_research.py
└── docs/                             # Dokumentasi proyek
    ├── dokumentasi_proyek_e_commerce.md
    └── ARSITEKTUR_MODULAR.md
```

---

## 🔍 2. Laporan Analisis Bug & Solusi Perbaikan

Berikut adalah rincian bug yang berhasil diidentifikasi, dianalisis akar masalahnya (*root cause*), dan diperbaiki:

### Bug 1: Crash `TypeError` pada Aspect-Based Sentiment Analysis (`src/aspect_analyzer.py`)
- **Gejala / Error**: `TypeError: expected string or bytes-like object, got 'float'` saat teks ulasan bernilai `np.nan`, `None`, atau angka.
- **Akar Masalah**: Pengecekan `if not text or not str(text).strip():` gagal menangani `np.nan` (float NaN) karena `not np.nan` bernilai `False` dan `str(np.nan)` bernilai `'nan'`. Akibatnya, `CLAUSE_SPLIT_REGEX.split(text)` dipanggil dengan objek float.
- **Solusi**:
  - Menambahkan validasi `if text is None or pd.isna(text) or not str(text).strip(): return ...`.
  - Mengonversi `text_str = str(text)` sebelum eksekusi regex.
- **Verifikasi**: Ditambahkan `test_analyze_aspects_nan_and_none` di [test_aspect_analyzer.py](file:///home/andiliani/code/proyek_analisis_sentimen_ecommerce/tests/test_aspect_analyzer.py). Status: **PASSED**.

### Bug 2: Hilangnya Hasil Batch Saat Interaksi Filter & Download (`src/dashboard.py`)
- **Gejala / Error**: Ketika pengguna mengklik radio filter sentimen atau tombol unduh CSV/Excel, seluruh hasil prediksi batch seketika lenyap dari layar.
- **Akar Masalah**: Siklus *rerun* Streamlit menyebabkan `if st.button("⚡ Mulai Analisis Batch"):` bernilai `False` pada interaksi widget turunan di dalamnya.
- **Solusi**:
  - Menyimpan DataFrame hasil ke dalam `st.session_state["batch_result_df"]`.
  - Merender kartu KPI, grafik, radio filter, dan tombol unduh di luar blok `if st.button(...)` dengan membaca `st.session_state`.
- **Verifikasi**: Dashboard mempertahankan hasil prediksi saat pengguna beralih opsi filter atau mengunduh file.

### Bug 3: Inkonsistensi Skema DataFrame Kosong pada Prediktor (`src/predictor.py`)
- **Gejala / Error**: Pemanggilan `predict_dataframe(df_kosong)` menghasilkan DataFrame tanpa kolom probabilitas (`prob_negatif`, `prob_netral`, `prob_positif`), berbeda dengan DataFrame normal.
- **Akar Masalah**: Blok `if df.empty:` hanya mendefinisikan kolom `clean_review`, `prediksi_sentimen`, dan `keyakinan`.
- **Solusi**: Menambahkan inisialisasi kolom `result_df[f"prob_{cls_name}"] = []` untuk seluruh kelas model saat input kosong.
- **Verifikasi**: Ditambahkan pengujian kolom probabilitas pada DataFrame kosong di [test_predictor.py](file:///home/andiliani/code/proyek_analisis_sentimen_ecommerce/tests/test_predictor.py). Status: **PASSED**.

### Bug 4: Potensi Crash `TypeError` Sort & Format pada Laporan Eksekutif (`src/report_generator.py`)
- **Gejala / Error**: `TypeError: '<' not supported between instances of 'int' and 'NoneType'` atau `unsupported format string passed to NoneType` jika metrik platform berisi nilai `None` atau `NaN`.
- **Akar Masalah**: Penggunaan `x.get("nss", 0)` mengembalikan `None` jika key `"nss"` ada dengan nilai `None`.
- **Solusi**:
  - Menggunakan fungsi pembanding yang aman: `key=lambda x: float(x.get("nss")) if (x.get("nss") is not None and not pd.isna(x.get("nss"))) else 0.0`.
  - Memastikan nilai fallback 0.0 sebelum string formatting `:+0.1f`.
- **Verifikasi**: Ditambahkan `test_generate_html_report_with_none_and_nan_metrics` di [test_report_generator.py](file:///home/andiliani/code/proyek_analisis_sentimen_ecommerce/tests/test_report_generator.py). Status: **PASSED**.

### Bug 5: Nilai Rating `NaN` pada Metrik Visualizer (`src/visualizer.py`)
- **Gejala / Error**: Jika dataset platform memiliki rating kosong/NaN seluruhnya, `calculate_platform_metrics` menghasilkan `rata_rata_rating: NaN`. Hal ini memicu penulisan literal non-standar `NaN` pada file JSON dan teks `"⭐ nan"` pada laporan HTML.
- **Akar Masalah**: `sub_df["rating"].mean()` menghasilkan `np.nan` jika semua baris NaN.
- **Solusi**: Menambahkan pengecekan: `if pd.isna(avg_rating): avg_rating = 0.0`.
- **Verifikasi**: Ditambahkan `test_platform_metrics_with_nan_ratings` di [test_visualizer.py](file:///home/andiliani/code/proyek_analisis_sentimen_ecommerce/tests/test_visualizer.py). Status: **PASSED**.

### Bug 6: Crash UnicodeDecodeError pada Input CSV File (`src/predictor.py`)
- **Gejala / Error**: File CSV ekspor Excel dengan encoding Windows (`cp1252`, `latin-1`) atau UTF-8 BOM melempar error saat dibaca melalui CLI `predict-batch`.
- **Akar Masalah**: `pd.read_csv` default hanya mengasumsikan UTF-8 murni.
- **Solusi**: Menambahkan mekanisme *cascading encoding fallback* (`utf-8` -> `utf-8-sig` -> `latin-1`).
- **Verifikasi**: Berhasil membaca file CSV dengan berbagai encoding karakter.

---

## 🚀 3. Panduan Menjalankan Sistem

```bash
# 1. Menjalankan seluruh pengujian (60 Unit Tests)
./venv/bin/python3 -m unittest discover tests

# 2. Menjalankan Web Dashboard Interaktif
./venv/bin/python3 main.py dashboard

# 3. Menjalankan REST API Server (Swagger UI di http://localhost:8000/docs)
./venv/bin/python3 main.py serve

# 4. Generate Laporan Eksekutif HTML & Siap Cetak PDF
./venv/bin/python3 main.py report

# 5. Uji Prediksi Satu Kalimat
./venv/bin/python3 main.py predict -t "Barang mantap kualitas jempolan!"

# 6. Uji Analisis Aspek (ABSA)
./venv/bin/python3 main.py aspect -t "Pengiriman cepat tapi barang rusak."
```
