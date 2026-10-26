# 🔬 Modul Riset & Skrip Eksperimental (Legacy Research)

Folder ini mengumpulkan skrip riset dan eksperimen awal dari proyek sentimen analisis e-commerce:

## 📁 Daftar Skrip Eksperimen

1. **`script_utama_nlp.py`**:
   - Skrip monolitik awal yang menggabungkan scraping Tokopedia, preprocessing Sastrawi sederhana, TF-IDF, Logistic Regression, dan WordCloud dalam satu berkas sekuensial.
   - **Status Refaktor Modular**: Telah dipecah dan dimodularisasi ke dalam:
     - `src/scraper.py` (Scraping Play Store multi-platform)
     - `src/preprocessor.py` (Pembersihan, slang dictionary, persistent caching)
     - `src/features.py` (Vektorisasi TF-IDF)
     - `src/model_trainer.py` (Benchmarking multi-model)
     - `src/visualizer.py` (Heatmap & WordCloud)

2. **`sentiment_pipeline.py` & `sentiment_pipeline1.py`**:
   - Skrip pipeline awal untuk pengujian cepat ulasan Play Store.
   - **Status Refaktor Modular**: Ditransformasi menjadi CLI terpadu `main.py` dan inferensi real-time/batch di `src/predictor.py`.

3. **`sentiment_research.py`**:
   - Eksperimen klasifikasi sentimen menggunakan Deep Learning & Transfer Learning model embedding NNLM Bahasa Indonesia (`https://tfhub.dev/google/nnlm-id-dim128/2`) via TensorFlow Hub.
   - **Status Refaktor Modular**: Diintegrasikan secara modular ke dalam `src/deep_learning.py`.
