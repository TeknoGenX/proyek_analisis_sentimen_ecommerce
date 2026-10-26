"""
Dashboard Web Interaktif untuk Analisis Sentimen E-Commerce Indonesia.
Dibangun menggunakan Streamlit.
"""
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.predictor import SentimentPredictor
from src.aspect_analyzer import AspectSentimentAnalyzer, ASPECT_TAXONOMY

logger = logging.getLogger(__name__)


# =========================================================================
# Inisialisasi & Cache Model
# =========================================================================
@st.cache_resource(show_spinner="Memuat model NLP & kosakata TF-IDF...")
def load_predictor() -> Optional[SentimentPredictor]:
    """Memuat instance SentimentPredictor dengan caching Streamlit."""
    if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
        return None
    try:
        return SentimentPredictor()
    except Exception as e:
        logger.error(f"Gagal menginisialisasi SentimentPredictor: {e}")
        return None


@st.cache_resource(show_spinner="Memuat penganalisis aspek (ABSA)...")
def load_aspect_analyzer(_predictor: SentimentPredictor) -> AspectSentimentAnalyzer:
    """Memuat instance AspectSentimentAnalyzer."""
    return AspectSentimentAnalyzer(predictor=_predictor)


# =========================================================================
# Helper Visual & Format
# =========================================================================
SENTIMENT_COLORS = {
    "positif": "#28a745",
    "netral": "#ffc107",
    "negatif": "#dc3545"
}

SENTIMENT_BADGES = {
    "positif": "🟢 POSITIF",
    "netral": "🟡 NETRAL",
    "negatif": "🔴 NEGATIF"
}

PRESET_EXAMPLES = {
    "Pilih Contoh Ulasan...": "",
    "Positif & Multi-Aspek (Produk Bagus + Pengiriman Cepat)":
        "Barang original sesuai deskripsi, packing rapi bubble wrap tebal dan pengiriman super cepat! Mantap seller!",
    "Keluhan Pengiriman & Kurir (Pain Point Logistik)":
        "Kurirnya lelet banget, paket sudah seminggu belum sampai dan status resi gak jelas!",
    "Masalah Aplikasi & Checkout (Pain Point Sistem)":
        "Aplikasi sering crash dan error pas mau checkout, saldo terpotong tapi pesanan tidak masuk. Kecewa parah!",
    "Sentimen Campuran / Kontras (Produk OK tapi Pengiriman Lambat)":
        "Barangnya bagus original mantap, tapi pengiriman lelet kurir lambat banget paket telat tiba.",
    "Pelayanan Seller & Customer Service Buruk":
        "Seller dichat gak balas sama sekali, barang cacat pas minta retur malah ditolak. Toko tidak amanah!",
    "Ulasan Netral / Standar":
        "Pengiriman standar, barang lumayan sesuai lah untuk harga segini."
}


def render_sentiment_card(sentiment: str, confidence: float, text_clean: str):
    """Menampilkan card sentimen dengan warna dinamis."""
    color = SENTIMENT_COLORS.get(sentiment.lower(), "#6c757d")
    badge = SENTIMENT_BADGES.get(sentiment.lower(), sentiment.upper())

    st.markdown(
        f"""
        <div style="border-radius: 10px; padding: 18px; background-color: {color}15; border-left: 6px solid {color}; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {color}; display: flex; align-items: center; gap: 8px;">
                {badge} <span style="font-size: 0.7em; color: #555;">(Keyakinan: {confidence:.1f}%)</span>
            </h3>
            <p style="margin: 8px 0 0 0; color: #444; font-size: 0.95em;">
                <strong>Teks Bersih (Preprocessed):</strong> <em>"{text_clean}"</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================================
# Halaman 1: Uji Sentimen Real-Time & ABSA
# =========================================================================
def render_page_realtime(predictor: SentimentPredictor, aspect_analyzer: AspectSentimentAnalyzer):
    st.subheader("🔍 Uji Sentimen Real-Time & Analisis Berbasis Aspek (ABSA)")
    st.write(
        "Ketik atau tempel teks ulasan e-commerce berbahasa Indonesia untuk mendeteksi sentimen utama "
        "serta sentimen spesifik pada 5 aspek utama (*Pengiriman, Kualitas Produk, Harga/Promo, Pelayanan CS, Aplikasi*)."
    )

    # Preset Selector
    selected_preset = st.selectbox("💡 Gunakan Contoh Ulasan Cepat:", list(PRESET_EXAMPLES.keys()))
    initial_text = PRESET_EXAMPLES[selected_preset] if selected_preset != "Pilih Contoh Ulasan..." else ""

    user_input = st.text_area(
        "Teks Ulasan Pelanggan:",
        value=initial_text,
        height=120,
        placeholder="Contoh: Barangnya bagus dan original, tapi kurirnya lambat pengiriman telat sampai..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)

    if analyze_clicked or (user_input and selected_preset != "Pilih Contoh Ulasan..."):
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
            return

        with st.spinner("Menganalisis teks & aspek sentimen..."):
            pred = predictor.predict_one(user_input)
            absa_result = aspect_analyzer.analyze_aspects(user_input)
            st.session_state["realtime_analysis"] = {
                "text": user_input,
                "pred": pred,
                "absa": absa_result
            }

    if "realtime_analysis" in st.session_state and st.session_state["realtime_analysis"]["text"] == user_input:
        curr = st.session_state["realtime_analysis"]
        pred = curr["pred"]
        absa_result = curr["absa"]

        st.markdown("---")
        # 1. Hasil Sentimen Utama
        render_sentiment_card(pred["sentimen"], pred["keyakinan"], pred["teks_bersih"])

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 📊 Rincian Probabilitas Model")
            probs = pred.get("probabilitas", {})
            if probs:
                prob_df = pd.DataFrame({
                    "Sentimen": [k.capitalize() for k in probs.keys()],
                    "Probabilitas (%)": [round(v * 100, 2) for v in probs.values()]
                })
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
                st.bar_chart(prob_df.set_index("Sentimen"), color="#1f77b4")
            else:
                st.info("Model tidak menyediakan estimasi probabilitas multi-kelas.")

        with col_right:
            st.markdown("#### 🎯 Aspek-Based Sentiment Analysis (ABSA)")
            detected_aspects = absa_result.get("aspek_terdeteksi", {})

            if not detected_aspects:
                st.info("ℹ️ Tidak ada aspek spesifik e-commerce (pengiriman, produk, harga, CS, app) yang terdeteksi.")
            else:
                for asp_key, data in detected_aspects.items():
                    asp_name = data["nama_aspek"]
                    asp_sent = data["sentimen"].lower()
                    asp_conf = data["keyakinan"]
                    asp_color = SENTIMENT_COLORS.get(asp_sent, "#555")
                    asp_badge = SENTIMENT_BADGES.get(asp_sent, asp_sent.upper())
                    clauses_str = ", ".join(f'"{k}"' for k in data["klausa_pendukung"])

                    with st.expander(f"📌 {asp_name} : {asp_badge} ({asp_conf:.1f}%)", expanded=True):
                        st.markdown(f"**Sentimen Aspek:** <span style='color:{asp_color}; font-weight:bold;'>{asp_sent.upper()}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Tingkat Keyakinan:** {asp_conf:.1f}%")
                        st.markdown(f"**Klausa Pendukung:** *{clauses_str}*")


# =========================================================================
# Halaman 2: Prediksi Massal (CSV / Excel)
# =========================================================================
def render_page_batch(predictor: SentimentPredictor, aspect_analyzer: AspectSentimentAnalyzer):
    st.subheader("📁 Prediksi Sentimen Massal (Batch Inference)")
    st.write(
        "Unggah file **CSV** atau **Excel (.xlsx)** berisi daftar ulasan e-commerce untuk mendapatkan "
        "klasifikasi sentimen secara otomatis beserta ringkasan metrik analitik bisnis."
    )

    uploaded_file = st.file_uploader(
        "Unggah file dataset ulasan (.csv atau .xlsx):",
        type=["csv", "xlsx", "xls"]
    )

    # Option to use sample dataset
    sample_col1, sample_col2 = st.columns([1, 3])
    with sample_col1:
        use_sample = st.checkbox("Gunakan dataset sampel bawaan", value=False)

    df_to_process = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_to_process = pd.read_csv(uploaded_file)
            else:
                df_to_process = pd.read_excel(uploaded_file)
            st.success(f"✓ Berhasil memuat file `{uploaded_file.name}` ({len(df_to_process)} baris)")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return
    elif use_sample:
        if config.RAW_DATASET_PATH.exists():
            df_to_process = pd.read_csv(config.RAW_DATASET_PATH).head(100)
            st.info(f"✓ Menggunakan 100 baris sampel dari `{config.RAW_DATASET_PATH.name}`")
        else:
            st.warning("File dataset sampel belum tersedia.")

    if df_to_process is not None:
        st.write("🔍 **Pratinjau Data (5 baris pertama):**")
        st.dataframe(df_to_process.head(), use_container_width=True)

        all_cols = list(df_to_process.columns)
        if not all_cols:
            st.warning("⚠️ File tidak memiliki kolom data.")
            return

        # Deteksi kolom teks secara aman
        try:
            detected_col = SentimentPredictor.detect_text_column(df_to_process)
        except KeyError:
            detected_col = all_cols[0]
        default_idx = all_cols.index(detected_col) if detected_col in all_cols else 0

        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            text_column = st.selectbox("Pilih kolom yang berisi teks ulasan:", all_cols, index=default_idx)
        with col_opt2:
            include_absa = st.checkbox("Sertakan analisis 5 aspek (ABSA per baris)", value=False)

        if st.button("⚡ Mulai Analisis Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Menjalankan inferensi model...")
            result_df = predictor.predict_dataframe(df_to_process, text_column=text_column)
            progress_bar.progress(60)

            if include_absa:
                status_text.text("Menjalankan analisis berbasis aspek (ABSA)...")
                result_df = aspect_analyzer.analyze_dataframe(result_df, text_column=text_column)

            progress_bar.progress(100)
            status_text.text("Selesai!")
            st.session_state["batch_result_df"] = result_df

        if "batch_result_df" in st.session_state:
            result_df = st.session_state["batch_result_df"]
            total = len(result_df)
            counts = result_df["prediksi_sentimen"].value_counts().to_dict()
            pos = counts.get("positif", 0)
            neu = counts.get("netral", 0)
            neg = counts.get("negatif", 0)
            pct_pos = round(pos / total * 100, 1) if total > 0 else 0
            pct_neg = round(neg / total * 100, 1) if total > 0 else 0
            pct_neu = round(neu / total * 100, 1) if total > 0 else 0
            nss = round(pct_pos - pct_neg, 1)
            avg_conf = round(float(result_df["keyakinan"].mean()), 1) if "keyakinan" in result_df.columns and not result_df.empty else 0.0

            st.markdown("---")
            st.markdown("### 📈 Ringkasan Metrik Bisnis")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Total Ulasan", f"{total:,}")
            kpi2.metric("Positif", f"{pos} ({pct_pos}%)")
            kpi3.metric("Netral", f"{neu} ({pct_neu}%)")
            kpi4.metric("Negatif", f"{neg} ({pct_neg}%)")
            kpi5.metric("Net Sentiment Score (NSS)", f"{nss:+0.1f}")

            # Visualisasi Distribusi
            st.markdown("#### 📊 Distribusi Sentimen")
            chart_df = pd.DataFrame({
                "Sentimen": ["Positif", "Netral", "Negatif"],
                "Jumlah": [pos, neu, neg]
            })
            st.bar_chart(chart_df.set_index("Sentimen"), color="#28a745")

            # Filter & Tampilan Tabel Hasil
            st.markdown("#### 📋 Tabel Hasil Prediksi")
            filter_sentiment = st.radio(
                "Filter Tampilan Berdasarkan Sentimen:",
                ["Semua", "Positif", "Netral", "Negatif"],
                horizontal=True
            )

            display_df = result_df.copy()
            if filter_sentiment != "Semua":
                display_df = display_df[display_df["prediksi_sentimen"] == filter_sentiment.lower()]

            st.dataframe(display_df, use_container_width=True)

            # Export Buttons
            st.markdown("#### 📥 Unduh Hasil Analisis")
            down_col1, down_col2 = st.columns(2)

            with down_col1:
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📄 Unduh sebagai CSV",
                    data=csv_data,
                    file_name="hasil_analisis_sentimen.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with down_col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    result_df.to_excel(writer, index=False, sheet_name="Hasil Sentimen")
                st.download_button(
                    label="📊 Unduh sebagai Excel (.xlsx)",
                    data=excel_buffer.getvalue(),
                    file_name="hasil_analisis_sentimen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


# =========================================================================
# Halaman 3: Benchmark Komparasi 7 Platform E-Commerce
# =========================================================================
def render_page_benchmark():
    st.subheader("📊 Benchmark Komparasi 7 Platform E-Commerce Indonesia")
    st.write(
        "Perbandingan performa sentimen kepuasan pengguna antara: "
        "**Shopee, Tokopedia, Lazada, Blibli, TikTok Shop, Bukalapak, dan Zalora**."
    )

    if config.COMPARISON_METRICS_PATH.exists():
        try:
            with open(config.COMPARISON_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            df_comp = pd.DataFrame(metrics_data)

            st.markdown("### 🏆 Peringkat Net Sentiment Score (NSS) Platform")
            st.caption("Net Sentiment Score (NSS) dihitung dengan: `% Positif - % Negatif` (Rentang: -100 s/d +100)")

            display_cols = ["nama_platform", "total_ulasan", "pct_positif", "pct_netral", "pct_negatif", "nss", "rata_rata_rating"]
            valid_cols = [c for c in display_cols if c in df_comp.columns]
            rename_map = {
                "nama_platform": "Platform",
                "total_ulasan": "Total Ulasan",
                "pct_positif": "% Positif",
                "pct_netral": "% Netral",
                "pct_negatif": "% Negatif",
                "nss": "NSS",
                "rata_rata_rating": "Rating Bintang"
            }
            styled_df = df_comp[valid_cols].rename(columns=rename_map)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Chart NSS
            if "Platform" in styled_df.columns and "NSS" in styled_df.columns:
                st.markdown("#### 📈 Grafik Net Sentiment Score (NSS)")
                chart_nss = styled_df[["Platform", "NSS"]].set_index("Platform")
                st.bar_chart(chart_nss)

        except Exception as e:
            st.error(f"Gagal memuat metrik komparasi: {e}")

    # Menampilkan Gambar Visualisasi Komparasi jika ada
    st.markdown("---")
    st.markdown("### 🖼️ Visualisasi Komparatif Hasil Pipeline")

    tab_vis1, tab_vis2, tab_vis3 = st.tabs([
        "Komparasi NSS",
        "Distribusi Sentimen per Platform",
        "WordCloud Keluhan Komparatif"
    ])

    with tab_vis1:
        if config.COMPARISON_NSS_PATH.exists():
            st.image(str(config.COMPARISON_NSS_PATH), caption="Benchmark Net Sentiment Score Antar Platform", use_container_width=True)
        else:
            st.info("Grafik komparasi NSS belum di-generate. Jalankan 'python main.py evaluate'.")

    with tab_vis2:
        if config.COMPARISON_SENTIMENT_PATH.exists():
            st.image(str(config.COMPARISON_SENTIMENT_PATH), caption="Proporsi Sentimen Antar Platform E-Commerce", use_container_width=True)
        else:
            st.info("Grafik proporsi sentimen belum di-generate. Jalankan 'python main.py evaluate'.")

    with tab_vis3:
        if config.COMPARISON_WORDCLOUD_GRID_PATH.exists():
            st.image(str(config.COMPARISON_WORDCLOUD_GRID_PATH), caption="WordCloud Pain Points / Keluhan Pelanggan Antar Platform", use_container_width=True)
        else:
            st.info("Grid WordCloud keluhan komparatif belum di-generate. Jalankan 'python main.py evaluate'.")


# =========================================================================
# Halaman 4: Metrik & Performa Model Machine Learning
# =========================================================================
def render_page_metrics():
    st.subheader("📈 Metrik Evaluasi & Benchmarking Model Machine Learning")
    st.write(
        "Hasil pengujian objektif algoritma klasifikasi sentimen menggunakan 5-Fold Stratified Cross Validation "
        "dan pengujian pada hold-out test set."
    )

    if not config.METRICS_PATH.exists():
        st.warning("⚠️ File `evaluation_metrics.json` belum ditemukan. Silakan jalankan `python main.py train`.")
        return

    try:
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        best_model = data.get("best_model_name", "N/A")
        st.success(f"🏆 **Model Terbaik Terpilih:** `{best_model}`")

        results = data.get("benchmark_results", {})
        if results:
            summary_rows = []
            for m_name, m_info in results.items():
                summary_rows.append({
                    "Model": m_name,
                    "CV F1 (Weighted)": f"{m_info.get('cv_f1_weighted_mean', 0)*100:.2f}%",
                    "Test Accuracy": f"{m_info.get('test_accuracy', 0)*100:.2f}%",
                    "Test F1 (Weighted)": f"{m_info.get('test_f1_weighted', 0)*100:.2f}%",
                    "Test F1 (Macro)": f"{m_info.get('test_f1_macro', 0)*100:.2f}%",
                })
            st.markdown("#### 📋 Tabel Perbandingan Benchmark Model")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("#### 🔲 Confusion Matrix")
            if config.CONFUSION_MATRIX_PATH.exists():
                st.image(str(config.CONFUSION_MATRIX_PATH), caption="Confusion Matrix Model Terbaik", use_container_width=True)
            else:
                st.info("Confusion matrix belum dibuat.")

        with col_img2:
            st.markdown("#### 📊 Distribusi Sentimen Dataset")
            if config.DISTRIBUTION_CHART_PATH.exists():
                st.image(str(config.DISTRIBUTION_CHART_PATH), caption="Distribusi Sentimen Dataset Ulasan", use_container_width=True)
            else:
                st.info("Grafik distribusi belum dibuat.")

        st.markdown("#### ☁️ WordCloud Analisis Kata Kunci")
        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            if config.WORDCLOUD_NEGATIF_PATH.exists():
                st.image(str(config.WORDCLOUD_NEGATIF_PATH), caption="Customer Pain Points (Keluhan Ulasan Negatif)", use_container_width=True)
        with wc_col2:
            if config.WORDCLOUD_POSITIF_PATH.exists():
                st.image(str(config.WORDCLOUD_POSITIF_PATH), caption="Customer Delight (Apresiasi Ulasan Positif)", use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membaca metrik model: {e}")


# =========================================================================
# Halaman 5: Panduan & REST API
# =========================================================================
def render_page_docs():
    st.subheader("ℹ️ Panduan Penggunaan & REST API Documentation")
    st.markdown(
        """
        ### 🚀 Arsitektur Sistem Analisis Sentimen
        Sistem ini dibangun dengan arsitektur modular end-to-end:
        1. **Data Scraper (`src/scraper.py`)**: Mengambil ulasan real-time Google Play Store untuk 7 platform e-commerce dengan penanganan offline otomatis.
        2. **Indonesian Text Preprocessor (`src/preprocessor.py`)**: Normalisasi emoji, kamus slang e-commerce, reduksi karakter berulang, dan stemming Sastrawi dengan *persistent disk caching*.
        3. **TF-IDF Feature Extractor (`src/features.py`)**: Unigram + Bigram `(1, 2)` dan Sublinear TF.
        4. **Multi-Model ML Trainer (`src/model_trainer.py`)**: Benchmarking Logistic Regression, Complement Naive Bayes, dan LinearSVC.
        5. **Aspect-Based Sentiment Analysis (`src/aspect_analyzer.py`)**: Pemecahan klausa dan pemetaan 5 aspek: Pengiriman, Kualitas Produk, Harga/Promo, CS, Sistem Aplikasi.
        6. **REST API Server (`src/api.py`)**: FastAPI berkecepatan tinggi dengan Swagger UI interaktif.
        7. **Web Dashboard (`app.py` & `src/dashboard.py`)**: Antarmuka interaktif berbasis Streamlit.

        ---

        ### 🌐 Endpoint REST API (FastAPI)
        Jalankan server API dengan perintah:
        ```bash
        python main.py serve --port 8000
        ```
        - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
        - `GET /health` : Memeriksa kesiapan dan status model.
        - `POST /predict` : Klasifikasi sentimen untuk 1 teks ulasan.
        - `POST /predict-batch` : Klasifikasi massal untuk list teks.
        - `POST /analyze-aspects` : Analisis sentimen berbasis aspek (ABSA).
        - `GET /metrics` : Mengambil akurasi dan metrik evaluasi model.
        - `GET /platforms` : Mengambil data benchmark 7 platform e-commerce.

        ---

        ### 💻 CLI Cheatsheet
        ```bash
        # Menjalankan Streamlit Dashboard
        python main.py dashboard
        # atau: streamlit run app.py

        # Menjalankan seluruh pipeline End-to-End
        python main.py run-all

        # Uji prediksi satu teks
        python main.py predict -t "Barangnya bagus packing aman mantap!"

        # Uji analisis aspek (ABSA)
        python main.py aspect -t "Barang ori bagus, tapi kurirnya lelet."

        # Prediksi batch dari file CSV / Excel
        python main.py predict-batch -i ulasan.csv -o hasil.csv
        ```
        """
    )


# =========================================================================
# Halaman: Laporan Eksekutif (HTML / PDF)
# =========================================================================
def render_page_report():
    st.subheader("📑 Laporan Analitik Bisnis Eksekutif (HTML / Siap Cetak PDF)")
    st.write(
        "Hasilkan laporan bisnis komprehensif dan mandiri (*self-contained*) yang merangkum seluruh temuan sentimen, "
        "peringkat Net Sentiment Score (NSS), Customer Pain Points, perbandingan model ML, dan rekomendasi strategis divisi."
    )

    report_path = config.OUTPUTS_DIR / "laporan_eksekutif_sentimen.html"

    col_act1, col_act2 = st.columns([1, 2])
    with col_act1:
        if st.button("🔄 Generate Laporan Eksekutif", type="primary", use_container_width=True):
            with st.spinner("Mengompilasi data dan merender laporan eksekutif..."):
                from src.report_generator import generate_business_report
                report_path = generate_business_report()
            st.success("✓ Laporan eksekutif berhasil diperbarui!")

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        with col_act2:
            st.download_button(
                label="📥 Unduh Laporan Lengkap (.html)",
                data=html_content.encode("utf-8"),
                file_name="laporan_eksekutif_sentimen_ecommerce.html",
                mime="text/html",
                use_container_width=True
            )

        st.info("💡 **Tips Cetak ke PDF**: Buka file HTML yang diunduh di browser, lalu klik tombol **'🖨️ Cetak / Ekspor PDF'** di bagian atas laporan atau tekan `Ctrl + P`, lalu pilih tujuan **Save as PDF**.")

        st.markdown("---")
        st.markdown("### 👁️ Pratinjau Dokumen Laporan")
        st.components.v1.html(html_content, height=850, scrolling=True)
    else:
        st.warning("File laporan belum dibuat. Klik tombol di atas untuk membuat laporan.")


# =========================================================================
# Main Entry Point Streamlit App
# =========================================================================
def main():
    st.set_page_config(
        page_title="E-Commerce Sentiment Analytics Dashboard",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inisialisasi Predictor
    predictor = load_predictor()
    if predictor is None:
        st.error(
            "❌ Model machine learning atau TF-IDF vectorizer belum ditemukan di direktori `models/`.\n\n"
            "Silakan latih model terlebih dahulu dengan menjalankan:\n"
            "```bash\npython main.py train\n```"
        )
        return

    aspect_analyzer = load_aspect_analyzer(predictor)

    # Sidebar Header & Navigasi
    st.sidebar.title("🛒 E-Commerce NLP")
    st.sidebar.markdown(
        "**Analisis Sentimen Tingkat Produksi untuk E-Commerce Indonesia**\n\n"
        "*Shopee • Tokopedia • Lazada • Blibli • TikTok Shop • Bukalapak • Zalora*"
    )
    st.sidebar.markdown("---")

    menu_options = [
        "🔍 Uji Sentimen Real-Time & ABSA",
        "📁 Prediksi Massal (CSV/Excel)",
        "📊 Benchmark 7 Platform",
        "📈 Metrik & Performa Model ML",
        "📑 Laporan Eksekutif (HTML/PDF)",
        "ℹ️ Panduan & Dokumentasi API"
    ]
    selected_menu = st.sidebar.radio("Navigasi Fitur:", menu_options)

    st.sidebar.markdown("---")
    st.sidebar.caption("Status Sistem:")
    st.sidebar.success("✓ Model ML: Siap Pakai")
    st.sidebar.info("✓ Persistent Cache: Aktif")
    st.sidebar.caption("Versi Aplikasi: 1.3.0")

    # Routing Halaman
    if selected_menu == "🔍 Uji Sentimen Real-Time & ABSA":
        render_page_realtime(predictor, aspect_analyzer)
    elif selected_menu == "📁 Prediksi Massal (CSV/Excel)":
        render_page_batch(predictor, aspect_analyzer)
    elif selected_menu == "📊 Benchmark 7 Platform":
        render_page_benchmark()
    elif selected_menu == "📈 Metrik & Performa Model ML":
        render_page_metrics()
    elif selected_menu == "📑 Laporan Eksekutif (HTML/PDF)":
        render_page_report()
    elif selected_menu == "ℹ️ Panduan & Dokumentasi API":
        render_page_docs()


if __name__ == "__main__":
    main()
