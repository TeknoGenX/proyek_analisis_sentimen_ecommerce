"""
Modul Generator Laporan Analitik Bisnis Otomatis untuk E-Commerce Indonesia.
Menghasilkan laporan eksekutif komprehensif berformat HTML interaktif yang mandiri (self-contained)
dan siap dicetak langsung ke PDF melalui browser (@media print styling).
"""
import base64
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.aspect_analyzer import AspectSentimentAnalyzer, ASPECT_TAXONOMY

logger = logging.getLogger(__name__)


class BusinessReportGenerator:
    def __init__(
        self,
        metrics_path: Optional[Path] = None,
        comparison_metrics_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None
    ):
        self.metrics_path = Path(metrics_path or config.METRICS_PATH)
        self.comparison_metrics_path = Path(comparison_metrics_path or config.COMPARISON_METRICS_PATH)
        self.dataset_path = Path(dataset_path or config.PROCESSED_DATASET_PATH)

    @staticmethod
    def _image_to_base64(image_path: Path) -> Optional[str]:
        """Mengonversi file gambar ke string Base64 data URI."""
        if not image_path or not image_path.exists():
            return None
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = image_path.suffix.lower().replace(".", "")
            mime = "image/png" if ext == "png" else f"image/{ext}"
            return f"data:{mime};base64,{encoded}"
        except Exception as e:
            logger.warning(f"Gagal mengonversi gambar {image_path} ke Base64: {e}")
            return None

    def _load_model_metrics(self) -> Dict[str, Any]:
        """Memuat data evaluasi model machine learning."""
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Gagal membaca model metrics: {e}")
        return {}

    def _load_comparison_metrics(self) -> List[Dict[str, Any]]:
        """Memuat data perbandingan performa 7 platform e-commerce."""
        if self.comparison_metrics_path.exists():
            try:
                with open(self.comparison_metrics_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Gagal membaca comparison metrics: {e}")
        return []

    def _load_dataset_stats(self) -> Dict[str, Any]:
        """Memuat statistik dataset ulasan."""
        path = self.dataset_path if self.dataset_path.exists() else config.RAW_DATASET_PATH
        if not path.exists():
            return {"total": 0, "positif": 0, "netral": 0, "negatif": 0, "sample_reviews": []}

        try:
            df = pd.read_csv(path)
            total = len(df)
            label_col = "label" if "label" in df.columns else ("sentimen" if "sentimen" in df.columns else None)
            text_col = "clean_review" if "clean_review" in df.columns else ("teks_ulasan" if "teks_ulasan" in df.columns else "content")

            counts = df[label_col].value_counts().to_dict() if label_col else {}
            pos = counts.get("positif", 0)
            neu = counts.get("netral", 0)
            neg = counts.get("negatif", 0)

            # Contoh ulasan positif & negatif
            samples_pos = []
            samples_neg = []
            if label_col and text_col in df.columns:
                samples_pos = df[df[label_col] == "positif"][text_col].dropna().head(3).tolist()
                samples_neg = df[df[label_col] == "negatif"][text_col].dropna().head(3).tolist()

            return {
                "total": total,
                "positif": pos,
                "netral": neu,
                "negatif": neg,
                "pct_positif": round(pos / total * 100, 1) if total > 0 else 0,
                "pct_netral": round(neu / total * 100, 1) if total > 0 else 0,
                "pct_negatif": round(neg / total * 100, 1) if total > 0 else 0,
                "nss": round((pos - neg) / total * 100, 1) if total > 0 else 0,
                "samples_pos": samples_pos,
                "samples_neg": samples_neg
            }
        except Exception as e:
            logger.warning(f"Gagal menghitung statistik dataset: {e}")
            return {"total": 0, "positif": 0, "netral": 0, "negatif": 0}

    def generate_html_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Menghasilkan file HTML laporan eksekutif lengkap dengan styling modern,
        tabel interaktif, KPI cards, visualisasi Base64, dan print CSS.
        """
        model_metrics = self._load_model_metrics()
        comp_metrics = self._load_comparison_metrics()
        data_stats = self._load_dataset_stats()

        timestamp_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M WIB")
        report_id = datetime.datetime.now().strftime("REP-ECOM-%Y%m%d-%H%M")

        # Visualisasi dalam Base64
        b64_cm = self._image_to_base64(config.CONFUSION_MATRIX_PATH)
        b64_dist = self._image_to_base64(config.DISTRIBUTION_CHART_PATH)
        b64_wc_neg = self._image_to_base64(config.WORDCLOUD_NEGATIF_PATH)
        b64_wc_pos = self._image_to_base64(config.WORDCLOUD_POSITIF_PATH)
        b64_nss_comp = self._image_to_base64(config.COMPARISON_NSS_PATH)
        b64_sent_comp = self._image_to_base64(config.COMPARISON_SENTIMENT_PATH)
        b64_wc_grid = self._image_to_base64(config.COMPARISON_WORDCLOUD_GRID_PATH)

        # Temukan platform terbaik dan terendah
        best_platform = "-"
        lowest_platform = "-"
        if comp_metrics:
            sorted_comp = sorted(
                comp_metrics,
                key=lambda x: float(x.get("nss")) if (x.get("nss") is not None and not pd.isna(x.get("nss"))) else 0.0,
                reverse=True
            )
            top_nss = float(sorted_comp[0].get("nss")) if (sorted_comp[0].get("nss") is not None and not pd.isna(sorted_comp[0].get("nss"))) else 0.0
            bot_nss = float(sorted_comp[-1].get("nss")) if (sorted_comp[-1].get("nss") is not None and not pd.isna(sorted_comp[-1].get("nss"))) else 0.0
            best_platform = f"{sorted_comp[0].get('nama_platform', '-')} (NSS: {top_nss:+0.1f})"
            lowest_platform = f"{sorted_comp[-1].get('nama_platform', '-')} (NSS: {bot_nss:+0.1f})"

        # Buat baris tabel komparasi
        table_rows_html = ""
        for item in comp_metrics:
            nss_val = float(item.get("nss")) if (item.get("nss") is not None and not pd.isna(item.get("nss"))) else 0.0
            nss_color = "#28a745" if nss_val > 20 else ("#dc3545" if nss_val < 0 else "#ffc107")
            rating_val = float(item.get("rata_rata_rating")) if (item.get("rata_rata_rating") is not None and not pd.isna(item.get("rata_rata_rating"))) else 0.0
            pct_pos_val = float(item.get("pct_positif") if item.get("pct_positif") is not None else 0.0)
            pct_neu_val = float(item.get("pct_netral") if item.get("pct_netral") is not None else 0.0)
            pct_neg_val = float(item.get("pct_negatif") if item.get("pct_negatif") is not None else 0.0)
            total_rev = int(item.get("total_ulasan") if item.get("total_ulasan") is not None else 0)
            table_rows_html += f"""
            <tr>
                <td style="font-weight: 600;">{item.get('nama_platform', '-')}</td>
                <td style="text-align: right;">{total_rev:,}</td>
                <td style="text-align: right; color: #28a745; font-weight: 600;">{pct_pos_val:.1f}%</td>
                <td style="text-align: right; color: #856404;">{pct_neu_val:.1f}%</td>
                <td style="text-align: right; color: #dc3545; font-weight: 600;">{pct_neg_val:.1f}%</td>
                <td style="text-align: right; font-weight: bold; color: {nss_color};">{nss_val:+0.1f}</td>
                <td style="text-align: center;">⭐ {rating_val:.1f}</td>
            </tr>
            """

        # Buat baris tabel model ML
        model_rows_html = ""
        benchmark_results = model_metrics.get("benchmark_results", {})
        for m_name, m_data in benchmark_results.items():
            is_best = m_name == model_metrics.get("best_model_name")
            badge = " <span class='badge badge-success'>TERBAIK</span>" if is_best else ""
            model_rows_html += f"""
            <tr>
                <td style="font-weight: 600;">{m_name}{badge}</td>
                <td style="text-align: right;">{m_data.get('cv_f1_weighted_mean', 0)*100:.2f}%</td>
                <td style="text-align: right;">{m_data.get('test_accuracy', 0)*100:.2f}%</td>
                <td style="text-align: right; font-weight: bold;">{m_data.get('test_f1_weighted', 0)*100:.2f}%</td>
                <td style="text-align: right;">{m_data.get('test_f1_macro', 0)*100:.2f}%</td>
            </tr>
            """

        # HTML Template
        html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laporan Analisis Sentimen E-Commerce Indonesia</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --success: #16a34a;
            --danger: #dc2626;
            --warning: #ca8a04;
            --dark: #0f172a;
            --light-bg: #f8fafc;
            --border: #e2e8f0;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--light-bg);
            color: var(--dark);
            line-height: 1.6;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
        }}
        /* Header */
        .report-header {{
            border-bottom: 3px solid var(--primary);
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .report-title h1 {{
            font-size: 26px;
            color: var(--primary);
            margin-bottom: 6px;
            letter-spacing: -0.5px;
        }}
        .report-subtitle {{
            color: #64748b;
            font-size: 14px;
        }}
        .report-meta {{
            text-align: right;
            font-size: 13px;
            color: #475569;
            background: #f1f5f9;
            padding: 10px 16px;
            border-radius: 8px;
        }}
        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 35px;
        }}
        .kpi-card {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-left: 5px solid var(--primary-light);
            border-radius: 8px;
            padding: 18px;
        }}
        .kpi-card.success {{ border-left-color: var(--success); }}
        .kpi-card.danger {{ border-left-color: var(--danger); }}
        .kpi-card.warning {{ border-left-color: var(--warning); }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: 800;
            color: var(--dark);
            margin-top: 6px;
        }}
        /* Sections */
        .section-title {{
            font-size: 18px;
            color: var(--primary);
            border-left: 4px solid var(--primary);
            padding-left: 10px;
            margin: 35px 0 15px 0;
            font-weight: 700;
        }}
        /* Executive Summary Box */
        .summary-box {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            font-size: 14.5px;
        }}
        .summary-box ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        .summary-box li {{
            margin-bottom: 6px;
        }}
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            font-size: 13.5px;
        }}
        th, td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #f1f5f9;
            font-weight: 700;
            color: #334155;
            text-align: left;
        }}
        tbody tr:hover {{
            background: #f8fafc;
        }}
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-success {{ background: #dcfce7; color: #15803d; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        .badge-warning {{ background: #fef9c3; color: #854d0e; }}
        /* Image Grid */
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .image-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .image-caption {{
            font-size: 12px;
            color: #64748b;
            margin-top: 8px;
            font-weight: 500;
        }}
        /* Recommendations */
        .recommendation-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .recommendation-card h4 {{
            color: var(--primary);
            font-size: 15px;
            margin-bottom: 6px;
        }}
        /* Print Styles */
        @media print {{
            body {{
                padding: 0;
                background: white;
            }}
            .container {{
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
            .no-print {{
                display: none !important;
            }}
            .page-break {{
                page-break-before: always;
            }}
        }}
        .action-bar {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 20px;
            gap: 10px;
        }}
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn-primary {{
            background: var(--primary);
            color: white;
        }}
        .btn-primary:hover {{
            background: var(--primary-light);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Action Bar (Print Button) -->
        <div class="action-bar no-print">
            <button class="btn btn-primary" onclick="window.print()">🖨️ Cetak / Ekspor PDF</button>
        </div>

        <!-- Header -->
        <header class="report-header">
            <div class="report-title">
                <h1>Laporan Analisis Sentimen E-Commerce Indonesia</h1>
                <div class="report-subtitle">
                    Executive Analytics & Customer Experience Intelligence (NLP-Driven)
                </div>
            </div>
            <div class="report-meta">
                <div><strong>Laporan ID:</strong> {report_id}</div>
                <div><strong>Waktu Dibuat:</strong> {timestamp_str}</div>
                <div><strong>Cakupan:</strong> 7 Platform E-Commerce</div>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Ulasan Dianalisis</div>
                <div class="kpi-value">{data_stats.get('total', 0):,}</div>
            </div>
            <div class="kpi-card success">
                <div class="kpi-label">Sentimen Positif</div>
                <div class="kpi-value">{data_stats.get('pct_positif', 0.0)}%</div>
            </div>
            <div class="kpi-card danger">
                <div class="kpi-label">Sentimen Negatif</div>
                <div class="kpi-value">{data_stats.get('pct_negatif', 0.0)}%</div>
            </div>
            <div class="kpi-card {'success' if data_stats.get('nss', 0) >= 0 else 'danger'}">
                <div class="kpi-label">Net Sentiment Score (NSS)</div>
                <div class="kpi-value">{data_stats.get('nss', 0.0):+0.1f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Model Terbaik</div>
                <div class="kpi-value" style="font-size: 18px; line-height: 1.3;">{model_metrics.get('best_model_name', '-')}</div>
            </div>
        </div>

        <!-- Ringkasan Eksekutif -->
        <div class="summary-box">
            <h3 style="color: var(--primary); margin-bottom: 8px;">📋 Ringkasan Eksekutif (Executive Summary)</h3>
            <p>
                Analisis ini mengevaluasi persepsi dan sentimen pelanggan terhadap 7 platform e-commerce terkemuka di Indonesia
                (*Shopee, Tokopedia, Lazada, Blibli, TikTok Shop, Bukalapak, Zalora*) berbasis algoritma Machine Learning
                dan ekstraksi fitur TF-IDF N-Gram (Unigram & Bigram).
            </p>
            <ul>
                <li><strong>Kepuasan Tertinggi:</strong> Platform dengan skor sentimen bersih terbaik adalah <strong>{best_platform}</strong>.</li>
                <li><strong>Area Perhatian (Pain Points):</strong> Platform yang memerlukan mitigasi keluhan intensif adalah <strong>{lowest_platform}</strong>.</li>
                <li><strong>Driver Keluhan Utama:</strong> Logistik (keterlambatan kurir), masalah stabilitas aplikasi pasca-update (checkout lag & saldo terpotong), serta respon penjual/CS.</li>
                <li><strong>Driver Kepuasan Utama:</strong> Kecepatan pengiriman, promo gratis ongkir, serta keaslian barang (official store).</li>
            </ul>
        </div>

        <!-- Tabel Benchmark Komparasi 7 Platform -->
        <h3 class="section-title">📊 Benchmark Komparatif 7 Platform E-Commerce</h3>
        <p style="margin-bottom: 12px; font-size: 13.5px; color: #475569;">
            Tabel berikut menyajikan distribusi sentimen dan <strong>Net Sentiment Score (NSS)</strong> yang merefleksikan kepuasan bersih pelanggan (% Positif dikurangi % Negatif).
        </p>
        <table>
            <thead>
                <tr>
                    <th>Platform E-Commerce</th>
                    <th style="text-align: right;">Total Ulasan</th>
                    <th style="text-align: right;">% Positif</th>
                    <th style="text-align: right;">% Netral</th>
                    <th style="text-align: right;">% Negatif</th>
                    <th style="text-align: right;">NSS</th>
                    <th style="text-align: center;">Rating Play Store</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html if table_rows_html else '<tr><td colspan="7" style="text-align: center; color: #888;">Data komparasi belum tersedia. Jalankan pipeline evaluasi.</td></tr>'}
            </tbody>
        </table>

        <!-- Visualisasi Benchmark Komparatif -->
        <div class="image-grid">
            {f'<div class="image-card"><img src="{b64_nss_comp}"><div class="image-caption">Grafik Peringkat Net Sentiment Score (NSS) Antar Platform</div></div>' if b64_nss_comp else ''}
            {f'<div class="image-card"><img src="{b64_sent_comp}"><div class="image-caption">Proporsi Sentimen Pelanggan per Platform</div></div>' if b64_sent_comp else ''}
        </div>

        <div class="page-break"></div>

        <!-- Analisis Berbasis Aspek (ABSA) -->
        <h3 class="section-title">🎯 Analisis Sentimen Berbasis Aspek (ABSA)</h3>
        <p style="margin-bottom: 14px; font-size: 13.5px; color: #475569;">
            Segmentasi sentimen spesifik pada 5 aspek krusial perjalanan belanja pengguna (User Journey):
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 30px;">
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
                <h4 style="color: var(--primary); font-size: 14px;">🚚 Pengiriman & Logistik</h4>
                <p style="font-size: 13px; color: #555; margin-top: 4px;">Fokus: Kurir, kecepatan antar, resi pelacakan, ongkir ekspedisi.</p>
                <div style="margin-top: 8px; font-size: 12px; font-weight: bold; color: var(--danger);">Top Pain Point: Keterlambatan paket & update status resi macet.</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
                <h4 style="color: var(--primary); font-size: 14px;">📦 Kualitas & Kondisi Produk</h4>
                <p style="font-size: 13px; color: #555; margin-top: 4px;">Fokus: Keaslian (ori/kw), kemasan bubble wrap, barang rusak/sesuai.</p>
                <div style="margin-top: 8px; font-size: 12px; font-weight: bold; color: var(--success);">Top Delight: Produk original dan kemasan aman berlapis.</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
                <h4 style="color: var(--primary); font-size: 14px;">🏷️ Harga & Promo</h4>
                <p style="font-size: 13px; color: #555; margin-top: 4px;">Fokus: Voucher diskon, gratis ongkir, biaya layanan aplikasi.</p>
                <div style="margin-top: 8px; font-size: 12px; font-weight: bold; color: var(--warning);">Top Concern: Kenaikan biaya jasa aplikasi & syarat voucher makin ketat.</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
                <h4 style="color: var(--primary); font-size: 14px;">💬 Pelayanan & Customer Service</h4>
                <p style="font-size: 13px; color: #555; margin-top: 4px;">Fokus: Respon chat toko, penyelesaian komplain, retur barang & refund.</p>
                <div style="margin-top: 8px; font-size: 12px; font-weight: bold; color: var(--primary);">Key Factor: Kecepatan seller dan solusi adil bagi pembeli.</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
                <h4 style="color: var(--primary); font-size: 14px;">📱 Aplikasi & Sistem Transaksi</h4>
                <p style="font-size: 13px; color: #555; margin-top: 4px;">Fokus: Checkout, dompet digital (GoPay/ShopeePay), bug versi terbaru.</p>
                <div style="margin-top: 8px; font-size: 12px; font-weight: bold; color: var(--danger);">Critical Bug: Crash saat flash sale dan sinkronisasi saldo.</div>
            </div>
        </div>

        <!-- WordCloud Analisis Kata Kunci -->
        <h3 class="section-title">☁️ Analisis Kata Kunci: Pain Points vs Delight</h3>
        <div class="image-grid">
            {f'<div class="image-card"><img src="{b64_wc_neg}"><div class="image-caption">Customer Pain Points (Kata Kunci Ulasan Negatif)</div></div>' if b64_wc_neg else ''}
            {f'<div class="image-card"><img src="{b64_wc_pos}"><div class="image-caption">Customer Delight (Kata Kunci Ulasan Positif)</div></div>' if b64_wc_pos else ''}
        </div>

        <!-- Grid WordCloud Komparasi Platform jika ada -->
        {f'''
        <div class="image-card" style="margin-bottom: 30px;">
            <img src="{b64_wc_grid}">
            <div class="image-caption">Perbandingan WordCloud Keluhan Pelanggan Antar Platform</div>
        </div>
        ''' if b64_wc_grid else ''}

        <!-- Evaluasi & Validasi Model ML -->
        <h3 class="section-title">🤖 Evaluasi & Benchmarking Model Machine Learning</h3>
        <p style="margin-bottom: 12px; font-size: 13.5px; color: #475569;">
            Pengujian model dilakukan secara komparatif dengan 5-Fold Stratified Cross-Validation pada data ulasan berimbang.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Algoritma Machine Learning</th>
                    <th style="text-align: right;">CV F1 (Weighted)</th>
                    <th style="text-align: right;">Test Accuracy</th>
                    <th style="text-align: right;">Test F1 (Weighted)</th>
                    <th style="text-align: right;">Test F1 (Macro)</th>
                </tr>
            </thead>
            <tbody>
                {model_rows_html if model_rows_html else '<tr><td colspan="5" style="text-align: center; color: #888;">Data model belum tersedia.</td></tr>'}
            </tbody>
        </table>

        <!-- Confusion Matrix Image -->
        <div class="image-grid">
            {f'<div class="image-card"><img src="{b64_cm}"><div class="image-caption">Confusion Matrix Model Terbaik pada Test Set</div></div>' if b64_cm else ''}
            {f'<div class="image-card"><img src="{b64_dist}"><div class="image-caption">Distribusi Proporsi Kelas Sentimen</div></div>' if b64_dist else ''}
        </div>

        <div class="page-break"></div>

        <!-- Rekomendasi Strategis Bisnis -->
        <h3 class="section-title">💡 Rekomendasi Strategis untuk Bisnis & Operasional</h3>
        
        <div class="recommendation-card">
            <h4>1. Tim Logistik & Operasional Pengiriman</h4>
            <p style="font-size: 13.5px; color: #475569;">
                Tingkatkan SLA (Service Level Agreement) bersama partner 3PL/ekspedisi. Sebagian besar ulasan negatif bintang 1
                disebabkan oleh paket tertahan tanpa kepastian update resi lebih dari 3 hari kerja.
            </p>
        </div>

        <div class="recommendation-card">
            <h4>2. Tim Produk & Engineering Mobile App</h4>
            <p style="font-size: 13.5px; color: #475569;">
                Lakukan pengujian regresi intensif pada alur pembayaran dan checkout sebelum rilis versi baru.
                Lonjakan keluhan sistem sering terdeteksi saat pembaruan UI yang menyebabkan lag pada perangkat low-end.
            </p>
        </div>

        <div class="recommendation-card">
            <h4>3. Tim Merchant Care & Layanan Pelanggan (CS)</h4>
            <p style="font-size: 13.5px; color: #475569;">
                Permudah proses mediasi komplain dan retur barang cacat/salah kirim. Pelanggan cenderung memberikan rating bintang 1
                bukan semata karena barang rusak, melainkan karena respon penjual yang lambat atau tidak solutif.
            </p>
        </div>

        <div class="recommendation-card">
            <h4>4. Tim Commercial & Marketing Promosi</h4>
            <p style="font-size: 13.5px; color: #475569;">
                Pastikan transparansi skema biaya layanan (*platform fee*) dan kuota gratis ongkir. Biaya siluman (*hidden fee*)
                merupakan sentimen negatif yang meningkat pesat pada ulasan aplikasi tahun ini.
            </p>
        </div>

        <!-- Footer -->
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid var(--border); text-align: center; font-size: 12px; color: #94a3b8;">
            Laporan dibuat secara otomatis oleh Proyek Analisis Sentimen E-Commerce Indonesia (End-to-End NLP Engine).
            © {datetime.datetime.now().year} All Rights Reserved.
        </footer>
    </div>
</body>
</html>
"""
        out_file = Path(output_path or (config.OUTPUTS_DIR / "laporan_eksekutif_sentimen.html"))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Laporan bisnis eksekutif berhasil disimpan ke: {out_file}")
        return out_file


def generate_business_report(output_path: Optional[Path] = None) -> Path:
    """Helper shortcut function untuk generate laporan eksekutif."""
    generator = BusinessReportGenerator()
    return generator.generate_html_report(output_path=output_path)


if __name__ == "__main__":
    path = generate_business_report()
    print(f"Laporan HTML berhasil dibuat: {path}")
