"""
Modul Visualisasi Bisnis dan Evaluasi Model NLP.
Mendukung:
1. Confusion Matrix Heatmap
2. WordCloud Pain Points (Keluhan Ulasan Negatif) & Delight (Ulasan Positif)
3. Diagram Distribusi Sentimen Tunggal
4. Komparasi Distribusi Proporsi Sentimen 7 Platform Berdampingan
5. Benchmark Net Sentiment Score (NSS) Komparatif Antar-Platform
6. Grid WordCloud Keluhan Spesifik Per Platform E-Commerce
7. WordCloud Mandiri untuk Masing-Masing Platform
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

import config

logger = logging.getLogger(__name__)

# Konfigurasi gaya visualisasi standar profesional
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


class SentimentVisualizer:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.platform_wc_dir = config.PLATFORM_WORDCLOUDS_DIR
        self.platform_wc_dir.mkdir(parents=True, exist_ok=True)

    def _get_platform_display_name(self, platform_key: str) -> str:
        """Mengambil nama tampilan ramah untuk platform."""
        info = config.ECOMMERCE_PLATFORMS.get(str(platform_key).lower())
        if info:
            return info["name"]
        return str(platform_key).replace("_", " ").title()

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        labels: List[str] = config.LABELS,
        save_path: Optional[Path] = None
    ) -> Path:
        """Membuat dan menyimpan heatmap Confusion Matrix beresolusi tinggi."""
        target_path = save_path or config.CONFUSION_MATRIX_PATH
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[label.capitalize() for label in labels],
            yticklabels=[label.capitalize() for label in labels],
            annot_kws={"size": 13, "fontweight": "bold"}
        )
        plt.title("Confusion Matrix: Prediksi vs Aktual", fontsize=15, fontweight="bold", pad=15)
        plt.xlabel("Label Prediksi (Model)", fontsize=12, labelpad=10)
        plt.ylabel("Label Aktual (Kenyataan)", fontsize=12, labelpad=10)
        plt.tight_layout()
        plt.savefig(target_path, dpi=300)
        plt.close()
        logger.info(f"Confusion Matrix disimpan ke: {target_path}")
        return target_path

    def plot_wordcloud(
        self,
        texts: pd.Series,
        title: str,
        colormap: str,
        save_path: Path,
        max_words: int = 100
    ) -> Optional[Path]:
        """Membuat dan menyimpan visualisasi WordCloud dari sekumpulan teks."""
        clean_text_joined = " ".join([str(t) for t in texts if str(t).strip()])
        if not clean_text_joined.strip():
            logger.warning(f"Teks kosong untuk WordCloud: {title}")
            return None

        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            colormap=colormap,
            max_words=max_words,
            collocations=False
        ).generate(clean_text_joined)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"WordCloud disimpan ke: {save_path}")
        return save_path

    def plot_pain_points_wordcloud(self, df: pd.DataFrame) -> Optional[Path]:
        """WordCloud untuk keluhan ulasan negatif umum."""
        text_col = "clean_review" if "clean_review" in df.columns else "teks_ulasan"
        neg_texts = df[df["label"] == "negatif"][text_col]
        return self.plot_wordcloud(
            texts=neg_texts,
            title="Customer Pain Points: Kata Terbanyak pada Keluhan / Ulasan Negatif",
            colormap="Reds",
            save_path=config.WORDCLOUD_NEGATIF_PATH
        )

    def plot_positive_highlights_wordcloud(self, df: pd.DataFrame) -> Optional[Path]:
        """WordCloud untuk keunggulan ulasan positif umum."""
        text_col = "clean_review" if "clean_review" in df.columns else "teks_ulasan"
        pos_texts = df[df["label"] == "positif"][text_col]
        return self.plot_wordcloud(
            texts=pos_texts,
            title="Customer Delight: Kata Terbanyak pada Ulasan Positif",
            colormap="Greens",
            save_path=config.WORDCLOUD_POSITIF_PATH
        )

    def plot_distribution(self, df: pd.DataFrame) -> Path:
        """Membuat diagram batang distribusi sentimen pada dataset tunggal."""
        plt.figure(figsize=(8, 5))
        counts = df["label"].value_counts()
        palette = {"positif": "#2ecc71", "netral": "#f39c12", "negatif": "#e74c3c"}
        
        colors = [palette.get(k, "#3498db") for k in counts.index]
        ax = counts.plot(kind="bar", color=colors, edgecolor="black", alpha=0.85)

        total = len(df)
        for p in ax.patches:
            val = int(p.get_height())
            pct = val / total * 100
            ax.annotate(
                f"{val} ({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5),
                textcoords='offset points'
            )

        plt.title("Distribusi Kelas Sentimen Ulasan E-Commerce", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Kategori Sentimen", fontsize=12)
        plt.ylabel("Jumlah Ulasan", fontsize=12)
        plt.xticks(rotation=0, fontsize=11)
        plt.tight_layout()
        plt.savefig(config.DISTRIBUTION_CHART_PATH, dpi=300)
        plt.close()
        logger.info(f"Diagram distribusi disimpan ke: {config.DISTRIBUTION_CHART_PATH}")
        return config.DISTRIBUTION_CHART_PATH

    # =========================================================================
    # MODUL METRIK & VISUALISASI KOMPARATIF 7 PLATFORM
    # =========================================================================

    def calculate_platform_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghitung metrik sentimen komparatif per platform:
        - Total Ulasan
        - Proporsi Positif, Netral, Negatif (%)
        - Net Sentiment Score (NSS = % Positif - % Negatif)
        - Rata-rata Rating Bintang
        """
        if "platform" not in df.columns or "label" not in df.columns:
            raise KeyError("DataFrame harus memiliki kolom 'platform' dan 'label'.")

        records = []
        platforms = df["platform"].unique()

        for p in platforms:
            sub_df = df[df["platform"] == p]
            total = len(sub_df)
            counts = sub_df["label"].value_counts()
            
            pos = int(counts.get("positif", 0))
            neu = int(counts.get("netral", 0))
            neg = int(counts.get("negatif", 0))

            pct_pos = (pos / total * 100) if total > 0 else 0.0
            pct_neu = (neu / total * 100) if total > 0 else 0.0
            pct_neg = (neg / total * 100) if total > 0 else 0.0

            # Rumus Net Sentiment Score (NSS)
            nss = pct_pos - pct_neg

            avg_rating = float(sub_df["rating"].mean()) if "rating" in sub_df.columns else 0.0

            records.append({
                "platform": p,
                "nama_platform": self._get_platform_display_name(p),
                "total_ulasan": total,
                "positif": pos,
                "netral": neu,
                "negatif": neg,
                "pct_positif": round(pct_pos, 2),
                "pct_netral": round(pct_neu, 2),
                "pct_negatif": round(pct_neg, 2),
                "nss": round(nss, 2),
                "rata_rata_rating": round(avg_rating, 2)
            })

        metrics_df = pd.DataFrame(records).sort_values("nss", ascending=False).reset_index(drop=True)

        # Simpan ke file JSON metrik
        metrics_dict = metrics_df.to_dict(orient="records")
        try:
            with open(config.COMPARISON_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Metrik komparatif disimpan ke: {config.COMPARISON_METRICS_PATH}")
        except Exception as e:
            logger.warning(f"Gagal menyimpan komparasi_metrics.json: {e}")

        return metrics_df

    def plot_comparison_sentiment_distribution(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Membuat diagram batang komparasi proporsi sentimen untuk seluruh platform secara berdampingan.
        """
        target_path = save_path or config.COMPARISON_SENTIMENT_PATH
        metrics_df = self.calculate_platform_metrics(df)

        # Siapkan data untuk grouped bar chart
        platforms = metrics_df["nama_platform"].tolist()
        x = np.arange(len(platforms))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))

        rects_pos = ax.bar(x - width, metrics_df["pct_positif"], width, label="Positif", color="#2ecc71", edgecolor="black", alpha=0.9)
        rects_neu = ax.bar(x, metrics_df["pct_netral"], width, label="Netral", color="#f39c12", edgecolor="black", alpha=0.9)
        rects_neg = ax.bar(x + width, metrics_df["pct_negatif"], width, label="Negatif", color="#e74c3c", edgecolor="black", alpha=0.9)

        # Menambahkan nilai persentase di atas setiap batang
        def _add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold"
                    )

        _add_labels(rects_pos)
        _add_labels(rects_neu)
        _add_labels(rects_neg)

        ax.set_ylabel("Proporsi Sentimen (%)", fontsize=12, fontweight="bold")
        ax.set_title("Perbandingan Distribusi Proporsi Sentimen Ulasan E-Commerce Indonesia", fontsize=15, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(platforms, fontsize=11, fontweight="bold", rotation=15, ha="right")
        ax.legend(title="Kategori Sentimen", fontsize=11, title_fontsize=11, loc="upper right")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(target_path, dpi=300)
        plt.close()
        logger.info(f"Grafik komparasi distribusi sentimen disimpan ke: {target_path}")
        return target_path

    def plot_comparison_nss(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Membuat visualisasi Benchmark Net Sentiment Score (NSS = % Positif - % Negatif)
        antar-platform e-commerce dalam bentuk horizontal bar chart terurut.
        """
        target_path = save_path or config.COMPARISON_NSS_PATH
        metrics_df = self.calculate_platform_metrics(df).sort_values("nss", ascending=True)

        platforms = metrics_df["nama_platform"].tolist()
        nss_values = metrics_df["nss"].tolist()

        # Warna kondisional: Hijau untuk positif, Merah untuk negatif, Abu untuk netral
        bar_colors = ["#27ae60" if val >= 0 else "#c0392b" for val in nss_values]

        fig, ax = plt.subplots(figsize=(12, 6.5))
        bars = ax.barh(platforms, nss_values, color=bar_colors, edgecolor="black", height=0.55, alpha=0.9)

        # Garis netral pada x = 0
        ax.axvline(0, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.8)

        # Tambahkan label angka NSS di ujung batang
        for bar in bars:
            val = bar.get_width()
            offset = 2 if val >= 0 else -2
            ha = "left" if val >= 0 else "right"
            prefix = "+" if val > 0 else ""
            ax.annotate(
                f"{prefix}{val:.1f}",
                xy=(val, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=ha, va="center",
                fontsize=11, fontweight="bold",
                color="#1a1a1a"
            )

        ax.set_xlabel("Net Sentiment Score (NSS = % Positif - % Negatif)", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title("Benchmark Net Sentiment Score (NSS) Antar-Platform E-Commerce", fontsize=15, fontweight="bold", pad=15)
        ax.set_xlim(-105, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Anotasi zona kepuasan
        ax.text(50, len(platforms) - 0.5, "▲ Zona Kepuasan Tinggi", color="#27ae60", fontweight="bold", ha="center")
        ax.text(-50, len(platforms) - 0.5, "▼ Zona Keluhan Dominan", color="#c0392b", fontweight="bold", ha="center")

        plt.tight_layout()
        plt.savefig(target_path, dpi=300)
        plt.close()
        logger.info(f"Grafik benchmark NSS disimpan ke: {target_path}")
        return target_path

    def plot_platform_pain_points_grid(
        self,
        df: pd.DataFrame,
        save_path: Optional[Path] = None,
        max_words: int = 50
    ) -> Optional[Path]:
        """
        Membuat grid panel WordCloud khusus keluhan/ulasan negatif untuk seluruh platform
        dalam satu kanvas perbandingan visual yang komprehensif.
        """
        target_path = save_path or config.COMPARISON_WORDCLOUD_GRID_PATH
        text_col = "clean_review" if "clean_review" in df.columns else "teks_ulasan"
        
        # Ambil daftar platform yang ada di df atau default 7 platform
        platforms = [p for p in config.ECOMMERCE_PLATFORMS.keys() if p in df["platform"].values]
        if not platforms:
            platforms = list(df["platform"].unique())

        n_plats = len(platforms)
        if n_plats == 0:
            return None

        # Tentukan layout grid (maksimal 4 kolom)
        n_cols = min(4, n_plats)
        n_rows = int(np.ceil(n_plats / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for i, p in enumerate(platforms):
            ax = axes[i]
            p_name = self._get_platform_display_name(p)
            neg_texts = df[(df["platform"] == p) & (df["label"] == "negatif")][text_col]
            joined_text = " ".join([str(t) for t in neg_texts if str(t).strip()])

            if joined_text.strip():
                try:
                    wc = WordCloud(
                        width=500,
                        height=350,
                        background_color="white",
                        colormap="Reds",
                        max_words=max_words,
                        collocations=False
                    ).generate(joined_text)
                    ax.imshow(wc, interpolation="bilinear")
                    ax.set_title(f"{p_name}\n({len(neg_texts)} Keluhan)", fontsize=12, fontweight="bold", pad=8)
                except Exception as err:
                    ax.text(0.5, 0.5, f"Gagal membuat WordCloud:\n{err}", ha="center", va="center", fontsize=10)
                    ax.set_title(p_name, fontsize=12, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "Belum Ada Keluhan Negatif", ha="center", va="center", color="#7f8c8d", fontsize=11, fontweight="bold")
                ax.set_title(f"{p_name}\n(0 Keluhan)", fontsize=12, fontweight="bold", pad=8)

            ax.axis("off")

        # Sembunyikan subplot yang kosong jika ada
        for j in range(n_plats, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            "Customer Pain Points Komparatif: Kata Keluhan Terbanyak Per Platform E-Commerce",
            fontsize=16,
            fontweight="bold",
            y=0.99
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(target_path, dpi=300)
        plt.close()
        logger.info(f"Grid WordCloud keluhan komparatif disimpan ke: {target_path}")
        return target_path

    def plot_individual_platform_wordclouds(self, df: pd.DataFrame) -> Dict[str, Dict[str, Path]]:
        """
        Menghasilkan file WordCloud individu untuk ulasan negatif dan positif per platform.
        Disimpan di outputs/platform_wordclouds/.
        """
        text_col = "clean_review" if "clean_review" in df.columns else "teks_ulasan"
        platforms = df["platform"].unique()
        saved_files: Dict[str, Dict[str, Path]] = {}

        for p in platforms:
            p_name = self._get_platform_display_name(p)
            saved_files[p] = {}

            # 1. Keluhan Negatif
            neg_texts = df[(df["platform"] == p) & (df["label"] == "negatif")][text_col]
            neg_path = self.platform_wc_dir / f"WordCloud_Keluhan_{p}.png"
            res_neg = self.plot_wordcloud(
                texts=neg_texts,
                title=f"Customer Pain Points ({p_name}): Ulasan Negatif",
                colormap="Reds",
                save_path=neg_path
            )
            if res_neg:
                saved_files[p]["negatif"] = res_neg

            # 2. Ulasan Positif
            pos_texts = df[(df["platform"] == p) & (df["label"] == "positif")][text_col]
            pos_path = self.platform_wc_dir / f"WordCloud_Positif_{p}.png"
            res_pos = self.plot_wordcloud(
                texts=pos_texts,
                title=f"Customer Delight ({p_name}): Ulasan Positif",
                colormap="Greens",
                save_path=pos_path
            )
            if res_pos:
                saved_files[p]["positif"] = res_pos

        logger.info(f"WordCloud per-platform selesai untuk {len(platforms)} platform.")
        return saved_files

    def plot_all_comparative_analytics(self, df: pd.DataFrame) -> Dict[str, Union[Path, Dict]]:
        """
        Pipeline lengkap eksekusi seluruh analisis & visualisasi komparatif multi-platform.
        """
        logger.info(f"Menjalankan visualisasi komparatif untuk {df['platform'].nunique()} platform...")
        metrics_df = self.calculate_platform_metrics(df)
        dist_path = self.plot_comparison_sentiment_distribution(df)
        nss_path = self.plot_comparison_nss(df)
        grid_path = self.plot_platform_pain_points_grid(df)
        individual_clouds = self.plot_individual_platform_wordclouds(df)

        return {
            "metrics_df": metrics_df,
            "distribution_chart": dist_path,
            "nss_chart": nss_path,
            "wordcloud_grid": grid_path,
            "individual_wordclouds": individual_clouds
        }
