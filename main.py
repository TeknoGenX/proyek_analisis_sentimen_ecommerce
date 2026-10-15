"""
CLI Utama untuk Proyek Analisis Sentimen E-Commerce Indonesia.
Penggunaan:
  python main.py run-all       # Menjalankan seluruh pipeline dari awal hingga akhir
  python main.py scrape        # Mengambil ulasan dari Google Play Store
  python main.py preprocess    # Melakukan pra-pemrosesan teks & caching
  python main.py train         # Melatih & memilih model terbaik (benchmarking)
  python main.py evaluate      # Menghasilkan visualisasi Confusion Matrix & WordCloud
  python main.py predict -t "..." # Memprediksi satu kalimat ulasan
  python main.py interactive   # Mode interaktif di terminal dengan Rich UI
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Tambahkan direktori root ke path pencarian
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.scraper import scrape_playstore_reviews, scrape_multi_platforms, load_dataset, rating_to_sentiment
from src.preprocessor import IndonesianTextPreprocessor
from src.features import FeatureExtractor
from src.model_trainer import SentimentModelTrainer
from src.visualizer import SentimentVisualizer
from src.predictor import SentimentPredictor

# Coba gunakan Rich untuk tampilan terminal yang menarik
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_banner():
    title = "PROYEK ANALISIS SENTIMEN E-COMMERCE INDONESIA"
    subtitle = "End-to-End NLP Pipeline: Scraper, Slang Normalizer, TF-IDF & Multi-Model ML"
    if HAS_RICH:
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]\n[italic]{subtitle}[/italic]", border_style="cyan"))
    else:
        print("=" * 70)
        print(title)
        print(subtitle)
        print("=" * 70)


def cmd_scrape(args):
    """Menjalankan scraping ulasan Google Play Store untuk satu atau seluruh 7 platform."""
    platform = getattr(args, "platform", None)
    filter_tiktok = not getattr(args, "no_filter_tiktok", False)
    workers = getattr(args, "workers", config.SCRAPE_MAX_WORKERS)
    count = getattr(args, "count", config.DEFAULT_SCRAPE_COUNT)
    
    if platform == "all":
        print(f"\n[*] Memulai scraping batch paralel untuk SELURUH 7 platform e-commerce...")
        print(f"[*] Target: {count} ulasan/platform (Paralel {workers} workers)...")
        df = scrape_multi_platforms(
            count_per_platform=count,
            max_workers=workers,
            save_path=str(config.COMPARISON_RAW_DATASET_PATH),
            filter_tiktok=filter_tiktok
        )
        print(f"[✓] Berhasil menyimpan dataset komparasi multi-platform ({len(df)} baris) ke: {config.COMPARISON_RAW_DATASET_PATH}")
        return df
    else:
        app_id = getattr(args, "app", None)
        target_display = platform or app_id or config.DEFAULT_PLATFORM
        print(f"\n[*] Mengambil {count} ulasan dari Play Store untuk platform/app: {target_display}...")
        df = scrape_playstore_reviews(
            app_id=app_id,
            platform_key=platform,
            count=count,
            save_path=str(config.RAW_DATASET_PATH),
            filter_tiktok=filter_tiktok
        )
        print(f"[✓] Berhasil menyimpan dataset mentah ke: {config.RAW_DATASET_PATH}")
        return df


def cmd_preprocess(args):
    """Melakukan pembersihan data, normalisasi slang, dan stemming dengan cache."""
    print("\n[*] Memulai tahap Pra-Pemrosesan Teks...")
    df = load_dataset(config.RAW_DATASET_PATH)
    
    preprocessor = IndonesianTextPreprocessor()
    print(f"[*] Memproses {len(df)} teks ulasan utama (caching aktif)...")
    
    df["clean_review"] = df["teks_ulasan"].apply(preprocessor.clean)
    
    # Hapus baris yang kosong setelah dibersihkan
    initial_len = len(df)
    df = df[df["clean_review"].str.strip() != ""]
    final_len = len(df)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DATASET_PATH, index=False)
    print(f"[✓] Pra-pemrosesan dataset utama selesai! {initial_len - final_len} baris kosong dibersihkan.")
    print(f"[✓] Dataset bersih disimpan ke: {config.PROCESSED_DATASET_PATH}")

    # Bersihkan juga dataset komparasi multi-platform jika file mentahnya ada
    if config.COMPARISON_RAW_DATASET_PATH.exists():
        print("\n[*] Memproses dataset komparasi multi-platform...")
        comp_df = pd.read_csv(config.COMPARISON_RAW_DATASET_PATH)
        if "content" in comp_df.columns and "teks_ulasan" not in comp_df.columns:
            comp_df = comp_df.rename(columns={"content": "teks_ulasan", "score": "rating"})
        if "teks_ulasan" in comp_df.columns:
            comp_df["clean_review"] = comp_df["teks_ulasan"].apply(preprocessor.clean)
            if "label" not in comp_df.columns and "rating" in comp_df.columns:
                comp_df["label"] = comp_df["rating"].astype(int).apply(rating_to_sentiment)
            comp_df = comp_df[comp_df["clean_review"].str.strip() != ""]
            comp_df.to_csv(config.COMPARISON_PROCESSED_DATASET_PATH, index=False)
            print(f"[✓] Dataset komparasi bersih disimpan ke: {config.COMPARISON_PROCESSED_DATASET_PATH}")

    preprocessor.save_cache()
    return df



def cmd_train(args):
    """Melakukan ekstraksi fitur TF-IDF dan benchmarking model."""
    if not config.PROCESSED_DATASET_PATH.exists():
        print(f"[!] Dataset bersih belum ada di {config.PROCESSED_DATASET_PATH}. Menjalankan preprocess terlebih dahulu...")
        cmd_preprocess(args)

    print("\n[*] Membaca dataset bersih...")
    df = pd.read_csv(config.PROCESSED_DATASET_PATH)
    df = df.dropna(subset=["clean_review", "label"])

    X = df["clean_review"]
    y = df["label"]

    if len(df) < 10:
        raise ValueError(
            f"Jumlah ulasan bersih ({len(df)}) terlalu sedikit untuk melatih model. "
            f"Minimal dibutuhkan 10 ulasan dengan beragam sentimen."
        )

    if y.nunique() < 2:
        raise ValueError(
            f"Dataset hanya memiliki 1 variasi label sentimen: {y.unique().tolist()}. "
            f"Model machine learning membutuhkan minimal 2 kelas berbeda untuk dilatih."
        )

    # Cek apakah stratify bisa digunakan
    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2
    stratify_arg = y if can_stratify else None
    if not can_stratify:
        print("[!] Catatan: Terdapat label dengan < 2 sampel. Stratifikasi train/test dinonaktifkan.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_arg
    )

    # 1. Ekstraksi Fitur
    extractor = FeatureExtractor()
    X_train_vec = extractor.fit_transform(X_train)
    X_test_vec = extractor.transform(X_test)
    extractor.save_vectorizer()

    # 2. Benchmarking & Training
    trainer = SentimentModelTrainer()
    results = trainer.train_and_benchmark(X_train_vec, y_train, X_test_vec, y_test)
    trainer.save_best_model()

    return trainer, extractor, df, (X_test, y_test)


def cmd_evaluate(args):
    """Menghasilkan seluruh visualisasi (Confusion Matrix, WordClouds, dan Komparasi Multi-Platform)."""
    viz = SentimentVisualizer()
    
    # 1. Evaluasi Model Machine Learning jika model & dataset pemodelan tersedia
    if config.PROCESSED_DATASET_PATH.exists() and config.MODEL_PATH.exists():
        df = pd.read_csv(config.PROCESSED_DATASET_PATH).dropna(subset=["clean_review", "label"])
        X = df["clean_review"]
        y = df["label"]
        class_counts = y.value_counts()
        stratify_arg = y if class_counts.min() >= 2 else None
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify_arg
        )

        extractor = FeatureExtractor()
        extractor.load_vectorizer()
        X_test_vec = extractor.transform(X_test)

        trainer = SentimentModelTrainer()
        model = trainer.load_model()
        y_pred = model.predict(X_test_vec)

        print("\n[*] Membuat visualisasi evaluasi model ML...")
        viz.plot_confusion_matrix(y_test, y_pred)
        viz.plot_pain_points_wordcloud(df)
        viz.plot_positive_highlights_wordcloud(df)
        viz.plot_distribution(df)
    else:
        print("[!] File dataset bersih atau model belum lengkap untuk evaluasi Confusion Matrix.")

    # 2. Visualisasi Komparasi Multi-Platform jika dataset komparasi tersedia
    comp_file = config.COMPARISON_PROCESSED_DATASET_PATH if config.COMPARISON_PROCESSED_DATASET_PATH.exists() else config.COMPARISON_RAW_DATASET_PATH
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
        if "platform" in comp_df.columns and comp_df["platform"].nunique() > 1:
            print("\n[*] Menghasilkan analisis & visualisasi komparatif multi-platform...")
            res_comp = viz.plot_all_comparative_analytics(comp_df)
            metrics_df = res_comp["metrics_df"]
            
            if HAS_RICH:
                table = Table(title="Benchmark Net Sentiment Score (NSS) E-Commerce Indonesia", border_style="cyan")
                table.add_column("Peringkat", justify="center")
                table.add_column("Platform", style="bold")
                table.add_column("Total Ulasan", justify="right")
                table.add_column("% Positif", justify="right")
                table.add_column("% Netral", justify="right")
                table.add_column("% Negatif", justify="right")
                table.add_column("NSS", justify="right", style="bold")
                for rank, row in enumerate(metrics_df.itertuples(), start=1):
                    color = "green" if row.nss >= 0 else "red"
                    table.add_row(
                        str(rank),
                        row.nama_platform,
                        str(row.total_ulasan),
                        f"{row.pct_positif:.1f}%",
                        f"{row.pct_netral:.1f}%",
                        f"{row.pct_negatif:.1f}%",
                        f"[{color}]{row.nss:+.1f}[/{color}]"
                    )
                console.print(table)
            else:
                print("\n" + "=" * 65)
                print("BENCHMARK NET SENTIMENT SCORE (NSS) ANTAR-PLATFORM")
                print("=" * 65)
                print(metrics_df[["nama_platform", "total_ulasan", "pct_positif", "pct_negatif", "nss"]].to_string(index=False))

    print(f"\n[✓] Seluruh visualisasi berhasil disimpan di direktori: {config.OUTPUTS_DIR}")


def cmd_predict(args):
    """Memprediksi sentimen untuk satu input teks ulasan."""
    if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
        print("[!] Model atau TF-IDF vectorizer belum dilatih.")
        print("[*] Silakan jalankan 'python main.py train' terlebih dahulu.")
        return

    predictor = SentimentPredictor()
    res = predictor.predict_one(args.text)

    if HAS_RICH:
        color = "green" if res["sentimen"] == "positif" else ("red" if res["sentimen"] == "negatif" else "yellow")
        tbl = Table(title="Hasil Prediksi Sentimen", border_style="cyan")
        tbl.add_column("Field", style="bold")
        tbl.add_column("Nilai")
        tbl.add_row("Ulasan Asli", res["teks_asli"])
        tbl.add_row("Teks Bersih", res["teks_bersih"])
        tbl.add_row("Prediksi Sentimen", f"[{color}]{res['sentimen'].upper()}[/{color}]")
        tbl.add_row("Tingkat Keyakinan", f"{res['keyakinan']:.2f}%")
        console.print(tbl)
    else:
        print("\n" + "=" * 50)
        print("HASIL PREDIKSI SENTIMEN")
        print("=" * 50)
        print(f"Ulasan Asli       : {res['teks_asli']}")
        print(f"Teks Bersih       : {res['teks_bersih']}")
        print(f"Prediksi Sentimen : {res['sentimen'].upper()}")
        print(f"Keyakinan         : {res['keyakinan']:.2f}%")
        print("=" * 50)


def cmd_interactive(args):
    """Mode pengujian sentimen real-time interaktif di terminal."""
    if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
        print("[!] Model atau TF-IDF vectorizer belum dilatih.")
        print("[*] Silakan jalankan 'python main.py train' terlebih dahulu.")
        return

    predictor = SentimentPredictor()
    print("\n" + "=" * 60)
    print(" MODE PENGUJIAN SENTIMEN INTERAKTIF (Ketik 'keluar' untuk stop) ")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nMasukkan teks ulasan: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["keluar", "exit", "quit", "q"]:
                print("Keluar dari program. Terima kasih!")
                break

            res = predictor.predict_one(user_input)
            sentimen = res["sentimen"].upper()
            confidence = res["keyakinan"]

            if HAS_RICH:
                color = "green" if res["sentimen"] == "positif" else ("red" if res["sentimen"] == "negatif" else "yellow")
                console.print(
                    f"-> Hasil: [{color} bold][ SENTIMEN {sentimen} ][/{color} bold] "
                    f"| Keyakinan: [bold]{confidence:.1f}%[/bold]"
                )
                console.print(f"   [dim](Normalisasi: '{res['teks_bersih']}') [/dim]")
            else:
                ansi_color = "\033[92m" if res["sentimen"] == "positif" else ("\033[91m" if res["sentimen"] == "negatif" else "\033[93m")
                reset = "\033[0m"
                print(f"-> Hasil: {ansi_color}[ SENTIMEN {sentimen} ]{reset} | Keyakinan: {confidence:.1f}%")
                print(f"   (Normalisasi: '{res['teks_bersih']}')")

        except KeyboardInterrupt:
            print("\nProgram dihentikan.")
            break


def cmd_run_all(args):
    """Menjalankan seluruh pipeline End-to-End."""
    print_banner()
    
    # 1. Cek apakah dataset mentah sudah ada atau perlu scrape
    force_scrape = getattr(args, "force_scrape", False)
    needs_scrape = False

    if force_scrape or not config.RAW_DATASET_PATH.exists():
        if force_scrape:
            print("[*] Argumen --force-scrape aktif. Melakukan scraping baru dari Play Store...")
        else:
            print("[!] File dataset_ulasan.csv belum ada. Melakukan scraping...")
        needs_scrape = True
    else:
        try:
            raw_check = pd.read_csv(config.RAW_DATASET_PATH)
            # Validasi apakah dataset cukup dan memiliki kelas beragam
            if len(raw_check) < 30 or "label" not in raw_check.columns or raw_check["label"].nunique() < 2:
                print(f"[!] Dataset di {config.RAW_DATASET_PATH} terlalu kecil ({len(raw_check)} baris) atau hanya 1 kelas.")
                parent_csv = config.BASE_DIR.parent / "dataset_ulasan.csv"
                if parent_csv.exists() and len(pd.read_csv(parent_csv)) >= 50:
                    print(f"[*] Mengimpor dataset ulasan representatif dari {parent_csv}...")
                    valid_df = load_dataset(parent_csv)
                    valid_df.to_csv(config.RAW_DATASET_PATH, index=False)
                    print(f"[✓] Berhasil memuat {len(valid_df)} ulasan ke {config.RAW_DATASET_PATH}")
                else:
                    print("[*] Melakukan scraping ulasan baru dari Google Play Store...")
                    needs_scrape = True
            else:
                print(f"[✓] Menggunakan dataset yang sudah tersedia di {config.RAW_DATASET_PATH} ({len(raw_check)} ulasan)")
        except Exception:
            needs_scrape = True

    if needs_scrape:
        cmd_scrape(args)

    # 2. Preprocess
    cmd_preprocess(args)

    # 3. Train & Benchmark
    cmd_train(args)

    # 4. Evaluate & Visualize
    cmd_evaluate(args)

    # 5. Quick Test Prediksi Contoh
    print("\n" + "=" * 60)
    print(" UJI PREDIKSI OTOMATIS (CONTOH KASUS BISNIS E-COMMERCE) ")
    print("=" * 60)
    contoh_ulasan = [
        "Barangnya bagus bgt, packing bubble wrap tebal dan pengiriman super cepat! Mantap seller!",
        "Aplikasi sering error pas mau checkout, saldo ketahan gajadi beli kecewa parah!",
        "Pengiriman standar, lumayan lah untuk harga segini."
    ]
    predictor = SentimentPredictor()
    for text in contoh_ulasan:
        res = predictor.predict_one(text)
        print(f"\nUlasan : \"{text}\"")
        print(f"Sentimen : {res['sentimen'].upper()} (Keyakinan: {res['keyakinan']:.2f}%)")
        print(f"Cleaned  : {res['teks_bersih']}")

    print("\n" + "=" * 60)
    print(" SELURUH PIPELINE NLP SELESAI DENGAN SUKSES! ")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CLI Proyek Analisis Sentimen E-Commerce Indonesia (End-to-End NLP Pipeline)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")

    # Command: run-all
    p_run = subparsers.add_parser("run-all", help="Jalankan semua tahap pipeline secara otomatis")
    p_run.add_argument("--app", default=config.DEFAULT_APP_ID, help="App ID di Play Store")
    p_run.add_argument("--count", type=int, default=config.DEFAULT_SCRAPE_COUNT, help="Jumlah ulasan scraping")
    p_run.add_argument("--force-scrape", action="store_true", help="Paksa scrape ulang data dari Play Store")

    # Command: scrape
    p_scrape = subparsers.add_parser("scrape", help="Scraping ulasan dari Google Play Store")
    p_scrape.add_argument("--platform", choices=list(config.ECOMMERCE_PLATFORMS.keys()) + ["all"], default=None,
                          help="Pilih platform (shopee, tokopedia, lazada, blibli, tiktok_shop, bukalapak, zalora, atau 'all' untuk batch paralel)")
    p_scrape.add_argument("--app", default=None, help="Custom App ID di Play Store (opsional jika platform ditentukan)")
    p_scrape.add_argument("--count", type=int, default=config.DEFAULT_SCRAPE_COUNT, help="Jumlah target ulasan")
    p_scrape.add_argument("--workers", type=int, default=config.SCRAPE_MAX_WORKERS, help="Jumlah thread paralel untuk scraping")
    p_scrape.add_argument("--no-filter-tiktok", action="store_true", help="Nonaktifkan filter transaksi belanja TikTok")

    # Command: preprocess
    subparsers.add_parser("preprocess", help="Pra-pemrosesan teks dan pembuatan cache Sastrawi")

    # Command: train
    subparsers.add_parser("train", help="Latih model TF-IDF dan algoritma ML")

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Buat visualisasi Confusion Matrix & WordCloud")

    # Command: predict
    p_pred = subparsers.add_parser("predict", help="Prediksi sentimen untuk satu kalimat ulasan")
    p_pred.add_argument("-t", "--text", required=True, help="Teks ulasan yang akan dianalisis")

    # Command: interactive
    subparsers.add_parser("interactive", help="Mode uji coba sentimen langsung di terminal")

    args = parser.parse_args()

    if args.command == "run-all" or args.command is None:
        cmd_run_all(args)
    elif args.command == "scrape":
        cmd_scrape(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
