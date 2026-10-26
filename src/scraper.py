"""
Modul Scraping Ulasan dari Google Play Store untuk Analisis Komparasi E-Commerce.
Mendukung 7 platform: Shopee, Tokopedia, Lazada, Blibli, TikTok Shop, Bukalapak, Zalora.
Dilengkapi scraping batch paralel dan filtering ulasan transaksi TikTok Shop.
"""
import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from google_play_scraper import Sort, reviews

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rating_to_sentiment(rating: Union[int, float, str, None]) -> str:
    """Mengubah rating bintang 1-5 menjadi label sentimen secara aman."""
    if rating is None or pd.isna(rating):
        return "netral"
    try:
        r = float(rating)
    except (ValueError, TypeError):
        return "netral"

    if r <= 2.0:
        return "negatif"
    elif r == 3.0:
        return "netral"
    else:
        return "positif"


def filter_tiktok_commerce_reviews(
    df: pd.DataFrame, 
    keywords: Optional[List[str]] = None,
    text_column: str = "teks_ulasan"
) -> pd.DataFrame:
    """
    Memfilter ulasan TikTok agar hanya menyaring ulasan terkait belanja/transaksi (TikTok Shop).
    
    Args:
        df: DataFrame ulasan
        keywords: Daftar kata kunci filter (default dari config.TIKTOK_COMMERCE_KEYWORDS)
        text_column: Nama kolom ulasan
        
    Returns:
        DataFrame yang telah difilter
    """
    if df.empty or text_column not in df.columns:
        return df

    kw_list = keywords or config.TIKTOK_COMMERCE_KEYWORDS
    # Regex case-insensitive dengan non-capturing group dan word boundary untuk mencocokkan kata kunci belanja secara tepat
    pattern_str = r"\b(?:" + "|".join([re.escape(k.lower()) for k in kw_list]) + r")\b"
    pattern = re.compile(pattern_str, re.IGNORECASE)

    initial_count = len(df)
    filtered_df = df[df[text_column].astype(str).str.contains(pattern, na=False)].copy()
    logger.info(
        f"[TikTok Shop Filter] Menyaring {len(filtered_df)} ulasan transaksi dari {initial_count} ulasan umum TikTok."
    )
    return filtered_df


def resolve_platform_info(
    platform_key: Optional[str] = None, 
    app_id: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Menemukan key platform, nama platform, dan app_id dari argumen yang diberikan.
    
    Returns:
        Tuple of (platform_key, platform_name, app_id)
    """
    # 1. Jika platform_key diberikan
    if platform_key and platform_key.lower() in config.ECOMMERCE_PLATFORMS:
        key = platform_key.lower()
        info = config.ECOMMERCE_PLATFORMS[key]
        return key, info["name"], info["app_id"]

    # 2. Jika app_id diberikan, cari key yang sesuai
    if app_id:
        for k, v in config.ECOMMERCE_PLATFORMS.items():
            if v["app_id"] == app_id:
                return k, v["name"], app_id
        # Custom app_id di luar daftar default
        return app_id, app_id, app_id

    # 3. Default fallback
    def_key = config.DEFAULT_PLATFORM
    def_info = config.ECOMMERCE_PLATFORMS[def_key]
    return def_key, def_info["name"], def_info["app_id"]


def _fetch_reviews_filtered(
    target_app_id: str,
    lang: str,
    country: str,
    target_count: int,
    filter_score: Optional[int] = None,
    is_tiktok: bool = False,
    filter_tiktok: bool = True
) -> List[dict]:
    """Helper internal untuk mengambil ulasan dengan paginasi dan filter rating bintang tertentu."""
    collected = []
    continuation_token = None
    max_attempts = max(target_count * 8, 2000) if (is_tiktok and filter_tiktok) else max(target_count * 3, 600)
    fetched_total = 0

    while len(collected) < target_count and fetched_total < max_attempts:
        batch_size = min(200, max(50, (target_count - len(collected)) * 2 if is_tiktok else target_count - len(collected)))
        try:
            if continuation_token is None:
                raw_batch, continuation_token = reviews(
                    target_app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=batch_size,
                    filter_score_with=filter_score
                )
            else:
                raw_batch, continuation_token = reviews(
                    target_app_id,
                    continuation_token=continuation_token
                )
        except Exception as err:
            logger.warning(f"Batch fetch error (app: {target_app_id}, score={filter_score}): {err}")
            break

        if not raw_batch:
            break

        fetched_total += len(raw_batch)

        if is_tiktok and filter_tiktok:
            batch_df = pd.DataFrame(raw_batch)
            if "content" in batch_df.columns:
                batch_df = batch_df.rename(columns={"content": "teks_ulasan"})
                batch_filtered = filter_tiktok_commerce_reviews(batch_df)
                collected.extend(batch_filtered.to_dict("records"))
        else:
            collected.extend(raw_batch)

        if continuation_token is None:
            break

    return collected[:target_count]


def scrape_playstore_reviews(
    app_id: Optional[str] = None,
    platform_key: Optional[str] = None,
    count: int = config.DEFAULT_SCRAPE_COUNT,
    lang: str = config.SCRAPE_LANG,
    country: str = config.SCRAPE_COUNTRY,
    save_path: Optional[str] = None,
    filter_tiktok: bool = True,
    balanced: bool = False,
    filter_score: Optional[int] = None
) -> pd.DataFrame:
    """
    Mengambil data ulasan terbaru dari Google Play Store untuk satu platform.
    Mendukung scraping seimbang (balanced) antar 3 kelas sentimen dan filter rating bintang tertentu.
    
    Args:
        app_id: Application ID di Play Store (opsional jika platform_key diisi)
        platform_key: Key platform (shopee, tokopedia, lazada, blibli, tiktok_shop, bukalapak, zalora)
        count: Target jumlah ulasan yang ingin diambil
        lang: Bahasa ulasan (default: id)
        country: Negara asal ulasan (default: id)
        save_path: Lokasi penyimpanan CSV (opsional)
        filter_tiktok: Jika True dan platform adalah TikTok, terapkan filter transaksi e-commerce
        balanced: Jika True, seimbangkan proporsi ulasan negatif (1-2), netral (3), dan positif (4-5)
        filter_score: Opsional, filter hanya ulasan dengan bintang 1-5 tertentu (misal: 3 untuk netral)
        
    Returns:
        DataFrame berisi kolom ['teks_ulasan', 'rating', 'label', 'platform', 'tanggal', 'thumbs_up']
    """
    key, name, target_app_id = resolve_platform_info(platform_key, app_id)
    is_tiktok = (key == "tiktok_shop" or target_app_id == config.ECOMMERCE_PLATFORMS["tiktok_shop"]["app_id"])
    
    mode_desc = f" (Filter Bintang: {filter_score})" if filter_score else (" [BALANCED 3-KELAS]" if balanced else "")
    logger.info(f"Memulai scraping {count} ulasan{mode_desc} untuk platform [{name}] (App ID: {target_app_id})...")

    try:
        if filter_score is not None:
            raw_results = _fetch_reviews_filtered(
                target_app_id=target_app_id,
                lang=lang,
                country=country,
                target_count=count,
                filter_score=filter_score,
                is_tiktok=is_tiktok,
                filter_tiktok=filter_tiktok
            )
        elif balanced:
            # Proporsi seimbang 3 kelas: Negatif (1&2), Netral (3), Positif (4&5)
            n_neu = count // 3
            n_neg = count // 3
            n_pos = count - n_neu - n_neg

            neg_1 = n_neg // 2
            neg_2 = n_neg - neg_1
            pos_4 = n_pos // 2
            pos_5 = n_pos - pos_4

            r_neg1 = _fetch_reviews_filtered(target_app_id, lang, country, neg_1, filter_score=1, is_tiktok=is_tiktok, filter_tiktok=filter_tiktok)
            r_neg2 = _fetch_reviews_filtered(target_app_id, lang, country, neg_2, filter_score=2, is_tiktok=is_tiktok, filter_tiktok=filter_tiktok)
            r_neu = _fetch_reviews_filtered(target_app_id, lang, country, n_neu, filter_score=3, is_tiktok=is_tiktok, filter_tiktok=filter_tiktok)
            r_pos4 = _fetch_reviews_filtered(target_app_id, lang, country, pos_4, filter_score=4, is_tiktok=is_tiktok, filter_tiktok=filter_tiktok)
            r_pos5 = _fetch_reviews_filtered(target_app_id, lang, country, pos_5, filter_score=5, is_tiktok=is_tiktok, filter_tiktok=filter_tiktok)

            combined_raw = r_neg1 + r_neg2 + r_neu + r_pos4 + r_pos5
            import random
            random.seed(config.RANDOM_STATE)
            random.shuffle(combined_raw)
            raw_results = combined_raw[:count]
            logger.info(
                f"[{name}] Scraping Seimbang Selesai: Negatif={len(r_neg1)+len(r_neg2)}, "
                f"Netral={len(r_neu)}, Positif={len(r_pos4)+len(r_pos5)}."
            )
        elif is_tiktok and filter_tiktok:
            all_reviews = []
            continuation_token = None
            max_fetch_attempts = max(count * 8, 2000)
            fetched_total = 0
            
            while len(all_reviews) < count and fetched_total < max_fetch_attempts:
                batch_size = min(200, count * 3)
                if continuation_token is None:
                    raw_batch, continuation_token = reviews(
                        target_app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=batch_size
                    )
                else:
                    raw_batch, continuation_token = reviews(
                        target_app_id,
                        continuation_token=continuation_token
                    )
                    
                if not raw_batch:
                    break
                    
                fetched_total += len(raw_batch)
                
                # Format batch menjadi DataFrame untuk difilter
                batch_df = pd.DataFrame(raw_batch)
                if "content" in batch_df.columns:
                    batch_df = batch_df.rename(columns={"content": "teks_ulasan"})
                    batch_filtered = filter_tiktok_commerce_reviews(batch_df)
                    all_reviews.extend(batch_filtered.to_dict("records"))
                
                if continuation_token is None:
                    break
                    
            raw_results = all_reviews[:count]
            logger.info(f"[TikTok Shop] Berhasil mengumpulkan {len(raw_results)} ulasan transaksi.")
        else:
            raw_results, _ = reviews(
                target_app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=count
            )

        if not raw_results:
            raise ValueError(f"Tidak ada ulasan yang ditemukan untuk {name} (app_id: {target_app_id})")

        df = pd.DataFrame(raw_results)
        
        # Standardisasi kolom
        rename_map = {
            "content": "teks_ulasan",
            "score": "rating",
            "at": "tanggal",
            "thumbsUpCount": "thumbs_up"
        }
        df = df.rename(columns=rename_map)
        
        # Pastikan kolom wajib ada
        required_cols = ["teks_ulasan", "rating"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Kolom wajib '{col}' tidak ditemukan dalam data scraping.")

        # Tambahkan metadata platform
        df["platform"] = key

        # Opsional kolom tanggal & thumbs_up jika tersedia
        if "tanggal" not in df.columns:
            df["tanggal"] = pd.Timestamp.now()
        if "thumbs_up" not in df.columns:
            df["thumbs_up"] = 0

        # Bersihkan string ulasan kosong atau NaN
        df = df.dropna(subset=["teks_ulasan"])
        df = df[df["teks_ulasan"].astype(str).str.strip() != ""]

        # Validasi dan konversi rating secara aman
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df = df.dropna(subset=["rating"])
            df["rating"] = df["rating"].astype(int)
            df["label"] = df["rating"].apply(rating_to_sentiment)
        else:
            df["rating"] = 3
            df["label"] = "netral"

        # Pilih urutan kolom standar
        cols_order = ["teks_ulasan", "rating", "label", "platform", "tanggal", "thumbs_up"]
        existing_cols = [c for c in cols_order if c in df.columns]
        df = df[existing_cols]

        logger.info(f"Scraping [{name}] selesai! Total ulasan didapatkan: {len(df)}")
        logger.info(f"Distribusi Sentimen [{name}]:\n{df['label'].value_counts().to_dict()}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Dataset [{name}] berhasil disimpan ke: {save_path}")

        return df

    except Exception as e:
        logger.error(f"Gagal melakukan scraping Play Store untuk [{name}]: {e}")
        # 1. Coba gunakan file save_path lokal jika ada
        if save_path and Path(save_path).exists():
            logger.warning(f"Menggunakan dataset lokal di: {save_path}")
            return load_dataset(save_path)

        # 2. Jika platform adalah DEFAULT_PLATFORM (Tokopedia), gunakan RAW_DATASET_PATH
        if key == config.DEFAULT_PLATFORM and config.RAW_DATASET_PATH.exists():
            logger.warning(f"Menggunakan dataset lokal cadangan di: {config.RAW_DATASET_PATH}")
            fallback_df = load_dataset(config.RAW_DATASET_PATH)
            return fallback_df

        # 3. Jika platform lain, periksa apakah ada data platform terkait di dataset komparasi lokal
        if config.COMPARISON_RAW_DATASET_PATH.exists():
            comp_df = load_dataset(config.COMPARISON_RAW_DATASET_PATH)
            if "platform" in comp_df.columns and key in comp_df["platform"].values:
                logger.warning(f"Menggunakan ulasan platform '{key}' dari dataset komparasi lokal.")
                matched_df = comp_df[comp_df["platform"] == key].copy()
                return matched_df

        raise e



def scrape_multi_platforms(
    platforms: Optional[List[str]] = None,
    count_per_platform: int = config.DEFAULT_COMPARISON_SCRAPE_COUNT,
    max_workers: int = config.SCRAPE_MAX_WORKERS,
    save_path: Optional[str] = None,
    filter_tiktok: bool = True,
    balanced: bool = False,
    filter_score: Optional[int] = None
) -> pd.DataFrame:
    """
    Mengambil data ulasan dari beberapa platform e-commerce secara paralel.
    
    Args:
        platforms: Daftar key platform (contoh: ['shopee', 'tokopedia', 'blibli', ...]).
                   Jika None, akan mengambil seluruh 7 platform di config.ECOMMERCE_PLATFORMS.
        count_per_platform: Target jumlah ulasan per platform
        max_workers: Jumlah thread paralel untuk mempercepat scraping
        save_path: Lokasi penyimpanan CSV gabungan
        filter_tiktok: Aktifkan filtering transaksi untuk TikTok Shop
        balanced: Seimbangkan ulasan 3 kelas sentimen per platform
        filter_score: Filter rating bintang tertentu (1-5) per platform
        
    Returns:
        Consolidated DataFrame berisi ulasan gabungan seluruh platform.
    """
    target_platforms = platforms or list(config.ECOMMERCE_PLATFORMS.keys())
    output_path = save_path or str(config.COMPARISON_RAW_DATASET_PATH)
    
    mode_str = f" [BINTANG: {filter_score}]" if filter_score else (" [BALANCED 3-KELAS]" if balanced else "")
    logger.info(
        f"Memulai scraping batch paralel untuk {len(target_platforms)} platform{mode_str}: "
        f"{', '.join(target_platforms)} (target {count_per_platform} ulasan/platform)..."
    )

    platform_dfs: List[pd.DataFrame] = []

    def _scrape_worker(platform_key: str) -> Optional[pd.DataFrame]:
        try:
            df = scrape_playstore_reviews(
                platform_key=platform_key,
                count=count_per_platform,
                filter_tiktok=filter_tiktok,
                balanced=balanced,
                filter_score=filter_score,
                save_path=None
            )
            return df
        except Exception as err:
            logger.error(f"Error scraping platform '{platform_key}': {err}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_platform = {
            executor.submit(_scrape_worker, p): p for p in target_platforms
        }
        for future in concurrent.futures.as_completed(future_to_platform):
            p = future_to_platform[future]
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    platform_dfs.append(result_df)
                    logger.info(f"[Batch Worker] Platform '{p}' selesai dengan {len(result_df)} ulasan.")
            except Exception as exc:
                logger.error(f"[Batch Worker] Exception pada platform '{p}': {exc}")

    if not platform_dfs:
        raise RuntimeError("Semua proses scraping platform gagal atau tidak ada data yang berhasil ditarik.")

    combined_df = pd.concat(platform_dfs, ignore_index=True)

    # Simpan dataset konsolidasi
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Dataset gabungan multi-platform ({len(combined_df)} baris) berhasil disimpan ke: {output_path}")

    # Log ringkasan hasil
    summary = combined_df.groupby(["platform", "label"]).size().unstack(fill_value=0)
    logger.info(f"Ringkasan Komparasi Ulasan Per Platform:\n{summary.to_string()}")

    return combined_df


def load_dataset(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Memuat dataset ulasan dari file CSV lokal dengan kompatibilitas mundur.
    
    Args:
        csv_path: Path ke file CSV. Jika None, menggunakan config.RAW_DATASET_PATH.
        
    Returns:
        DataFrame siap olah dengan kolom standar teks_ulasan, rating, label, platform.
    """
    path = csv_path or config.RAW_DATASET_PATH
    if not Path(path).exists():
        # Fallback ke dataset komparasi HANYA jika csv_path tidak ditentukan (None)
        if csv_path is None and config.COMPARISON_RAW_DATASET_PATH.exists():
            path = config.COMPARISON_RAW_DATASET_PATH
        else:
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {path}")


    df = pd.read_csv(path)

    # Standardisasi kolom secara independen jika format berbeda
    if "content" in df.columns and "teks_ulasan" not in df.columns:
        df = df.rename(columns={"content": "teks_ulasan"})
    if "score" in df.columns and "rating" not in df.columns:
        df = df.rename(columns={"score": "rating"})

    if "teks_ulasan" not in df.columns:
        raise KeyError("Dataset harus memiliki kolom 'teks_ulasan' atau 'content'.")

    # Pastikan tipe rating aman dan beri label jika belum ada
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "label" not in df.columns and "rating" in df.columns:
        df["label"] = df["rating"].apply(rating_to_sentiment)

    # Jika platform belum ada (dataset lama), beri default
    if "platform" not in df.columns:
        df["platform"] = config.DEFAULT_PLATFORM

    return df
