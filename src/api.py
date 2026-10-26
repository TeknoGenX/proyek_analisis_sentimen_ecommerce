"""
REST API Server untuk Analisis Sentimen E-Commerce Indonesia.
Menggunakan FastAPI dan Uvicorn untuk melayani endpoint inferensi real-time,
prediksi batch, dan metrik analitik sentimen komparatif.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config
from src.predictor import SentimentPredictor

logger = logging.getLogger(__name__)

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="API Analisis Sentimen E-Commerce Indonesia",
    description=(
        "Production-ready REST API untuk klasifikasi sentimen ulasan e-commerce (Shopee, Tokopedia, "
        "Lazada, Blibli, TikTok Shop, Bukalapak, Zalora). "
        "Mendukung inferensi tunggal, batch, evaluasi model ML, dan benchmark Net Sentiment Score (NSS)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Aktifkan CORS Middleware untuk kemudahan integrasi Frontend/Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (lazy-loaded saat startup atau pemanggilan)
_predictor: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """Mengambil atau menginisialisasi singleton SentimentPredictor."""
    global _predictor
    if _predictor is None:
        if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model atau TF-IDF vectorizer belum dilatih. Jalankan 'python main.py train' terlebih dahulu."
            )
        try:
            _predictor = SentimentPredictor()
        except Exception as e:
            logger.error(f"Gagal memuat SentimentPredictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gagal memuat model machine learning: {e}"
            )
    return _predictor


# =========================================================================
# Pydantic Schemas untuk Request & Response
# =========================================================================

class SentimentRequest(BaseModel):
    text: str = Field(..., example="Barang sangat bagus, packing aman dan pengiriman cepat sekali!")


class SentimentBatchRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        example=[
            "Barang sangat bagus, pengiriman cepat mantap!",
            "Aplikasi sering error pas checkout, kecewa parah!",
            "Biasa saja lumayan untuk harga segini."
        ]
    )


class SentimentResponse(BaseModel):
    teks_asli: str
    teks_bersih: str
    sentimen: str
    keyakinan: float
    probabilitas: Dict[str, float]


class SentimentBatchResponse(BaseModel):
    total: int
    summary: Dict[str, int]
    rata_rata_keyakinan: float
    results: List[SentimentResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    vectorizer_path: str
    target_platforms: List[str]


# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/", tags=["General"])
def root():
    """Root endpoint dengan ringkasan API dan link dokumentasi."""
    return {
        "message": "Selamat datang di API Analisis Sentimen E-Commerce Indonesia",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict-batch",
            "analyze_aspects": "POST /analyze-aspects",
            "model_metrics": "GET /metrics",
            "platform_benchmarks": "GET /platforms",
            "executive_report_html": "GET /report/html",
            "generate_report": "POST /report/generate"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Memeriksa status kesiapan server dan ketersediaan model."""
    is_ready = config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_path=str(config.MODEL_PATH),
        vectorizer_path=str(config.VECTORIZER_PATH),
        target_platforms=list(config.ECOMMERCE_PLATFORMS.keys())
    )


@app.post("/predict", response_model=SentimentResponse, tags=["Inference"])
def predict_sentiment(req: SentimentRequest):
    """
    Prediksi sentimen untuk satu kalimat teks ulasan pengguna.
    Mengembalikan label sentimen ('positif', 'netral', 'negatif'), teks hasil pembersihan,
    skor keyakinan (confidence %), dan rincian probabilitas tiap kelas.
    """
    predictor = get_predictor()
    res = predictor.predict_one(req.text)
    return SentimentResponse(
        teks_asli=res["teks_asli"],
        teks_bersih=res["teks_bersih"],
        sentimen=res["sentimen"],
        keyakinan=res["keyakinan"],
        probabilitas=res.get("probabilitas", {})
    )


@app.post("/predict-batch", response_model=SentimentBatchResponse, tags=["Inference"])
def predict_batch_sentiment(req: SentimentBatchRequest):
    """
    Prediksi sentimen untuk sekumpulan teks ulasan sekaligus (batch).
    Mengembalikan daftar hasil prediksi beserta agregasi ringkasan sentimen.
    """
    if not req.texts:
        return SentimentBatchResponse(
            total=0,
            summary={"positif": 0, "netral": 0, "negatif": 0},
            rata_rata_keyakinan=0.0,
            results=[]
        )

    predictor = get_predictor()
    raw_results = predictor.predict_batch(req.texts)

    summary = {"positif": 0, "netral": 0, "negatif": 0}
    total_conf = 0.0
    formatted_results = []

    for r in raw_results:
        sentimen = r["sentimen"]
        summary[sentimen] = summary.get(sentimen, 0) + 1
        total_conf += r["keyakinan"]
        formatted_results.append(SentimentResponse(
            teks_asli=r["teks_asli"],
            teks_bersih=r["teks_bersih"],
            sentimen=sentimen,
            keyakinan=r["keyakinan"],
            probabilitas=r.get("probabilitas", {})
        ))

    avg_conf = round(total_conf / len(raw_results), 2) if raw_results else 0.0

    return SentimentBatchResponse(
        total=len(raw_results),
        summary=summary,
        rata_rata_keyakinan=avg_conf,
        results=formatted_results
    )


@app.post("/analyze-aspects", tags=["Aspect-Based Sentiment (ABSA)"])
def analyze_aspect_sentiment(req: SentimentRequest):
    """
    Menganalisis sentimen berbasis aspek (ABSA) pada satu teks ulasan e-commerce.
    Mendeteksi aspek: pengiriman, kualitas_produk, harga_promo, pelayanan_cs, aplikasi_sistem.
    """
    predictor = get_predictor()
    return predictor.analyze_aspects(req.text)


@app.get("/metrics", tags=["Analytics & Evaluation"])
def get_model_evaluation_metrics():
    """
    Mengambil metrik evaluasi model machine learning (akurasi, F1-Score, confusion matrix)
    dari file evaluation_metrics.json.
    """
    if not config.METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File evaluation_metrics.json belum tersedia. Jalankan 'python main.py train' terlebih dahulu."
        )
    try:
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal membaca metrik: {e}")


@app.get("/platforms", tags=["Analytics & Evaluation"])
def get_platform_benchmarks():
    """
    Mengambil metrik komparasi Net Sentiment Score (NSS) dan proporsi sentimen
    antar 7 platform e-commerce dari file komparasi_metrics.json.
    """
    if not config.COMPARISON_METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File komparasi_metrics.json belum tersedia. Jalankan 'python main.py evaluate' terlebih dahulu."
        )
    try:
        with open(config.COMPARISON_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal membaca data platform: {e}")


@app.get("/report/html", response_class=HTMLResponse, tags=["Executive Reports"])
def get_executive_report_html():
    """
    Mengambil Laporan Analitik Bisnis Eksekutif dalam format HTML lengkap (siap cetak PDF).
    """
    report_path = config.OUTPUTS_DIR / "laporan_eksekutif_sentimen.html"
    if not report_path.exists():
        from src.report_generator import generate_business_report
        generate_business_report(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/report/generate", tags=["Executive Reports"])
def trigger_generate_report():
    """
    Memicu pembuatan ulang Laporan Analitik Bisnis Eksekutif terbaru.
    """
    from src.report_generator import generate_business_report
    try:
        report_path = generate_business_report()
        return {
            "status": "success",
            "message": "Laporan eksekutif berhasil dibuat",
            "report_path": str(report_path),
            "html_url": "/report/html"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal membuat laporan eksekutif: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
