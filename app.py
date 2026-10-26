"""
Entry point utama untuk menjalankan Web Dashboard Analisis Sentimen (Streamlit).
Penggunaan:
    streamlit run app.py
atau melalui CLI:
    python main.py dashboard
"""
import sys
from pathlib import Path

# Tambahkan direktori root proyek ke sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dashboard import main

if __name__ == "__main__":
    main()
