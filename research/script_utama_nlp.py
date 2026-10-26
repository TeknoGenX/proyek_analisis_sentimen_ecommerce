import pandas as pd
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from google_play_scraper import reviews, Sort
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def print_header(title):
    print("\n" + "="*60)
    print(f" {title.upper()} ".center(60, "="))
    print("="*60)

# ==========================================
# 1. DATA COLLECTION (SCRAPING)
# ==========================================
print_header("1. Mengumpulkan Data Ulasan E-Commerce")
APP_ID = 'com.tokopedia.tkpd'  # Ganti 'com.shopee.id' jika ingin Shopee
TOTAL_REVIEWS = 800            # Jumlah data untuk eksperimen (naikkan ke 3000+ untuk produksi)

print(f"[*] Melakukan scraping {TOTAL_REVIEWS} ulasan dari aplikasi '{APP_ID}'...")
result, _ = reviews(
    APP_ID,
    lang='id', 
    country='id',
    sort=Sort.NEWEST,
    count=TOTAL_REVIEWS
)

df = pd.DataFrame(result)[['content', 'score']]
df = df.rename(columns={'content': 'teks_ulasan', 'score': 'rating'})

# Pelabelan Sentimen berdasar rating
def classify_sentiment(rating):
    if rating <= 2: return 'negatif'
    elif rating == 3: return 'netral'
    else: return 'positif'

df['label'] = df['rating'].apply(classify_sentiment)

# Simpan dataset mentah
df.to_csv('dataset_ulasan.csv', index=False)
print(f"[✓] Berhasil menyimpan dataset_ulasan.csv ({len(df)} baris)")
print(f"[*] Distribusi Sentimen:\n{df['label'].value_counts()}")


# ==========================================
# 2. TEXT PREPROCESSING (DENGAN CACHING)
# ==========================================
print_header("2. Pembersihan & Pemrosesan Teks (Sastrawi)")
print("[*] Inisialisasi Sastrawi Stemmer & Stopword (Membutuhkan beberapa detik)...")

factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
factory_stop = StopWordRemoverFactory()
stopword = factory_stop.create_stop_word_remover()

stem_cache = {} # Dictionary untuk caching

def preprocess_text(text):
    # Case folding & Hapus karakter non-alfabet
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    # Hapus stopword
    text = stopword.remove(text)
    
    # Stemming dengan Cache (sangat mempercepat komputasi)
    stemmed_words = []
    for word in text.split():
        if word in stem_cache:
            stemmed_words.append(stem_cache[word])
        else:
            stem_word = stemmer.stem(word)
            stem_cache[word] = stem_word
            stemmed_words.append(stem_word)
    return ' '.join(stemmed_words)

print(f"[*] Memproses teks... (Mohon tunggu, memproses {len(df)} ulasan)")
df['clean_review'] = df['teks_ulasan'].apply(preprocess_text)
df = df[df['clean_review'] != ''] # Hapus jika ada ulasan yang kosong setelah dibersihkan
print(f"[✓] Pemrosesan selesai. Total kosakata dalam cache Sastrawi: {len(stem_cache)}")


# ==========================================
# 3. FEATURE EXTRACTION & MODELING
# ==========================================
print_header("3. Pelatihan Model AI (Machine Learning)")

X = df['clean_review']
y = df['label']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
print("[*] Mengubah teks menjadi vektor (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Logistic Regression (Balanced untuk menangani kelas ulasan negatif yang sedikit)
print("[*] Melatih algoritma Logistic Regression...")
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Prediksi & Evaluasi
y_pred = model.predict(X_test_tfidf)
print("\n[✓] Laporan Evaluasi Bisnis (Classification Report):")
print(classification_report(y_test, y_pred, zero_division=0))


# ==========================================
# 4. EXPORT VISUALISASI BUSINESS INSIGHTS
# ==========================================
print_header("4. Menghasilkan Visualisasi Data")

# 4A. WordCloud Sentimen Negatif (Pain Points)
print("[*] Membuat WordCloud untuk sentimen negatif...")
ulasan_negatif = df[df['label'] == 'negatif']['clean_review']
if not ulasan_negatif.empty:
    teks_negatif = " ".join(review for review in ulasan_negatif)
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Reds', max_words=100).generate(teks_negatif)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Pain Points: Keluhan Terbanyak Pengguna {APP_ID.split(".")[1].title()}', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('WordCloud_Keluhan.png', dpi=300)
    plt.close()
    print("  [✓] Disimpan sebagai 'WordCloud_Keluhan.png'")

# 4B. Confusion Matrix
print("[*] Membuat Confusion Matrix...")
kelas = ['negatif', 'netral', 'positif']
cm = confusion_matrix(y_test, y_pred, labels=kelas)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negatif', 'Netral', 'Positif'], 
            yticklabels=['Negatif', 'Netral', 'Positif'], 
            annot_kws={"size": 14})
plt.title('Evaluasi Model: Prediksi vs Aktual', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Aktual (Benar)', fontsize=14)
plt.xlabel('Prediksi Model', fontsize=14)
plt.tight_layout()
plt.savefig('ConfusionMatrix.png', dpi=300)
plt.close()
print("  [✓] Disimpan sebagai 'ConfusionMatrix.png'")


# ==========================================
# 5. MENYIMPAN MODEL (DEPLOYMENT READY)
# ==========================================
print_header("5. Menyimpan Model untuk Deployment")
joblib.dump(model, 'model_sentimen.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("[✓] Model (model_sentimen.pkl) dan Vectorizer (tfidf_vectorizer.pkl) berhasil disimpan.")


# ==========================================
# 6. MODE INTERAKTIF (LIVE TESTING)
# ==========================================
print_header("6. Pengujian Sentimen Langsung (Interactive Mode)")
print("Ketikkan kalimat ulasan E-Commerce (Ketik 'keluar' untuk mengakhiri program).")

while True:
    input_text = input("\nMasukkan Ulasan: ")
    if input_text.lower() == 'keluar':
        print("\nTerima kasih telah menggunakan program ini! Sampai jumpa.")
        break
        
    if input_text.strip() == "":
        continue
        
    # Preprocess text yang diinput
    clean_input = preprocess_text(input_text)
    # Transform ke TF-IDF
    vectorized_input = vectorizer.transform([clean_input])
    # Prediksi
    prediction = model.predict(vectorized_input)[0]
    
    # Warna terminal
    if prediction == 'positif':
        color = '\033[92m' # Hijau
    elif prediction == 'negatif':
        color = '\033[91m' # Merah
    else:
        color = '\033[93m' # Kuning
    reset = '\033[0m'
    
    print(f"Hasil Analisis AI : {color}[ SENTIMEN {prediction.upper()} ]{reset}")