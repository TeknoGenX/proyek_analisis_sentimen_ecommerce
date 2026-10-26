import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from google_play_scraper import Sort, reviews
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("="*50)
print("MEMULAI PIPELINE ANALISIS SENTIMEN NLP")
print("="*50)

# ==========================================
# 1. SCRAPING DATA DARI GOOGLE PLAY STORE
# ==========================================
print("\n[1/6] Mengambil ulasan dari Google Play Store...")
# Mengambil aplikasi Tokopedia sebagai studi kasus
app_id = 'com.tokopedia.tkpd' 
jumlah_ulasan = 300 # Dibatasi 300 agar cepat. Untuk proyek nyata, gunakan 2000+

hasil_scraping, continuation_token = reviews(
    app_id,
    lang='id', # Bahasa Indonesia
    country='id', 
    sort=Sort.NEWEST,
    count=jumlah_ulasan
)

# Menyimpan ke dalam DataFrame
df = pd.DataFrame(hasil_scraping)
df = df[['content', 'score']] # Kita hanya butuh teks ulasan dan rating bintang
df.rename(columns={'content': 'teks_ulasan', 'score': 'rating'}, inplace=True)

# Membuat Label Otomatis Berdasarkan Rating
def beri_label(rating):
    if rating <= 2:
        return 'negatif'
    elif rating == 3:
        return 'netral'
    else:
        return 'positif'

df['label'] = df['rating'].apply(beri_label)
print(f"Berhasil mengumpulkan {len(df)} ulasan.")
print(df['label'].value_counts())

# ==========================================
# 2. PRA-PEMROSESAN TEKS (DENGAN CACHE)
# ==========================================
print("\n[2/6] Memulai pra-pemrosesan teks (Sastrawi)... Ini mungkin memakan waktu.")
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

factory_stop = StopWordRemoverFactory()
stopword = factory_stop.create_stop_word_remover()

stem_cache = {} # Cache untuk mempercepat Sastrawi

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    text = stopword.remove(text)
    
    stemmed_words = []
    for word in text.split():
        if word in stem_cache:
            stemmed_words.append(stem_cache[word])
        else:
            stem_word = stemmer.stem(word)
            stem_cache[word] = stem_word
            stemmed_words.append(stem_word)
            
    return ' '.join(stemmed_words)

df['clean_review'] = df['teks_ulasan'].apply(clean_text)
# Hapus baris yang teksnya kosong setelah dibersihkan
df = df[df['clean_review'] != '']
print("Pra-pemrosesan selesai!")

# ==========================================
# 3. EKSTRAKSI FITUR & SPLIT DATA
# ==========================================
print("\n[3/6] Ekstraksi fitur menggunakan TF-IDF...")
X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# 4. PEMODELAN MACHINE LEARNING
# ==========================================
print("\n[4/6] Melatih model Logistic Regression (Balanced)...")
# Menggunakan class_weight='balanced' untuk menangani data imbalance
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\n=== Laporan Klasifikasi ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 5. VISUALISASI: CONFUSION MATRIX
# ==========================================
print("\n[5/6] Menyimpan gambar Confusion Matrix...")
labels_urut = ['negatif', 'netral', 'positif']
cm = confusion_matrix(y_test, y_pred, labels=labels_urut)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_urut, yticklabels=labels_urut,
            annot_kws={"size": 14})
plt.title('Confusion Matrix: Prediksi vs Aktual', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Label Aktual (Benar)')
plt.xlabel('Label Prediksi (Tebakan Model)')
plt.tight_layout()
plt.savefig('ConfusionMatrix.png', dpi=300)
plt.close() # Tutup memori plot
print("-> Disimpan sebagai 'ConfusionMatrix.png'")

# ==========================================
# 6. VISUALISASI: WORDCLOUD (SENTIMEN NEGATIF)
# ==========================================
print("\n[6/6] Menyimpan gambar WordCloud untuk sentimen negatif...")
ulasan_negatif = df[df['label'] == 'negatif']['clean_review']

if len(ulasan_negatif) > 0:
    semua_teks_negatif = " ".join(review for review in ulasan_negatif)
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Reds', max_words=100).generate(semua_teks_negatif)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Pain Points: Kata Terbanyak di Ulasan Negatif', fontsize=16)
    plt.tight_layout()
    plt.savefig('WordCloud_Negatif.png', dpi=300)
    plt.close()
    print("-> Disimpan sebagai 'WordCloud_Negatif.png'")
else:
    print("-> Tidak ada ulasan negatif untuk dijadikan WordCloud.")

print("\n="*50)
print("PIPELINE SELESAI! Silakan cek folder Anda untuk melihat file gambar.")
print("="*50)