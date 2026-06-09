import pandas as pd

import pandas as pd

df = pd.read_csv(
    "output/preprocessed.csv"
)


positif = {"cuan", "uptrend", "bullish", "rebound", "akumulasi", "serok", "terbang", "hijau", "naik", "untung", "bagus", "mantap", "roket", "hold", "dividen", "profit", "buy", "beli", "murah", "diskon", "potensi", "peluang", "aman"}
negatif = {"nyangkut", "cutloss", "cl", "turun", "longsor", "junam", "anjlok", "downtrend", "bearish", "distribusi", "merah", "rugi", "parah", "nyungsep", "arb", "bawah", "koreksi", "jatuh", "jual", "sell", "mahal", "hancur", "jebol", "gagal", "panik"}


# Ubah Tokens Menjadi List
import ast

df['tokens'] = df['tokens'].apply(ast.literal_eval)


# Buat Function Sentiment dengan Fuzzy Matching
import difflib

# Dictionary cache agar proses fuzzy tidak lambat karena diulang-ulang
fuzzy_cache = {}

def hitung_sentimen(tokens):
    score = 0
    for kata in tokens:
        # Gunakan memori cache
        if kata in fuzzy_cache:
            score += fuzzy_cache[kata]
            continue
            
        # Fast Exact Match
        if kata in positif:
            fuzzy_cache[kata] = 1
            score += 1
            continue
        elif kata in negatif:
            fuzzy_cache[kata] = -1
            score -= 1
            continue
            
        # Fuzzy Match (Toleransi Typo 80%)
        match_pos = difflib.get_close_matches(kata, positif, n=1, cutoff=0.8)
        if match_pos:
            fuzzy_cache[kata] = 1
            score += 1
            continue
            
        match_neg = difflib.get_close_matches(kata, negatif, n=1, cutoff=0.8)
        if match_neg:
            fuzzy_cache[kata] = -1
            score -= 1
            continue
            
        # Jika bukan kata sentimen
        fuzzy_cache[kata] = 0

    return score


# Hitung Skor Lexicon
df['score_lexicon'] = df['tokens'].apply(
    hitung_sentimen
)


# Buat Label Lexicon
def label_lexicon(score):
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    else:
        return "Netral"
    
df['sentiment_lexicon'] = df['score_lexicon'].apply(
    label_lexicon
)

# ==========================================
# 2. INDOBERT SENTIMENT ANALYSIS
# ==========================================
print("Loading IndoBERT model (Hanya lambat saat download pertama kali)...")
from transformers import pipeline
import os

try:
    if os.path.exists("./models/indobert_finetuned"):
        print("Mendeteksi model Fine-Tuned lokal! Menggunakan model dari ./models/indobert_finetuned")
        model_path = "./models/indobert_finetuned"
    else:
        print("Model lokal tidak ditemukan. Mengunduh model default mdhugol/indonesia-bert-sentiment-classification...")
        model_path = "mdhugol/indonesia-bert-sentiment-classification"
        
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sentiment_pipeline = None

def predict_indobert(text):
    if not sentiment_pipeline:
        return "Netral", 0
        
    text = str(text)[:500] # Potong agar tidak melebihi 512 token
    if not text.strip():
        return "Netral", 0
    try:
        res = sentiment_pipeline(text)[0]
        label = res['label']
        # Default mapping model mdhugol
        if label == "LABEL_0" or label.lower() == "positive":
            return "Positif", 1
        elif label == "LABEL_2" or label.lower() == "negative":
            return "Negatif", -1
        else:
            return "Netral", 0
    except Exception as e:
        return "Netral", 0

print("Memulai prediksi IndoBERT pada data (Memakan waktu 1-5 menit)...")
indobert_results = df['text'].apply(predict_indobert)
df['sentiment_indobert'] = [r[0] for r in indobert_results]
df['score_indobert'] = [r[1] for r in indobert_results]


# Simpan Hasil
df.to_csv(
    "output/sentiment_result.csv",
    index=False
)
print("Sentiment analisis selesai")