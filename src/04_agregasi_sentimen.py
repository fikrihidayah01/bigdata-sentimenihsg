import pandas as pd

df = pd.read_csv(
    "output/sentiment_result.csv"
)

df['timestamp'] = pd.to_datetime(
    df['timestamp']
)


# Konversi Timestamp ke WIB (Asia/Jakarta)
if df['timestamp'].dt.tz is None:
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Jakarta')
else:
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Jakarta')

df['date'] = (
    df['timestamp']
    .dt.date
)


# Hitung Sentimen Per Hari
agregasi_lex = pd.crosstab(
    df['date'],
    df['sentiment_lexicon']
).reset_index()

agregasi_indo = pd.crosstab(
    df['date'],
    df['sentiment_indobert']
).reset_index()


# Pastikan kolom-kolom sentimen exist, kalau tidak, buat dummy 0
for col in ['Positif', 'Netral', 'Negatif']:
    if col not in agregasi_lex.columns:
        agregasi_lex[col] = 0
    if col not in agregasi_indo.columns:
        agregasi_indo[col] = 0

# Hitung Sentiment Index untuk Lexicon (Mengembalikan rumus original: Positif - Negatif)
agregasi_lex['sentiment_index_lexicon'] = agregasi_lex['Positif'] - agregasi_lex['Negatif']

# Hitung Sentiment Index untuk IndoBERT
agregasi_indo['sentiment_index_indobert'] = agregasi_indo['Positif'] - agregasi_indo['Negatif']

# Gabungkan keduanya
agregasi = pd.merge(agregasi_lex[['date', 'sentiment_index_lexicon', 'Positif', 'Netral', 'Negatif']],
                    agregasi_indo[['date', 'sentiment_index_indobert', 'Positif', 'Netral', 'Negatif']],
                    on='date', suffixes=('_lexicon', '_indobert'))



# Simpan Hasil
agregasi.to_csv(
    "output/sentiment_daily.csv"
)