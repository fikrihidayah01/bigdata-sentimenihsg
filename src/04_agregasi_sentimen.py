import pandas as pd

df = pd.read_csv(
    "output/sentiment_result.csv"
)

df['timestamp'] = pd.to_datetime(
    df['timestamp']
)


# Konversi Timestamp
df['date'] = (
    df['timestamp']
    .dt.date
)


# Hitung Sentimen Per Hari
hasil = pd.crosstab(
    df['date'],
    df['sentiment']
)


# Lihat Hasil
print(hasil)



# Buat Sentiment Index
hasil['sentiment_index'] = (
    hasil['Positif']
    -
    hasil['Negatif']
)



# Simpan Hasil
hasil.to_csv(
    "output/sentiment_daily.csv"
)