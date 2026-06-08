import pandas as pd

import pandas as pd

df = pd.read_csv(
    "output/preprocessed.csv"
)


# Kamus Sentimen Sementara
positif = {
    "naik",
    "untung",
    "cuan",
    "bagus",
    "mantap"
}

negatif = {
    "turun",
    "rugi",
    "anjlok",
    "merah",
    "parah"
}


# Ubah Tokens Menjadi List
import ast

df['tokens'] = df['tokens'].apply(ast.literal_eval)


# Buat Function Sentiment
def hitung_sentimen(tokens):

    score = 0

    for kata in tokens:

        if kata in positif:
            score += 1

        elif kata in negatif:
            score -= 1

    return score


# Hitung Skor
df['score'] = df['tokens'].apply(
    hitung_sentimen
)


# Buat Label
def label(score):

    if score > 0:
        return "Positif"

    elif score < 0:
        return "Negatif"

    else:
        return "Netral"
    

# Buat Kolom Sentiment
df['sentiment'] = df['score'].apply(
    label
)


# Cek Hasil
print(
    df[
        ['text','score','sentiment']
    ].head(20)
)


# Simpan Hasil
df.to_csv(
    "output/sentiment_result.csv",
    index=False
)