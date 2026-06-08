import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Load data
df = pd.read_csv(
    "data/threads_IHSG_20260505_113515_V2_cleaned.csv"
)


# Case Folding
def casefolding(text):

    text = str(text)

    return text.lower()


# Cleaning
def cleaning(text):

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Tokenizing
def tokenizing(text):
    return text.split()


# Stopword
factory = StopWordRemoverFactory()

stopwords = set(
    factory.get_stop_words()
)

custom_stopwords = {
    "ihsg",
    "saham",
    "stockbit",
    "idx"
}

stopwords.update(custom_stopwords)


# Remove Stopword
def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word not in stopwords
    ]


# Stemming
factory = StemmerFactory()

stemmer = factory.create_stemmer()

def stemming(tokens):

    hasil = []

    for token in tokens:

        hasil.append(
            stemmer.stem(token)
        )

    return hasil


# Eksekusi Pipeline
df['text_clean'] = df['text']

df['text_clean'] = (
    df['text_clean']
    .apply(casefolding)
    .apply(cleaning)
)

df['tokens'] = (
    df['text_clean']
    .apply(tokenizing)
)

df['tokens'] = (
    df['tokens']
    .apply(remove_stopwords)
)

df['tokens'] = (
    df['tokens']
    .apply(stemming)
)


# Simpan
df.to_csv(
    "output/preprocessed.csv",
    index=False
)

print("Preprocessing selesai")

print(
    df[['text', 'tokens']]
    .head(10)
)
