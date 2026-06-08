import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("output/final_merge.csv")

# =========================
# FORMAT TANGGAL
# =========================

df["date"] = pd.to_datetime(df["date"])

# =========================
# STYLE
# =========================

sns.set_style("whitegrid")

# ==================================================
# GRAFIK 1
# SENTIMENT INDEX
# ==================================================

plt.figure(figsize=(12,6))

plt.plot(
    df["date"],
    df["sentiment_index"]
)

plt.title("Pergerakan Sentiment Index Threads IHSG")

plt.xlabel("Tanggal")

plt.ylabel("Sentiment Index")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/grafik_sentiment_index.png"
)

plt.show()

# ==================================================
# GRAFIK 2
# RETURN IHSG
# ==================================================

plt.figure(figsize=(12,6))

plt.plot(
    df["date"],
    df["return"]
)

plt.title("Pergerakan Return IHSG")

plt.xlabel("Tanggal")

plt.ylabel("Return (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/grafik_return_ihsg.png"
)

plt.show()

# ==================================================
# GRAFIK 3
# PERBANDINGAN SENTIMEN VS IHSG
# ==================================================

fig, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(
    df["date"],
    df["sentiment_index"],
    label="Sentiment Index"
)

ax1.set_ylabel(
    "Sentiment Index"
)

ax1.set_xlabel(
    "Tanggal"
)

ax2 = ax1.twinx()

ax2.plot(
    df["date"],
    df["return"],
    label="Return IHSG"
)

ax2.set_ylabel(
    "Return (%)"
)

plt.title(
    "Sentiment Threads vs Return IHSG"
)

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig(
    "output/perbandingan_sentimen_vs_ihsg.png"
)

plt.show()

# ==================================================
# GRAFIK 4
# SCATTER KORELASI
# ==================================================

plt.figure(figsize=(8,6))

sns.regplot(
    data=df,
    x="sentiment_index",
    y="return"
)

plt.title(
    "Hubungan Sentiment Index dan Return IHSG"
)

plt.xlabel(
    "Sentiment Index"
)

plt.ylabel(
    "Return IHSG (%)"
)

plt.tight_layout()

plt.savefig(
    "output/scatter_korelasi.png"
)

plt.show()

print("\nVisualisasi berhasil dibuat.")