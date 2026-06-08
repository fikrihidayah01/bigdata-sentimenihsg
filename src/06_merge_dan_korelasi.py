import pandas as pd

# =========================
# LOAD DATA
# =========================

sentiment = pd.read_csv(
    "output/sentiment_daily.csv"
)

ihsg = pd.read_csv(
    "output/ihsg_trend.csv"
)

# =========================
# FORMAT TANGGAL
# =========================

sentiment["date"] = pd.to_datetime(
    sentiment["date"]
)

ihsg["Date"] = pd.to_datetime(
    ihsg["Date"]
)

# =========================
# SAMAKAN NAMA KOLOM
# =========================

ihsg = ihsg.rename(
    columns={"Date": "date"}
)

# =========================
# MERGE
# =========================

merged = pd.merge(
    sentiment,
    ihsg,
    on="date",
    how="inner"
)

print("Jumlah data setelah merge:")
print(len(merged))

print()
print(merged.head())


# HITUNG KORELASI
print()
print("KORELASI")

korelasi = merged[
    ["sentiment_index", "return"]
].corr()

print(korelasi)



# SIMPAN HASIL MERGE
merged.to_csv(
    "output/final_merge.csv",
    index=False
)

print()
print("final_merge.csv berhasil dibuat")