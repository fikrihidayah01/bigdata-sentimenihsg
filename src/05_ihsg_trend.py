import pandas as pd

# ======================
# LOAD DATA
# ======================

ihsg = pd.read_csv("data/ihsg.csv")

# ======================
# UBAH FORMAT TANGGAL
# ======================

ihsg["Date"] = pd.to_datetime(
    ihsg["Date"],
    format="%m/%d/%Y"
)

# ======================
# BERSIHKAN PRICE
# ======================

ihsg["Price"] = (
    ihsg["Price"]
    .astype(str)
    .str.replace(",", "")
    .astype(float)
)

# ======================
# URUTKAN TANGGAL
# ======================

ihsg = ihsg.sort_values(
    by="Date"
)

# ======================
# HITUNG RETURN
# ======================

ihsg["return"] = (
    ihsg["Price"]
    .pct_change()
    * 100
)

# ======================
# LABEL TREND
# ======================

def trend(x):

    if pd.isna(x):
        return "Awal"

    elif x > 0:
        return "Naik"

    elif x < 0:
        return "Turun"

    else:
        return "Stabil"

ihsg["trend"] = (
    ihsg["return"]
    .apply(trend)
)

# ======================
# CEK HASIL
# ======================

print(
    ihsg[
        ["Date","Price","return","trend"]
    ].head(10)
)

# ======================
# SIMPAN
# ======================

ihsg.to_csv(
    "output/ihsg_trend.csv",
    index=False
)

print("\nBerhasil membuat ihsg_trend.csv")