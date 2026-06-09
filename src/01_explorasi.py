import pandas as pd

df = pd.read_csv(
    "data/threads_IHSG_20260505_113515_V2_cleaned.csv"
)

print(df.head())

print("\nJumlah data:")
print(len(df))

print("\nKolom:")
print(df.columns)

print("\nMissing Value:")
print(df.isnull().sum())


df['timestamp'] = pd.to_datetime(
    df['timestamp']
)
df['date'] = df['timestamp'].dt.date
print(df[['timestamp','date']].head())