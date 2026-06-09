import pandas as pd
import re
import sys

def clean_threads_csv(input_path, output_path=None):
    df = pd.read_csv(input_path, encoding="utf-8", dtype=str).fillna("")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Drop single-value columns (e.g. 'keyword' col where every row = 'IHSG')
    single_val_cols = [c for c in df.columns if df[c].nunique() == 1]
    if single_val_cols:
        df = df.drop(columns=single_val_cols)
        print(f"Dropped single-value columns: {single_val_cols}")

    if "text" in df.columns:
        # Replace non-breaking spaces
        df["text"] = df["text"].str.replace("\u00a0", " ", regex=False)

        # Remove scraper UI artifacts: "Translate 1/9", "Translate", etc.
        df["text"] = df["text"].str.replace(r"\s*Translate\s+\d+/\d+", "", regex=True)
        df["text"] = df["text"].str.replace(r"\s*Translate\b", "", regex=True)

        # Strip trailing username echo (e.g. "\nsahamcuan_everyday6" at end of text)
        if "username" in df.columns:
            def strip_username(row):
                if not row["username"] or not row["text"]:
                    return row["text"]
                u = re.escape(str(row["username"]))
                return re.sub(rf"\n{u}\s*$", "", str(row["text"]))
            df["text"] = df.apply(strip_username, axis=1)

        # Collapse 3+ newlines into 2
        df["text"] = df["text"].str.replace(r"\n{3,}", "\n\n", regex=True)

        # Trim whitespace
        df["text"] = df["text"].str.strip()

    # Remove duplicate posts (exact text match, keep first)
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    print(f"Removed {before - len(df)} duplicate rows")

    if output_path is None:
        output_path = input_path.replace(".csv", "_cleaned.csv")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_threads.py <threads_IHSG_20260505_113515_V2.csv> [cleaned_threads_IHSG_20260505_113515_V2.csv]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    clean_threads_csv(inp, out)