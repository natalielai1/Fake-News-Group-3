#!/usr/bin/env python3

import re
import pandas as pd

# --- Simple regex tokenizer ---
# Splits text into words, keeping apostrophes inside words (e.g., "don't")
WORD_RE = re.compile(r"[A-Za-z0-9']+")

def simple_tokenize(s: str):
    """Convert a text string into a list of word-like tokens."""
    if s is None:
        return []
    return WORD_RE.findall(str(s))

def main():
    # 1) Load the Kaggle dataset
    df = pd.read_csv("WELFake_Dataset.csv")

    # 2) Rename the "Unnamed: 0" column to "id"
    #    (We know this is the serial/index column in the Kaggle dataset.)
    df = df.rename(columns={"Unnamed: 0": "id"})

    # 3) Tokenize the "text" column to create a new "tokens" column
    df["tokens"] = df["text"].astype(str).apply(simple_tokenize)

    # 4) Keep only the necessary columns (plus title/text for later use)
    out = df[["id", "title", "text", "tokens", "label"]]

    # 5) Save the cleaned DataFrame to a new CSV
    out.to_csv("cleaned_fake_news.csv", index=False)

    # 6) Print confirmation so we know it worked
    print(f"Saved {len(out)} rows → cleaned_fake_news.csv")
    print("Columns:", list(out.columns))

if __name__ == "__main__":
    main()