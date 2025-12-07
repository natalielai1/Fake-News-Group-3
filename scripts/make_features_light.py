import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import csv

# Increase CSV limit
csv.field_size_limit(sys.maxsize)

def clean_source_text_simple(text):
    import re
    if not text: return ""
    reuters_pattern = r"^([A-Z\s]+)?\s*\((Reuters|AP|AFP|UPI)\)\s*-\s*"
    return re.sub(reuters_pattern, "", text, flags=re.IGNORECASE)

def create_features_lightweight():
    print("Running lightweight feature extraction...")
    
    # 1. Load Data using simple CSV reader to save memory if pandas crashes
    # But pandas is usually okay if we read only needed columns
    print("Loading data...")
    try:
        train_df = pd.read_csv("data/processed/train.csv", usecols=['text', 'label'])
        test_df = pd.read_csv("data/processed/test.csv", usecols=['text', 'label'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    # 2. Clean text (redundant if files are clean, but safe)
    print("Cleaning text...")
    train_texts = train_df['text'].astype(str).apply(clean_source_text_simple)
    test_texts = test_df['text'].astype(str).apply(clean_source_text_simple)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # 3. TF-IDF
    print("Vectorizing (TF-IDF)...")
    # Limit features to 5000 to save memory/compute
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"Train shape: {X_train.shape}")
    
    # 4. Save
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Saving files...")
    np.savez(os.path.join(processed_dir, "features_train_tfidf.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(processed_dir, "features_test_tfidf.npz"), X=X_test, y=y_test)
    joblib.dump(vectorizer, os.path.join(processed_dir, "tfidf_vectorizer.pkl"))
    
    print("Done!")

if __name__ == "__main__":
    create_features_lightweight()


