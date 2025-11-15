import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import joblib


def load_preprocessed_data(path="data/processed/preprocessed_dataset.csv"):
    """
    Load the cleaned dataset created during preprocessing.
    Expected columns: text (or tokens), label.
    """
    return pd.read_csv(path)


def create_tfidf_features(texts, max_features=5000):
    """
    Convert text into TF-IDF vectors.
    - max_features limits vocabulary size.
    - ngram_range=(1,2) includes unigrams + bigrams.
    - stop_words="english" removes common stopwords.
    Returns:
        X: sparse matrix of TF-IDF features
        vectorizer: trained TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def run_feature_engineering():
    """
    Complete feature engineering pipeline:
    1. Load preprocessed dataset
    2. Split text + labels into train/test sets
    3. Fit TF-IDF on train text only
    4. Transform train and test text
    5. Save features + vectorizer to data/processed/
    """
    print("Loading preprocessed dataset...")
    df = load_preprocessed_data()

    # Use whichever column your preprocessing created (tokens or text)
    texts = df["text"].astype(str)  
    labels = df["label"]

    print("Splitting dataset into train/test...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,           # 80% train, 20% test
        random_state=42,         # fixed seed for reproducibility
        stratify=labels          # preserve label distribution
    )

    print("Creating TF-IDF features (fit on training text only)...")
    X_train, vectorizer = create_tfidf_features(X_train_text)

    # Transform test text using the *same* fitted vectorizer
    X_test = vectorizer.transform(X_test_text)

    # Ensure directory exists
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    print("Saving training feature matrix...")
    np.savez(os.path.join(processed_dir, "features_train.npz"), X=X_train, y=y_train)

    print("Saving testing feature matrix...")
    np.savez(os.path.join(processed_dir, "features_test.npz"), X=X_test, y=y_test)

    print("Saving TF-IDF vectorizer (for inference and future models)...")
    joblib.dump(vectorizer, os.path.join(processed_dir, "tfidf_vectorizer.pkl"))

    print("Feature engineering complete!")


# Run pipeline when executed as a script
if __name__ == "__main__":
    run_feature_engineering()