"""
Feature engineering module.

This module converts text data into numerical features for ML models:
- TF-IDF vectors (sparse, high-dimensional)
- Word2Vec / Sentence2Vec (dense, lower-dimensional)

The feature and label columns are configurable to make it clear
which data is used for training.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import joblib
from gensim.models import Word2Vec
import nltk

# Ensure nltk data is available (simple tokenizer)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_split_data(processed_dir="data/processed"):
    """
    Load the train and test datasets.
    
    Args:
        processed_dir: Directory containing train.csv and test.csv
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/test split files not found. Please run src/data/make_split.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def create_tfidf_features(texts, max_features=5000):
    """
    Convert text into TF-IDF vectors.
    
    Args:
        texts: List/Series of text strings
        max_features: Maximum vocabulary size
    
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


def train_word2vec(texts, vector_size=100, window=5, min_count=1):
    """
    Train a Word2Vec model on the provided texts.
    
    Args:
        texts: List of text strings
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word frequency
        
    Returns:
        Trained Word2Vec model
    """
    tokenized_texts = [str(text).split() for text in texts]
    
    model = Word2Vec(
        sentences=tokenized_texts, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=4
    )
    return model


def get_sentence_vector(text, model):
    """
    Calculate the average word vector for a given text.
    
    Args:
        text: Input text string
        model: Trained Word2Vec model
        
    Returns:
        Averaged word vector (numpy array)
    """
    words = str(text).split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)


def create_word2vec_features(texts, model=None, vector_size=100):
    """
    Convert texts into Sentence2Vec features (averaged Word2Vec).
    
    Args:
        texts: List of text strings
        model: Pre-trained Word2Vec model (trains new one if None)
        vector_size: Dimensionality if training new model
        
    Returns:
        X: numpy array of sentence vectors
        model: trained Word2Vec model
    """
    if model is None:
        print("   Training Word2Vec model...")
        model = train_word2vec(texts, vector_size=vector_size)
    
    print("   Generating sentence vectors...")
    X = np.array([get_sentence_vector(text, model) for text in texts])
    return X, model


def run_feature_engineering(
    processed_dir: str = "data/processed",
    feature_column: str = "text",
    label_column: str = "label",
    max_features: int = 5000
):
    """
    Complete feature engineering pipeline.
    
    Converts text data into numerical features for ML models:
    1. Load train/test datasets
    2. Fit TF-IDF on train text only (prevents data leakage)
    3. Transform train and test text
    4. Train Word2Vec on train text only
    5. Generate Sentence2Vec features for train and test
    6. Save features + vectorizer/model
    
    Args:
        processed_dir: Directory containing train.csv/test.csv, and output location
        feature_column: Column name containing text to vectorize (X)
        label_column: Column name containing labels (y)
        max_features: Maximum TF-IDF vocabulary size
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading train/test split...")
    try:
        train_df, test_df = load_split_data(processed_dir)
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        return
    
    # Step 2: Extract feature and label columns
    print(f"\n2. Extracting features and labels...")
    print(f"   Feature column (X): '{feature_column}'")
    print(f"   Label column (y):   '{label_column}'")
    
    # Validate columns exist
    for col in [feature_column, label_column]:
        if col not in train_df.columns:
            raise ValueError(f"Column '{col}' not found in train data. Available: {list(train_df.columns)}")
        if col not in test_df.columns:
            raise ValueError(f"Column '{col}' not found in test data. Available: {list(test_df.columns)}")
    
    X_train_text = train_df[feature_column].astype(str)
    y_train = train_df[label_column]
    
    X_test_text = test_df[feature_column].astype(str)
    y_test = test_df[label_column]
    
    print(f"\n   Train samples: {len(X_train_text):,}")
    print(f"   Test samples:  {len(X_test_text):,}")
    
    # Show label distribution
    print(f"\n   Label distribution (train):")
    for label_val in sorted(y_train.unique()):
        count = (y_train == label_val).sum()
        pct = count / len(y_train) * 100
        label_name = "Real" if label_val == 0 else "Fake"
        print(f"     {label_val} ({label_name}): {count:,} ({pct:.1f}%)")

    # Step 3: TF-IDF Features
    print(f"\n3. Creating TF-IDF features...")
    print(f"   Max features: {max_features:,}")
    print(f"   N-gram range: (1, 2)")
    
    X_train_tfidf, tfidf_vectorizer = create_tfidf_features(X_train_text, max_features=max_features)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    
    print(f"   ✓ Train TF-IDF shape: {X_train_tfidf.shape}")
    print(f"   ✓ Test TF-IDF shape:  {X_test_tfidf.shape}")

    # Step 4: Word2Vec / Sentence2Vec Features
    print(f"\n4. Creating Word2Vec/Sentence2Vec features...")
    X_train_w2v, w2v_model = create_word2vec_features(X_train_text)
    X_test_w2v = np.array([get_sentence_vector(text, w2v_model) for text in X_test_text])
    
    print(f"   ✓ Train W2V shape: {X_train_w2v.shape}")
    print(f"   ✓ Test W2V shape:  {X_test_w2v.shape}")

    # Step 5: Save features
    print(f"\n5. Saving features to {processed_dir}...")
    os.makedirs(processed_dir, exist_ok=True)

    # Save TF-IDF
    np.savez(os.path.join(processed_dir, "features_train_tfidf.npz"), 
             X=X_train_tfidf, y=y_train,
             feature_column=feature_column, label_column=label_column)
    np.savez(os.path.join(processed_dir, "features_test_tfidf.npz"), 
             X=X_test_tfidf, y=y_test,
             feature_column=feature_column, label_column=label_column)
    joblib.dump(tfidf_vectorizer, os.path.join(processed_dir, "tfidf_vectorizer.pkl"))
    print("   ✓ TF-IDF features saved")

    # Save Word2Vec
    np.savez(os.path.join(processed_dir, "features_train_w2v.npz"), 
             X=X_train_w2v, y=y_train,
             feature_column=feature_column, label_column=label_column)
    np.savez(os.path.join(processed_dir, "features_test_w2v.npz"), 
             X=X_test_w2v, y=y_test,
             feature_column=feature_column, label_column=label_column)
    w2v_model.save(os.path.join(processed_dir, "word2vec.model"))
    print("   ✓ Word2Vec features saved")

    print("\n" + "=" * 60)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  Feature column: '{feature_column}'")
    print(f"  Label column:   '{label_column}'")
    print(f"  TF-IDF shape:   {X_train_tfidf.shape}")
    print(f"  Word2Vec shape: {X_train_w2v.shape}")


# Run pipeline when executed as a script
if __name__ == "__main__":
    run_feature_engineering()
