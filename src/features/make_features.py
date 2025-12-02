import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import joblib
from gensim.models import Word2Vec
import nltk
from src.preprocessing import clean_source_text

# Ensure nltk data is available (simple tokenizer)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_split_data(processed_dir="data/processed"):
    """
    Load the train and test datasets.
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
    - max_features limits vocabulary size.
    - ngram_range=(1,2) includes unigrams + bigrams.
    - stop_words="english" removes common stopwords.
    - Includes source leakage cleaning.
    
    Returns:
        X: sparse matrix of TF-IDF features
        vectorizer: trained TF-IDF vectorizer
    """
    # Apply leakage cleaning
    print("Applying leakage cleaning (removing source attribution)...")
    cleaned_texts = [clean_source_text(text) for text in texts]
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(cleaned_texts)
    return X, vectorizer


def train_word2vec(texts, vector_size=100, window=5, min_count=1):
    """
    Train a Word2Vec model on the provided texts.
    Expects texts to be a list of strings.
    """
    # Simple tokenization: split by space. 
    # Ideally, preprocessing should have handled tokenization.
    # If 'texts' are already space-separated tokens, this works fine.
    tokenized_texts = [text.split() for text in texts]
    
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
    Calculate the average word vector for a given text (sentence).
    """
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)


def create_word2vec_features(texts, model=None, vector_size=100):
    """
    Convert texts into Sentence2Vec features (averaged Word2Vec).
    If model is None, trains a new one.
    Returns:
        X: numpy array of sentence vectors
        model: trained Word2Vec model
    """
    if model is None:
        print("Training Word2Vec model...")
        model = train_word2vec(texts, vector_size=vector_size)
    
    print("Generating Sentence2Vec features...")
    X = np.array([get_sentence_vector(text, model) for text in texts])
    return X, model


def run_feature_engineering(processed_dir="data/processed"):
    """
    Complete feature engineering pipeline:
    1. Load train/test datasets
    2. Fit TF-IDF on train text only
    3. Transform train and test text
    4. Train Word2Vec on train text only
    5. Generate Sentence2Vec features for train and test
    6. Save features + vectorizer/model to data/processed/
    
    Args:
        processed_dir: Directory containing train.csv and test.csv, and where output will be saved
    """
    print("Loading train/test split...")
    try:
        train_df, test_df = load_split_data(processed_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # Assuming 'text' column exists and contains space-separated tokens
    X_train_text = train_df["text"].astype(str)
    y_train = train_df["label"]
    
    X_test_text = test_df["text"].astype(str)
    y_test = test_df["label"]

    # --- TF-IDF ---
    print("Creating TF-IDF features (fit on training text only)...")
    X_train_tfidf, tfidf_vectorizer = create_tfidf_features(X_train_text)
    
    # Clean test text as well to match training distribution
    X_test_text_cleaned = [clean_source_text(text) for text in X_test_text]
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text_cleaned)

    # --- Word2Vec / Sentence2Vec ---
    print("Creating Word2Vec/Sentence2Vec features...")
    
    # Clean text for Word2Vec too
    X_train_text_cleaned = [clean_source_text(text) for text in X_train_text]
    X_train_w2v, w2v_model = create_word2vec_features(X_train_text_cleaned)
    
    # Transform test text using the *same* trained model
    print("Generating Sentence2Vec features for test set...")
    X_test_w2v = np.array([get_sentence_vector(text, w2v_model) for text in X_test_text_cleaned])

    # Ensure directory exists
    os.makedirs(processed_dir, exist_ok=True)

    print("Saving TF-IDF features...")
    np.savez(os.path.join(processed_dir, "features_train_tfidf.npz"), X=X_train_tfidf, y=y_train)
    np.savez(os.path.join(processed_dir, "features_test_tfidf.npz"), X=X_test_tfidf, y=y_test)
    joblib.dump(tfidf_vectorizer, os.path.join(processed_dir, "tfidf_vectorizer.pkl"))

    print("Saving Word2Vec features and model...")
    np.savez(os.path.join(processed_dir, "features_train_w2v.npz"), X=X_train_w2v, y=y_train)
    np.savez(os.path.join(processed_dir, "features_test_w2v.npz"), X=X_test_w2v, y=y_test)
    w2v_model.save(os.path.join(processed_dir, "word2vec.model"))

    print("Feature engineering complete!")


# Run pipeline when executed as a script
if __name__ == "__main__":
    run_feature_engineering()