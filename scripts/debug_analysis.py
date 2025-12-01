import joblib
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

def analyze_simple():
    print("Starting simple analysis...")
    
    # Load data with try/except
    try:
        print("Loading NPZ data...")
        train_data = np.load("data/processed/features_train_tfidf.npz")
        # .item() is needed because sparse matrices are saved as object arrays in npz by default if not careful
        # But let's check keys first
        print(f"Keys: {list(train_data.keys())}")
        
        # Handle scipy sparse matrix loading from npz
        # Often saved as data, indices, indptr, shape
        # Or just 'X' as an object if pickle allowed
        
        # Let's try to just load the vectorizer first to see if that's the crash point
    except Exception as e:
        print(f"Data load error: {e}")
        return

    try:
        print("Loading vectorizer...")
        vectorizer = joblib.load("data/processed/tfidf_vectorizer.pkl")
        print("Vectorizer loaded.")
    except Exception as e:
        print(f"Vectorizer load error: {e}")
        return

if __name__ == "__main__":
    analyze_simple()

