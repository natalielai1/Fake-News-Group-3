import argparse
import yaml
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(processed_dir):
    """Load training features and labels."""
    # For simplicity, we'll use TF-IDF features for standard ML models
    # In a real scenario, you might choose between TF-IDF and Word2Vec based on the model
    print("Loading TF-IDF features...")
    train_data = np.load(os.path.join(processed_dir, "features_train_tfidf.npz"))
    test_data = np.load(os.path.join(processed_dir, "features_test_tfidf.npz"))
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    # Convert sparse matrix if necessary (some models might need dense)
    # X_train = X_train.item() # If saved as object array
    # But np.savez with sparse matrix usually requires careful handling.
    # Let's assume for now it loads correctly as a sparse matrix or we might need to use scipy.sparse.load_npz if we saved it that way.
    # Wait, make_features.py used np.savez. 
    # If X is sparse, np.savez saves it as a 0-d array containing the matrix.
    # We need to extract it properly.
    
    # Actually, let's check how it was saved. 
    # make_features.py: np.savez(..., X=X_train_tfidf, y=y_train)
    # If X_train_tfidf is sparse, this might be tricky.
    # Standard practice for sparse matrices is scipy.sparse.save_npz.
    # However, since we are just reading what was written, let's try to handle what's there.
    # If it's a 0-d array wrapping the sparse matrix:
    if X_train.ndim == 0:
        X_train = X_train.item()
    if X_test.ndim == 0:
        X_test = X_test.item()
        
    return X_train, y_train, X_test, y_test

def train_model(model_name, config):
    processed_dir = config["data"]["processed_data_dir"]
    models_dir = config["results"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data(processed_dir)
    
    print(f"Training {model_name}...")
    model_params = config["models"].get(model_name, {})
    
    if model_name == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**model_params)
    elif model_name == "knn":
        model = KNeighborsClassifier(**model_params)
    elif model_name == "lgbm":
        model = lgb.LGBMClassifier(**model_params)
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    save_path = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on config")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., logistic_regression, lgbm)")
    args = parser.parse_args()
    
    config = load_config()
    train_model(args.model, config)
