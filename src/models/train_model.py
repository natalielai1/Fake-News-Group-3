import argparse
import yaml
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(processed_dir):
    """Load training features and labels."""
    # For simplicity, we'll use TF-IDF features for standard ML models
    print("Loading TF-IDF features...")
    train_data = np.load(os.path.join(processed_dir, "features_train_tfidf.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(processed_dir, "features_test_tfidf.npz"), allow_pickle=True)
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    # Handle sparse matrix wrapped in 0-d array
    if X_train.ndim == 0:
        X_train = X_train.item()
    if X_test.ndim == 0:
        X_test = X_test.item()
        
    return X_train, y_train, X_test, y_test

def train_model(model_name, config, return_metrics=False, processed_dir=None, models_dir=None):
    """
    Train and evaluate a model.
    
    Args:
        model_name: Name of the model to train (key in config['models'] or specific supported name)
        config: Configuration dictionary
        return_metrics: Whether to return metrics dictionary and predictions (default: False)
        processed_dir: Override for processed data directory (uses config if None)
        models_dir: Override for models output directory (uses config if None)
        
    Returns:
        If return_metrics is True: (model, metrics_dict, y_pred)
        Else: None
    """
    if processed_dir is None:
        processed_dir = config["data"]["processed_data_dir"]
    if models_dir is None:
        models_dir = config["results"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data(processed_dir)
    
    print(f"Training {model_name}...")
    model_params = config["models"].get(model_name, {})
    
    if model_name == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_name == "ridge_classifier":
        # RidgeClassifier doesn't take all same params as LogReg, so filter or use defaults if needed
        # Assuming params in config are compatible or empty
        model = RidgeClassifier(**model_params)
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

    if return_metrics:
        metrics = {
            "accuracy": acc,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        return model, metrics, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on config")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., logistic_regression, ridge_classifier, lgbm)")
    args = parser.parse_args()
    
    config = load_config()
    train_model(args.model, config)
