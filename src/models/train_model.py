"""
Model training module.

This module trains and evaluates ML models on pre-computed features.
It provides verbose logging of which features and labels are used.
"""

import argparse
import yaml
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_dir, verbose=True):
    """
    Load training and test features/labels from saved .npz files.
    
    Args:
        processed_dir: Directory containing feature files
        verbose: Whether to print detailed info about loaded data
        
    Returns:
        X_train: TF-IDF feature matrix (sparse) for training
        y_train: Labels for training (0=Real, 1=Fake)
        X_test: TF-IDF feature matrix (sparse) for testing
        y_test: Labels for testing (0=Real, 1=Fake)
        metadata: Dict with feature_column and label_column names
    """
    train_path = os.path.join(processed_dir, "features_train_tfidf.npz")
    test_path = os.path.join(processed_dir, "features_test_tfidf.npz")
    
    if verbose:
        print(f"   Loading: {train_path}")
    train_data = np.load(train_path, allow_pickle=True)
    
    if verbose:
        print(f"   Loading: {test_path}")
    test_data = np.load(test_path, allow_pickle=True)
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    # Extract metadata if available
    metadata = {}
    if "feature_column" in train_data:
        metadata["feature_column"] = str(train_data["feature_column"])
    else:
        metadata["feature_column"] = "unknown"
    if "label_column" in train_data:
        metadata["label_column"] = str(train_data["label_column"])
    else:
        metadata["label_column"] = "unknown"
    
    # Handle sparse matrix wrapped in 0-d array (numpy savez quirk)
    if X_train.ndim == 0:
        X_train = X_train.item()
    if X_test.ndim == 0:
        X_test = X_test.item()
        
    return X_train, y_train, X_test, y_test, metadata


def train_model(model_name, config, return_metrics=False, processed_dir=None, models_dir=None):
    """
    Train and evaluate a model.
    
    Args:
        model_name: Name of the model to train (e.g., "ridge_classifier", "logistic_regression")
        config: Configuration dictionary
        return_metrics: Whether to return metrics dictionary and predictions
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
    
    print("=" * 60)
    print(f"MODEL TRAINING: {model_name}")
    print("=" * 60)
    
    # Load data with verbose output
    print("\n1. Loading features and labels...")
    X_train, y_train, X_test, y_test, metadata = load_data(processed_dir, verbose=True)
    
    # Print detailed info about features and labels
    print(f"\n2. Data summary:")
    print(f"   Feature source: '{metadata['feature_column']}' column → TF-IDF vectors")
    print(f"   Label source:   '{metadata['label_column']}' column (0=Real, 1=Fake)")
    print(f"\n   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    
    # Label distribution
    print(f"\n   Label distribution:")
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist_str = ", ".join([f"{v}={c}" for v, c in zip(unique, counts)])
        print(f"     {split_name}: {dist_str}")
    
    # Get model parameters from config
    print(f"\n3. Initializing {model_name}...")
    model_params = config["models"].get(model_name, {})
    
    if model_name == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_name == "ridge_classifier":
        model = RidgeClassifier(**model_params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**model_params)
    elif model_name == "knn":
        model = KNeighborsClassifier(**model_params)
    elif model_name == "lgbm":
        model = lgb.LGBMClassifier(**model_params)
    else:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Supported: logistic_regression, ridge_classifier, random_forest, knn, lgbm")
    
    if model_params:
        print(f"   Parameters: {model_params}")
    else:
        print("   Using default parameters")
    
    # Train
    print(f"\n4. Training model...")
    model.fit(X_train, y_train)
    print("   ✓ Training complete")
    
    # Evaluate
    print(f"\n5. Evaluating on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n   Accuracy: {acc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fake (1)"]))
    
    # Save model
    save_path = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    print(f"6. Model saved to: {save_path}")
    
    print("\n" + "=" * 60)
    print(f"✓ TRAINING COMPLETE: {model_name}")
    print("=" * 60)
    print(f"  Features: '{metadata['feature_column']}' → TF-IDF ({X_train.shape[1]} features)")
    print(f"  Labels:   '{metadata['label_column']}' (0=Real, 1=Fake)")
    print(f"  Accuracy: {acc:.4f}")

    if return_metrics:
        metrics = {
            "accuracy": acc,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "feature_column": metadata["feature_column"],
            "label_column": metadata["label_column"]
        }
        return model, metrics, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on config")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (e.g., logistic_regression, ridge_classifier, lgbm)")
    args = parser.parse_args()
    
    config = load_config()
    train_model(args.model, config)
