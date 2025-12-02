"""
Train/test split functionality.

This module creates stratified train/test splits from cleaned data.
Note: Source cleaning and deduplication should already be done in process_dataset.py
"""

import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_split(
    input_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_seed: int = 42,
    label_column: str = "label",
    feature_column: str = "text"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test sets.
    
    Args:
        input_path: Path to the cleaned dataset CSV
        output_dir: Directory to save train.csv and test.csv
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        label_column: Column name to use as labels (y) and stratify on
        feature_column: Column name to use as features (X)
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If required columns not in dataset
    """
    # Validate input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    print("=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df):,} rows")
    
    # Validate required columns
    print(f"\n2. Validating columns...")
    print(f"   Feature column (X): '{feature_column}'")
    print(f"   Label column (y):   '{label_column}'")
    
    missing_cols = []
    if feature_column not in df.columns:
        missing_cols.append(feature_column)
    if label_column not in df.columns:
        missing_cols.append(label_column)
    
    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    print("   ✓ All required columns present")
    
    # Show label distribution before split
    print(f"\n3. Label distribution (before split):")
    label_counts = df[label_column].value_counts().sort_index()
    for label_val, count in label_counts.items():
        label_name = "Real" if label_val == 0 else "Fake"
        pct = count / len(df) * 100
        print(f"   {label_val} ({label_name}): {count:,} ({pct:.1f}%)")
    
    # Split with stratification on label column
    print(f"\n4. Splitting data (test_size={test_size}, random_seed={random_seed})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[label_column]
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    print(f"\n5. Saving splits...")
    train_df.to_csv(train_path, index=False)
    print(f"   Train: {train_path} ({len(train_df):,} rows)")
    
    test_df.to_csv(test_path, index=False)
    print(f"   Test:  {test_path} ({len(test_df):,} rows)")
    
    print("\n" + "=" * 60)
    print("✓ SPLIT COMPLETE")
    print("=" * 60)
    print(f"  Feature column: '{feature_column}'")
    print(f"  Label column:   '{label_column}'")
    print(f"  Train samples:  {len(train_df):,}")
    print(f"  Test samples:   {len(test_df):,}")
    
    return train_df, test_df


def make_split_from_config(config_path: str = "config/config.yaml"):
    """Wrapper that reads parameters from config file."""
    config = load_config(config_path)
    
    processed_dir = config["data"]["processed_data_dir"]
    cleaned_filename = config["data"]["cleaned_dataset"]
    input_path = os.path.join(processed_dir, cleaned_filename)
    
    # Get column names from config
    columns_config = config.get("columns", {})
    feature_column = columns_config.get("feature_column", "text")
    label_column = columns_config.get("label_column", "label")
    
    return make_split(
        input_path=input_path,
        output_dir=processed_dir,
        test_size=config["models"]["test_size"],
        random_seed=config["models"]["random_seed"],
        label_column=label_column,
        feature_column=feature_column
    )


if __name__ == "__main__":
    make_split_from_config()
