import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
import argparse

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def make_split():
    config = load_config()
    
    # Paths
    processed_dir = config["data"]["processed_data_dir"]
    cleaned_filename = config["data"]["cleaned_dataset"]
    input_path = os.path.join(processed_dir, cleaned_filename)
    
    # Load data
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}. Please run preprocessing first.")
    
    df = pd.read_csv(input_path)
    
    # Split parameters
    test_size = config["models"]["test_size"]
    random_seed = config["models"]["random_seed"]
    
    # Check if label column exists (assuming 'label' based on previous file analysis)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for stratification.")
    
    print(f"Splitting data (test_size={test_size}, random_seed={random_seed})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"]
    )
    
    # Save
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    print(f"Saving train set to {train_path} ({len(train_df)} rows)...")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving test set to {test_path} ({len(test_df)} rows)...")
    test_df.to_csv(test_path, index=False)
    
    print("Split complete!")

if __name__ == "__main__":
    make_split()
