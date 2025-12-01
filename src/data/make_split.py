import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
import argparse
from src.preprocessing import clean_source_text

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def make_split(
    input_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify_column: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test sets.
    
    Args:
        input_path: Path to the preprocessed dataset CSV
        output_dir: Directory to save train.csv and test.csv
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        stratify_column: Column name to stratify on
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If stratify_column not in dataset
    """
    # Validate input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Apply leakage cleaning to the 'text' column permanently
    print("Applying source leakage cleaning to 'text' column...")
    df['text'] = df['text'].apply(clean_source_text)
    
    # Deduplicate
    print("Checking for duplicates...")
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"Removed {initial_len - len(df)} duplicate rows. New size: {len(df)}")
    
    # Validate stratify column
    if stratify_column not in df.columns:
        raise ValueError(
            f"Column '{stratify_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Split
    print(f"Splitting data (test_size={test_size}, random_seed={random_seed})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[stratify_column]
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    print(f"Saving train set to {train_path} ({len(train_df)} rows)...")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving test set to {test_path} ({len(test_df)} rows)...")
    test_df.to_csv(test_path, index=False)
    
    print("Split complete!")
    return train_df, test_df


def make_split_from_config(config_path: str = "config/config.yaml"):
    """Wrapper that reads parameters from config file."""
    config = load_config(config_path)
    
    processed_dir = config["data"]["processed_data_dir"]
    cleaned_filename = config["data"]["cleaned_dataset"]
    input_path = os.path.join(processed_dir, cleaned_filename)
    
    return make_split(
        input_path=input_path,
        output_dir=processed_dir,
        test_size=config["models"]["test_size"],
        random_seed=config["models"]["random_seed"]
    )

if __name__ == "__main__":
    make_split_from_config()
