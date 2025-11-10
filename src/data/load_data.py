"""Data loading utilities for fake news detection project."""

import pandas as pd
from pathlib import Path


def load_kaggle_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the Kaggle WELFake dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Rename the index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    
    return df


def save_processed_data(df: pd.DataFrame, filepath: str, columns: list = None) -> None:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        columns: Optional list of columns to save (saves all if None)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if columns:
        df = df[columns]
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df)} rows to {filepath}")

