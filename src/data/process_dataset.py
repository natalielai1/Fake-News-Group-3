"""
Dataset processing logic.

This module handles the initial cleaning of raw data:
1. Load raw CSV
2. Clean source attribution (e.g., "(Reuters) -") to prevent data leakage
3. Remove duplicate articles
4. Save cleaned data

Note: This step does NOT create train/test splits or extract features.
"""

from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in path if needed for imports when run as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.data.load_data import load_kaggle_dataset, save_processed_data
from src.preprocessing import clean_source_text


def process_dataset(
    input_path: str,
    output_path: str,
    feature_column: str = "text",
    label_column: str = "label",
    id_column: str = "id"
) -> pd.DataFrame:
    """
    Process the raw dataset: load, clean source attribution, deduplicate, and save.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save cleaned CSV file
        feature_column: Name of the text column to clean (default: "text")
        label_column: Name of the label column (default: "label")
        id_column: Name of the ID column (default: "id")
        
    Returns:
        Cleaned DataFrame
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print("=" * 60)
    print("DATA PROCESSING")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Step 1: Load data
    print("\n1. Loading dataset...")
    df = load_kaggle_dataset(input_path)
    initial_count = len(df)
    print(f"   Loaded {initial_count:,} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Step 2: Clean source attribution from feature column
    print(f"\n2. Cleaning source attribution from '{feature_column}' column...")
    print("   Removing patterns like '(Reuters) -', '(AP) -', etc.")
    df[feature_column] = df[feature_column].apply(clean_source_text)
    print("   ✓ Source attribution removed")
    
    # Step 3: Remove duplicates based on feature column
    print(f"\n3. Removing duplicate articles (based on '{feature_column}')...")
    df = df.drop_duplicates(subset=[feature_column])
    removed_count = initial_count - len(df)
    print(f"   Removed {removed_count:,} duplicates")
    print(f"   Remaining: {len(df):,} unique articles")
    
    # Step 4: Show label distribution
    print(f"\n4. Label distribution ('{label_column}' column):")
    label_counts = df[label_column].value_counts().sort_index()
    for label_val, count in label_counts.items():
        label_name = "Real" if label_val == 0 else "Fake"
        pct = count / len(df) * 100
        print(f"   {label_val} ({label_name}): {count:,} ({pct:.1f}%)")
    
    # Step 5: Save cleaned data
    print("\n5. Saving cleaned data...")
    # Only keep essential columns
    columns_to_save = [id_column, "title", feature_column, label_column]
    columns_to_save = [c for c in columns_to_save if c in df.columns]
    save_processed_data(df, output_path, columns=columns_to_save)
    
    print("\n" + "=" * 60)
    print("✓ DATA PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Feature column: '{feature_column}'")
    print(f"  Label column:   '{label_column}'")
    print(f"  Total rows:     {len(df):,}")
    print(f"  Saved to:       {output_path}")
    
    return df


if __name__ == "__main__":
    # Default paths when run as script
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "raw" / "WELFake_Dataset.csv"
    output_path = project_root / "data" / "processed" / "cleaned_dataset.csv"
    
    process_dataset(input_path, output_path)
