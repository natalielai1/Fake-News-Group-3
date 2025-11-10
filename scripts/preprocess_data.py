#!/usr/bin/env python3
"""
Preprocess the raw Kaggle dataset.

This script loads the raw WELFake dataset, applies comprehensive text preprocessing,
and saves the cleaned version for downstream processing.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_kaggle_dataset, save_processed_data
from src.preprocessing import preprocess_dataframe, tokenize_dataframe


def main():
    """Main preprocessing pipeline."""
    # Define paths
    input_path = project_root / "data" / "raw" / "WELFake_Dataset.csv"
    output_path = project_root / "data" / "processed" / "preprocessed_dataset.csv"
    
    print("Starting data preprocessing...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Load data
    print("\n1. Loading dataset...")
    df = load_kaggle_dataset(input_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Basic tokenization (preserves original tokens)
    print("\n2. Tokenizing text...")
    df = tokenize_dataframe(df, text_column="text", output_column="tokens")
    print(f"   ✓ Basic tokenization complete")
    
    # Advanced preprocessing (clean tokens for modeling)
    print("\n3. Applying preprocessing pipeline...")
    print("   - Lowercasing")
    print("   - Removing punctuation")
    print("   - Removing stopwords")
    print("   - Removing short tokens (< 3 chars)")
    print("   - Lemmatizing")
    
    df = preprocess_dataframe(
        df,
        text_column="text",
        output_column="processed_tokens",
        lowercase=True,
        remove_punct=True,
        remove_stops=True,
        remove_nums=False,
        min_length=3,
        stem=False,
        lemmatize=True
    )
    print(f"   ✓ Preprocessing complete")
    
    # Show sample statistics
    print("\n4. Sample statistics:")
    sample_idx = 0
    print(f"   Original text length: {len(df.iloc[sample_idx]['text'])} chars")
    print(f"   Basic tokens: {len(df.iloc[sample_idx]['tokens'])} tokens")
    print(f"   Processed tokens: {len(df.iloc[sample_idx]['processed_tokens'])} tokens")
    print(f"   Example tokens: {df.iloc[sample_idx]['processed_tokens'][:10]}")
    
    # Save
    print("\n5. Saving processed data...")
    columns = ["id", "title", "text", "tokens", "processed_tokens", "label"]
    save_processed_data(df, output_path, columns=columns)
    
    print("\n✓ Preprocessing complete!")
    print(f"  Saved columns: {columns}")
    print(f"  Total rows: {len(df)}")


if __name__ == "__main__":
    main()


