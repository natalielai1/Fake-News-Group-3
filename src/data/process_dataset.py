"""
Dataset processing logic.
"""

from pathlib import Path
import sys
import pandas as pd

# Ensure project root is in path if needed for imports when run as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.data.load_data import load_kaggle_dataset, save_processed_data
from src.preprocessing import preprocess_dataframe, tokenize_dataframe

def process_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Process the raw dataset: load, tokenize, preprocess, and save.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed CSV file
        
    Returns:
        Processed DataFrame
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
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
    if len(df) > 0:
        print(f"   Original text length: {len(str(df.iloc[sample_idx]['text']))} chars")
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
    
    return df

if __name__ == "__main__":
    # Default paths when run as script
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "raw" / "WELFake_Dataset.csv"
    output_path = project_root / "data" / "processed" / "preprocessed_dataset.csv"
    
    process_dataset(input_path, output_path)

