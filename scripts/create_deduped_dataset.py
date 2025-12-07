"""
Create de-duplicated dataset by removing near-duplicate articles.

This script:
1. Loads the cleaned dataset and near_duplicate_pairs.csv
2. Filters pairs with Jaccard similarity >= threshold (default 0.7)
3. Builds connected components (groups) from duplicate pairs
4. Keeps one article from each group (lowest index)
5. Applies enhanced source marker cleaning
6. Saves de-duplicated dataset
7. Creates train/test split and TF-IDF features
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class UnionFind:
    """Union-Find data structure for grouping near-duplicates."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_groups(self):
        """Return dictionary mapping root -> list of members."""
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)


def enhanced_clean_source_text(text: str) -> str:
    """
    Enhanced source marker cleaning that catches patterns missed by original cleaner.
    
    Additional patterns handled:
    - "Reuters \n" or "Reuters\n" at start
    - "REUTERS\n" (all caps)
    - Date + source patterns like "11/02/2016 \nREUTERS"
    - Standalone "Reuters" at the very start of text
    - "(Source)" patterns without the dash
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Wire services to clean
    wire_services = (
        r"Reuters|AP|AFP|UPI|BBC|CNN|Al Jazeera|Xinhua|IANS|PTI|ANI|dpa|"
        r"Associated Press|Agence France-Presse|United Press International"
    )
    
    # Pattern 1: Standalone wire service name at start (with optional whitespace/newlines)
    # e.g., "Reuters \n" or "REUTERS\n" at the very beginning
    text = re.sub(rf"^({wire_services})\s*\n+", "", text, flags=re.IGNORECASE)
    
    # Pattern 2: Date + wire service patterns
    # e.g., "11/02/2016 \nREUTERS" or "Nov 2, 2016 REUTERS"
    date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    text = re.sub(rf"^{date_pattern}\s*\n*\s*({wire_services})\s*\n*", "", text, flags=re.IGNORECASE)
    
    # Pattern 3: Word date + wire service
    # e.g., "November 2, 2016 Reuters"
    word_date_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}"
    text = re.sub(rf"^{word_date_pattern}\s*\n*\s*({wire_services})\s*\n*", "", text, flags=re.IGNORECASE)
    
    # Pattern 4: Location (Source) - at start (more aggressive)
    # Uses [^(]* to match ANY characters before the first parenthesis
    text = re.sub(rf"^[^(]*\(({wire_services})\)\s*[-–—]?\s*", "", text, flags=re.IGNORECASE)
    
    # Pattern 5: "By Reuters" or "By AP" etc. at start
    text = re.sub(rf"^By\s+({wire_services})\s*[-–—]?\s*", "", text, flags=re.IGNORECASE)
    
    # Pattern 6: "- Reuters" or "— Reuters" at end of paragraphs/text
    text = re.sub(rf"\s*[-–—]\s*({wire_services})\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 7: Standalone "(Reuters)" without the dash
    text = re.sub(rf"^\s*\(({wire_services})\)\s*", "", text, flags=re.IGNORECASE)
    
    # Pattern 8: Wire service followed by colon at start
    # e.g., "Reuters: " or "AP: "
    text = re.sub(rf"^({wire_services})\s*:\s*", "", text, flags=re.IGNORECASE)
    
    # Pattern 9: U.S./BREAKING style headers with wire service
    # e.g., "U.S. militia girds... REUTERS"
    text = re.sub(rf"\b({wire_services})\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up multiple newlines that might result from cleaning
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def create_deduped_dataset(
    cleaned_path: str,
    pairs_path: str,
    output_path: str,
    similarity_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Create de-duplicated dataset by removing near-duplicate articles.
    
    Args:
        cleaned_path: Path to cleaned_dataset.csv
        pairs_path: Path to near_duplicate_pairs.csv
        output_path: Path to save de-duplicated dataset
        similarity_threshold: Minimum Jaccard similarity to consider as duplicate
        
    Returns:
        De-duplicated DataFrame
    """
    print("=" * 60)
    print("CREATING DE-DUPLICATED DATASET")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading datasets...")
    df = pd.read_csv(cleaned_path)
    pairs_df = pd.read_csv(pairs_path)
    print(f"   Cleaned dataset: {len(df):,} articles")
    print(f"   Near-duplicate pairs: {len(pairs_df):,} pairs")
    
    # Filter pairs by similarity threshold
    print(f"\n2. Filtering pairs with Jaccard >= {similarity_threshold}...")
    filtered_pairs = pairs_df[pairs_df['jaccard_similarity'] >= similarity_threshold]
    print(f"   Pairs above threshold: {len(filtered_pairs):,}")
    
    # Build connected components using Union-Find
    print(f"\n3. Building duplicate groups (connected components)...")
    uf = UnionFind()
    
    for _, row in filtered_pairs.iterrows():
        idx1, idx2 = int(row['idx1']), int(row['idx2'])
        uf.union(idx1, idx2)
    
    groups = uf.get_groups()
    print(f"   Found {len(groups):,} duplicate groups")
    
    # Count articles involved in duplicates
    all_dup_indices = set()
    for members in groups.values():
        all_dup_indices.update(members)
    print(f"   Total articles in duplicate groups: {len(all_dup_indices):,}")
    
    # For each group, keep only the article with the lowest index
    indices_to_remove = set()
    for root, members in groups.items():
        # Sort members and keep the first (lowest index)
        sorted_members = sorted(members)
        # Remove all except the first
        indices_to_remove.update(sorted_members[1:])
    
    print(f"   Articles to remove: {len(indices_to_remove):,}")
    
    # Remove duplicates
    print(f"\n4. Removing duplicate articles...")
    # Filter by index (iloc-based) - need to map to actual DataFrame indices
    df_deduped = df[~df.index.isin(indices_to_remove)].copy()
    print(f"   Articles remaining: {len(df_deduped):,}")
    
    # Apply enhanced source cleaning
    print(f"\n5. Applying enhanced source marker cleaning...")
    df_deduped['text'] = df_deduped['text'].apply(enhanced_clean_source_text)
    print(f"   ✓ Source markers cleaned")
    
    # Reset index
    df_deduped = df_deduped.reset_index(drop=True)
    
    # Show label distribution
    print(f"\n6. Label distribution after de-duplication:")
    label_counts = df_deduped['label'].value_counts().sort_index()
    for label_val, count in label_counts.items():
        label_name = "Real" if label_val == 0 else "Fake"
        pct = count / len(df_deduped) * 100
        print(f"   {label_val} ({label_name}): {count:,} ({pct:.1f}%)")
    
    # Save
    print(f"\n7. Saving de-duplicated dataset...")
    df_deduped.to_csv(output_path, index=False)
    print(f"   ✓ Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✓ DE-DUPLICATION COMPLETE")
    print("=" * 60)
    print(f"  Original articles:  {len(df):,}")
    print(f"  Removed duplicates: {len(indices_to_remove):,}")
    print(f"  Final articles:     {len(df_deduped):,}")
    
    return df_deduped


def create_train_test_split(
    input_path: str,
    output_dir: str,
    train_filename: str = "train_deduped.csv",
    test_filename: str = "test_deduped.csv",
    test_size: float = 0.2,
    random_seed: int = 42
) -> tuple:
    """
    Create train/test split for de-duplicated dataset.
    
    Args:
        input_path: Path to de-duplicated dataset
        output_dir: Directory to save train/test files
        train_filename: Name of training file
        test_filename: Name of test file
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print("\n" + "=" * 60)
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading de-duplicated dataset...")
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df):,} articles")
    
    # Show label distribution
    print(f"\n2. Label distribution (before split):")
    label_counts = df['label'].value_counts().sort_index()
    for label_val, count in label_counts.items():
        label_name = "Real" if label_val == 0 else "Fake"
        pct = count / len(df) * 100
        print(f"   {label_val} ({label_name}): {count:,} ({pct:.1f}%)")
    
    # Split
    print(f"\n3. Splitting data (test_size={test_size}, seed={random_seed})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df['label']
    )
    
    print(f"   Train samples: {len(train_df):,}")
    print(f"   Test samples:  {len(test_df):,}")
    
    # Save
    print(f"\n4. Saving splits...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / train_filename
    test_path = output_dir / test_filename
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   ✓ Train: {train_path}")
    print(f"   ✓ Test:  {test_path}")
    
    return train_df, test_df


def create_tfidf_features(
    processed_dir: str,
    train_filename: str = "train_deduped.csv",
    test_filename: str = "test_deduped.csv",
    output_prefix: str = "_deduped",
    max_features: int = 5000
):
    """
    Create TF-IDF features for de-duplicated dataset.
    
    Args:
        processed_dir: Directory containing train/test files
        train_filename: Name of training file
        test_filename: Name of test file
        output_prefix: Prefix for output feature files
        max_features: Maximum TF-IDF vocabulary size
    """
    print("\n" + "=" * 60)
    print("CREATING TF-IDF FEATURES")
    print("=" * 60)
    
    processed_dir = Path(processed_dir)
    
    # Load data
    print(f"\n1. Loading train/test data...")
    train_df = pd.read_csv(processed_dir / train_filename)
    test_df = pd.read_csv(processed_dir / test_filename)
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")
    
    # Extract text and labels
    X_train_text = train_df['text'].astype(str)
    y_train = train_df['label'].values
    X_test_text = test_df['text'].astype(str)
    y_test = test_df['label'].values
    
    # Create TF-IDF features
    print(f"\n2. Creating TF-IDF features...")
    print(f"   Max features: {max_features:,}")
    print(f"   N-gram range: (1, 2)")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    print(f"   ✓ Train TF-IDF shape: {X_train_tfidf.shape}")
    print(f"   ✓ Test TF-IDF shape:  {X_test_tfidf.shape}")
    
    # Save features
    print(f"\n3. Saving features...")
    
    train_features_path = processed_dir / f"features_train_tfidf{output_prefix}.npz"
    test_features_path = processed_dir / f"features_test_tfidf{output_prefix}.npz"
    vectorizer_path = processed_dir / f"tfidf_vectorizer{output_prefix}.pkl"
    
    np.savez(train_features_path, X=X_train_tfidf, y=y_train,
             feature_column='text', label_column='label')
    np.savez(test_features_path, X=X_test_tfidf, y=y_test,
             feature_column='text', label_column='label')
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"   ✓ {train_features_path}")
    print(f"   ✓ {test_features_path}")
    print(f"   ✓ {vectorizer_path}")
    
    print("\n" + "=" * 60)
    print("✓ TF-IDF FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


def main():
    """Main execution function."""
    # Paths
    processed_dir = project_root / "data" / "processed"
    cleaned_path = processed_dir / "cleaned_dataset.csv"
    pairs_path = processed_dir / "near_duplicate_pairs.csv"
    deduped_path = processed_dir / "cleaned_dataset_deduped.csv"
    
    # Configuration
    SIMILARITY_THRESHOLD = 0.7
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    MAX_FEATURES = 5000
    
    print("\n" + "=" * 60)
    print("DE-DUPLICATION PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Max TF-IDF features: {MAX_FEATURES}")
    
    # Step 1: Create de-duplicated dataset
    df_deduped = create_deduped_dataset(
        cleaned_path=str(cleaned_path),
        pairs_path=str(pairs_path),
        output_path=str(deduped_path),
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    # Step 2: Create train/test split
    train_df, test_df = create_train_test_split(
        input_path=str(deduped_path),
        output_dir=str(processed_dir),
        train_filename="train_deduped.csv",
        test_filename="test_deduped.csv",
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # Step 3: Create TF-IDF features
    create_tfidf_features(
        processed_dir=str(processed_dir),
        train_filename="train_deduped.csv",
        test_filename="test_deduped.csv",
        output_prefix="_deduped",
        max_features=MAX_FEATURES
    )
    
    print("\n" + "=" * 60)
    print("✓ ALL STEPS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {deduped_path}")
    print(f"  - {processed_dir / 'train_deduped.csv'}")
    print(f"  - {processed_dir / 'test_deduped.csv'}")
    print(f"  - {processed_dir / 'features_train_tfidf_deduped.npz'}")
    print(f"  - {processed_dir / 'features_test_tfidf_deduped.npz'}")
    print(f"  - {processed_dir / 'tfidf_vectorizer_deduped.pkl'}")


if __name__ == "__main__":
    main()


