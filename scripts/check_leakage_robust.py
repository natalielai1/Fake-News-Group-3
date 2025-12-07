import pandas as pd
import sys

def check_leakage():
    print("Loading train and test data...")
    try:
        # Load only necessary columns to save memory
        train_df = pd.read_csv("data/processed/train.csv", usecols=['id', 'text', 'label'])
        test_df = pd.read_csv("data/processed/test.csv", usecols=['id', 'text', 'label'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Checking for exact text duplicates between train and test...")
    train_texts = set(train_df['text'].astype(str))
    test_texts = set(test_df['text'].astype(str))
    
    overlap = train_texts.intersection(test_texts)
    num_overlap = len(overlap)
    
    print(f"Overlapping text count: {num_overlap}")
    
    if num_overlap > 0:
        print("LEAKAGE DETECTED!")
        print(f"Example overlapping texts: {list(overlap)[:3]}")
    else:
        print("No exact text overlap detected.")

    # Check for ID overlap
    train_ids = set(train_df['id'])
    test_ids = set(test_df['id'])
    id_overlap = train_ids.intersection(test_ids)
    print(f"ID Overlap: {len(id_overlap)}")

if __name__ == "__main__":
    check_leakage()


