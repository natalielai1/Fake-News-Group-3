import pandas as pd
import csv
import sys

def check_leakage_efficient():
    print("Checking leakage with efficient memory usage...")
    
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    # Read first few rows to inspect 'text' column
    print("\n--- Inspecting Train Data Sample ---")
    try:
        df_head = pd.read_csv(train_path, nrows=3)
        print("Columns:", df_head.columns.tolist())
        for i, row in df_head.iterrows():
            text_preview = str(row['text'])[:200].replace('\n', ' ')
            print(f"Row {i} Text Start: {text_preview}...")
            print(f"Row {i} Label: {row['label']}")
    except Exception as e:
        print(f"Error reading head: {e}")

    # Check overlap using hashes
    print("\n--- Checking Text Overlap via Hashes ---")
    train_hashes = set()
    
    # Read train
    print("Reading train file...")
    try:
        # Use csv.DictReader to avoid pandas memory overhead if possible, 
        # but pandas with chunksize is usually fine if we process and discard.
        # Let's use pandas with chunks for robustness against CSV quoting issues.
        chunk_size = 5000
        for chunk in pd.read_csv(train_path, usecols=['text'], chunksize=chunk_size):
            for text in chunk['text']:
                train_hashes.add(hash(str(text)))
        print(f"Train unique text hashes: {len(train_hashes)}")
    except Exception as e:
        print(f"Error reading train: {e}")
        return

    # Read test and check
    print("Reading test file and checking overlap...")
    overlap_count = 0
    test_count = 0
    
    try:
        for chunk in pd.read_csv(test_path, usecols=['text'], chunksize=chunk_size):
            for text in chunk['text']:
                test_count += 1
                if hash(str(text)) in train_hashes:
                    overlap_count += 1
                    if overlap_count <= 3:
                        print(f"Overlap found! Hash: {hash(str(text))}")
    except Exception as e:
        print(f"Error reading test: {e}")
        return

    print(f"Total Test Rows Processed: {test_count}")
    print(f"Overlapping Texts: {overlap_count}")
    if overlap_count > 0:
        print(f"Leakage Percentage: {overlap_count/test_count*100:.2f}%")

if __name__ == "__main__":
    check_leakage_efficient()


