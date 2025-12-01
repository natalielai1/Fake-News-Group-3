import pandas as pd
import os

def check_leakage():
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    except FileNotFoundError:
        print("Train/test files not found.")
        return

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    # Check for exact text duplicates
    train_texts = set(train_df['text'].unique())
    test_texts = set(test_df['text'].unique())
    
    overlap = train_texts.intersection(test_texts)
    print(f"Number of overlapping texts: {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"Percentage of test set in train set: {len(overlap) / len(test_df) * 100:.2f}%")
        
    # Check for duplicates within train and test
    print(f"Duplicates in train: {len(train_df) - len(train_texts)}")
    print(f"Duplicates in test: {len(test_df) - len(test_texts)}")

if __name__ == "__main__":
    check_leakage()
