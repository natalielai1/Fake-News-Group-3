import pandas as pd
import sys
import os
from src.preprocessing import clean_source_text

# Increase CSV field limit to handle large text fields
import csv
csv.field_size_limit(sys.maxsize)

def debug_make_split():
    print("Starting debug script for make_split logic...")
    
    input_path = "data/processed/preprocessed_dataset.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    try:
        # Read in chunks to avoid memory issues which might be causing segfault (exit 139)
        chunk_size = 5000
        print(f"Reading {input_path} in chunks...")
        
        processed_chunks = []
        
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
            if i == 0:
                print(f"Processing first chunk with columns: {chunk.columns.tolist()}")
                
            # Apply cleaning
            chunk['text'] = chunk['text'].apply(clean_source_text)
            processed_chunks.append(chunk)
            
            if i % 5 == 0:
                print(f"Processed chunk {i}...")
                
        print("Concatenating chunks...")
        df = pd.concat(processed_chunks, ignore_index=True)
        
        print(f"Total rows loaded and cleaned: {len(df)}")
        
        # Deduplicate
        initial_len = len(df)
        df = df.drop_duplicates(subset=['text'])
        print(f"Removed {initial_len - len(df)} duplicate rows. New size: {len(df)}")
        
        # Save for verification (optional, or overwrite train/test manually here if needed)
        # For now just verifying the process completes without segfault
        print("Process completed successfully in memory.")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    debug_make_split()


