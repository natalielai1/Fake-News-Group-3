import csv
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

def check_leakage_csv():
    print("Checking leakage with csv module...")
    
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    train_hashes = set()
    
    print("Reading train...")
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                if row_count < 3:
                     print(f"Row {row_count} Text Start: {row['text'][:100]}...")
                
                if 'text' in row:
                    train_hashes.add(hash(row['text']))
            print(f"Train rows: {row_count}")
            print(f"Unique train hashes: {len(train_hashes)}")
    except Exception as e:
        print(f"Error reading train: {e}")
        return

    print("Reading test...")
    overlap_count = 0
    test_count = 0
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_count += 1
                if 'text' in row:
                    if hash(row['text']) in train_hashes:
                        overlap_count += 1
            print(f"Test rows: {test_count}")
            print(f"Overlapping texts: {overlap_count}")
    except Exception as e:
        print(f"Error reading test: {e}")

if __name__ == "__main__":
    check_leakage_csv()


