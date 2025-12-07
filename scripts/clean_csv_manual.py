import csv
import re
import os
import sys

# Increase CSV field limit
csv.field_size_limit(sys.maxsize)

def clean_source_text(text: str) -> str:
    if not text: return ""
    reuters_pattern = r"^([A-Z\s]+)?\s*\((Reuters|AP|AFP|UPI)\)\s*-\s*"
    return re.sub(reuters_pattern, "", text, flags=re.IGNORECASE)

def clean_csv_pure_python():
    print("Cleaning CSV using standard library only...")
    input_path = "data/processed/train.csv"
    output_path = "data/processed/train_cleaned.csv"
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8', newline='') as fout:
            
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            
            count = 0
            cleaned_count = 0
            
            for row in reader:
                original = row['text']
                cleaned = clean_source_text(original)
                
                if original != cleaned:
                    cleaned_count += 1
                    
                row['text'] = cleaned
                writer.writerow(row)
                count += 1
                
                if count % 5000 == 0:
                    print(f"Processed {count} rows...")
                    
        print(f"Finished! Processed {count} rows.")
        print(f"Cleaned {cleaned_count} rows containing source attribution.")
        
        # Replace original file
        os.replace(output_path, input_path)
        print("Replaced original train.csv with cleaned version.")
        
        # Also clean test.csv
        print("\nCleaning test.csv...")
        input_test = "data/processed/test.csv"
        output_test = "data/processed/test_cleaned.csv"
        
        if os.path.exists(input_test):
            with open(input_test, 'r', encoding='utf-8') as fin, \
                 open(output_test, 'w', encoding='utf-8', newline='') as fout:
                
                reader = csv.DictReader(fin)
                writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                writer.writeheader()
                
                for row in reader:
                    row['text'] = clean_source_text(row['text'])
                    writer.writerow(row)
            
            os.replace(output_test, input_test)
            print("Replaced original test.csv with cleaned version.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_csv_pure_python()


