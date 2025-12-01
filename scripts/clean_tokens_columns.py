import csv
import sys
import ast
import os

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

SOURCES = {'reuters', 'ap', 'afp', 'upi'}
# Common dateline locations to catch at the start
LOCATIONS = {
    'washington', 'paris', 'london', 'berlin', 'beijing', 'moscow', 'tokyo', 
    'new', 'york', 'dubai', 'jakarta', 'istanbul', 'brussels', 'jerusalem', 
    'geneva', 'cairo', 'beirut', 'frankfurt', 'madrid', 'rome', 'vienna', 
    'warsaw', 'seoul', 'taipei', 'shanghai', 'hong', 'kong', 'dublin', 'ottawa',
    'toronto', 'sydney', 'melbourne', 'brasilia', 'mexico', 'city'
}

def clean_token_list(token_str: str, is_processed: bool = False) -> str:
    try:
        # Parse string representation of list
        tokens = ast.literal_eval(token_str)
        if not tokens:
            return token_str
            
        # Check first few tokens for source attribution
        # Strategy: Find the source (e.g., 'reuters') in the first 5 tokens
        # If found, remove it and everything before it.
        
        found_index = -1
        for i in range(min(5, len(tokens))):
            token_lower = tokens[i].lower()
            if token_lower in SOURCES:
                found_index = i
                break
        
        if found_index != -1:
            # Check if we should remove " - " after source (common in raw tokens)
            # In processed_tokens, punctuation is usually gone.
            end_removal_index = found_index + 1
            
            if not is_processed and end_removal_index < len(tokens):
                if tokens[end_removal_index] == '-':
                    end_removal_index += 1
            
            # Remove everything up to and including the source (and optional hyphen)
            # This effectively removes "PARIS", "Reuters", "-"
            cleaned_tokens = tokens[end_removal_index:]
            return str(cleaned_tokens)
            
        return token_str
    except:
        return token_str

def process_file(filepath):
    print(f"Processing {filepath}...")
    temp_path = filepath + ".tmp"
    
    cleaned_count = 0
    total_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as fin, \
             open(temp_path, 'w', encoding='utf-8', newline='') as fout:
            
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                total_count += 1
                
                # Clean 'tokens'
                orig_tokens = row.get('tokens', '[]')
                new_tokens = clean_token_list(orig_tokens, is_processed=False)
                if new_tokens != orig_tokens:
                    cleaned_count += 1
                row['tokens'] = new_tokens
                
                # Clean 'processed_tokens'
                orig_proc = row.get('processed_tokens', '[]')
                new_proc = clean_token_list(orig_proc, is_processed=True)
                row['processed_tokens'] = new_proc
                
                writer.writerow(row)
                
        os.replace(temp_path, filepath)
        print(f"✓ Updated {filepath} ({cleaned_count} rows cleaned)")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    process_file("data/processed/train.csv")
    process_file("data/processed/test.csv")

