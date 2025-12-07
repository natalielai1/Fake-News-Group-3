import csv
import sys

csv.field_size_limit(sys.maxsize)

def check_source_bias():
    print("Checking for source attribution bias (e.g. 'Reuters')...")
    
    train_path = "data/processed/train.csv"
    
    reuters_real = 0
    reuters_fake = 0
    total_real = 0
    total_fake = 0
    
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row['label'])
                text = row['text']
                
                if label == 0: # Assuming 0 is Real (based on notebook output "Real -> 0: 27696")
                    total_real += 1
                    if "(Reuters)" in text or "Reuters" in text[:50]: # Check start of text
                        reuters_real += 1
                else: # 1 is Fake
                    total_fake += 1
                    if "(Reuters)" in text or "Reuters" in text[:50]:
                        reuters_fake += 1
                        
        print(f"Total Real: {total_real}")
        print(f"Total Fake: {total_fake}")
        print(f"Real with 'Reuters' signature: {reuters_real} ({reuters_real/total_real*100:.1f}%)")
        print(f"Fake with 'Reuters' signature: {reuters_fake} ({reuters_fake/total_fake*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_source_bias()


