import sys
from src.preprocessing import clean_source_text

def verify_cleaning():
    examples = [
        "WASHINGTON (Reuters) - The president said today...",
        "PARIS (Reuters) - France will discuss...",
        "BERLIN (Reuters) - German officials...",
        "(Reuters) - Short version...",
        "No source here just text.",
        "Some text mentioning Reuters in the middle.",
        "LONDON (AFP) - Another source.",
        "NEW YORK (AP) - Associated Press."
    ]
    
    print("Verifying Source Cleaning:\n")
    for text in examples:
        cleaned = clean_source_text(text)
        print(f"Original: '{text}'")
        print(f"Cleaned:  '{cleaned}'")
        print("-" * 40)

if __name__ == "__main__":
    verify_cleaning()


