import re

def clean_source_text(text: str) -> str:
    if text is None: return ""
    
    # Same regex as tokenizer.py
    reuters_pattern = r"^([A-Z\s]+)?\s*\((Reuters|AP|AFP|UPI)\)\s*-\s*"
    return re.sub(reuters_pattern, "", text, flags=re.IGNORECASE)

examples = [
    "PARIS (Reuters) - France will discuss...",
    "WASHINGTON (Reuters) - The president...",
    "(Reuters) - Short version...",
    "LONDON (AFP) - Another source.",
    "NEW YORK (AP) - Associated Press.",
    "BEIJING (REUTERS) - All caps source.",
    "tokyo (reuters) - lowercase source.",
    "Just text no source.",
    "Text with (Reuters) in the middle."
]

print(f"{'Original':<50} | {'Cleaned':<50}")
print("-" * 105)

for text in examples:
    cleaned = clean_source_text(text)
    print(f"{text:<50} | {cleaned:<50}")

