import re

def clean_source_text(text):
    if text is None: return ""
    # Regex from tokenizer.py
    reuters_pattern = r"^([A-Z\s]+)?\s*\(Reuters\)\s*-\s*"
    return re.sub(reuters_pattern, "", text, flags=re.IGNORECASE)

examples = [
    "WASHINGTON (Reuters) - The president said today...",
    "PARIS (Reuters) - France will discuss...",
    "(Reuters) - Short version...",
    "No source here."
]

for t in examples:
    print(f"'{t}' -> '{clean_source_text(t)}'")


