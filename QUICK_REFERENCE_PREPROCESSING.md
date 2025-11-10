# Preprocessing Quick Reference Card

## Installation (Run Once)

```bash
conda activate fake-news-detection
# NLTK is already in requirements.txt, data downloads automatically on first use
```

## Import

```python
from src.preprocessing import preprocess_text, preprocess_dataframe
```

## Most Common Usage

### Single Text
```python
# Standard preprocessing
tokens = preprocess_text(text, lemmatize=True)

# Minimal (just tokenize + lowercase)
tokens = preprocess_text(text, remove_stops=False, remove_punct=True)

# Maximum cleaning
tokens = preprocess_text(text, lemmatize=True, remove_nums=True, min_length=3)
```

### DataFrame
```python
df = preprocess_dataframe(
    df,
    text_column="text",
    output_column="tokens",
    lemmatize=True,
    min_length=3
)
```

## Function Cheat Sheet

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `preprocess_text()` | str | List[str] | Single text preprocessing |
| `preprocess_dataframe()` | DataFrame | DataFrame | Batch preprocessing |
| `simple_tokenize()` | str | List[str] | Basic tokenization only |
| `remove_stopwords()` | List[str] | List[str] | Remove common words |
| `lemmatize_tokens()` | List[str] | List[str] | Get word base forms |
| `stem_tokens()` | List[str] | List[str] | Get word stems |

## Parameters Quick Guide

```python
preprocess_text(
    text,
    lowercase=True,      # a-z normalization
    remove_punct=True,   # strip .,!? etc.
    remove_stops=True,   # remove the, a, is, etc.
    remove_nums=False,   # keep/remove 123
    min_length=2,        # token length filter
    stem=False,          # Porter stemmer
    lemmatize=False      # WordNet lemmatizer (better than stem)
)
```

## Common Patterns

### For ML Classification
```python
tokens = preprocess_text(text, lemmatize=True, min_length=3)
```

### For Topic Modeling
```python
tokens = preprocess_text(text, lemmatize=True, remove_nums=True, min_length=3)
```

### For Word Embeddings
```python
tokens = preprocess_text(text, remove_stops=False, lemmatize=True)
```

### Keep Everything
```python
tokens = preprocess_text(text, lowercase=True, remove_stops=False, remove_punct=False)
```

## Example Output

```python
text = "The dogs are running quickly! They ran 123 times."

# Default preprocessing
preprocess_text(text)
# ['dogs', 'running', 'quickly', 'ran', '123', 'times']

# With lemmatization
preprocess_text(text, lemmatize=True)
# ['dog', 'run', 'quickly', 'run', '123', 'time']

# Remove numbers, min length 3
preprocess_text(text, lemmatize=True, remove_nums=True, min_length=3)
# ['dog', 'run', 'quickly', 'run', 'time']
```

## Stopwords

```python
from src.preprocessing import STOPWORDS

# View stopwords
print(len(STOPWORDS))  # 179 words

# Custom stopwords
custom = STOPWORDS | {'additional', 'words'}
tokens = preprocess_text(text, custom_stopwords=custom)
```

## Stemming vs Lemmatization

```python
text = "running runs runner"

# Stemming (faster, less accurate)
stem_tokens(text.split())
# ['run', 'run', 'runner']

# Lemmatization (slower, more accurate)
lemmatize_tokens(text.split(), pos='v')
# ['run', 'run', 'runner']

# In pipeline (lemmatize preferred)
preprocess_text(text, stem=True)       # uses stemming
preprocess_text(text, lemmatize=True)  # uses lemmatization
```

## Run Scripts

```bash
# Demo all features
python scripts/demo_preprocessing.py

# Process full dataset
python scripts/preprocess_data.py

# Run tests
pytest tests/test_preprocessing.py -v
```

## Files Added

- ✅ `src/preprocessing/tokenizer.py` - Enhanced with 9 new functions
- ✅ `scripts/demo_preprocessing.py` - Demo script
- ✅ `tests/test_preprocessing.py` - Unit tests
- ✅ `PREPROCESSING_GUIDE.md` - Full documentation
- ✅ `src/preprocessing/README.md` - Module docs
- ✅ `PREPROCESSING_FEATURES.md` - Feature summary

## Need Help?

1. **Full documentation:** `PREPROCESSING_GUIDE.md`
2. **See examples:** `python scripts/demo_preprocessing.py`
3. **Module reference:** `src/preprocessing/README.md`
4. **Feature list:** `PREPROCESSING_FEATURES.md`

