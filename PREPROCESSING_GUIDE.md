# Text Preprocessing Guide

This guide explains how to use the text preprocessing utilities in this project.

## Installation

First, ensure all dependencies are installed:

```bash
# Using conda (recommended)
conda activate fake-news-detection
conda install nltk

# Or using pip
pip install nltk
```

## Available Preprocessing Functions

### 1. Basic Tokenization

```python
from src.preprocessing import simple_tokenize

text = "Hello, world! Don't forget to visit example.com today."
tokens = simple_tokenize(text)
# Result: ['Hello', 'world', "Don't", 'forget', 'to', 'visit', 'example', 'com', 'today']
```

### 2. Individual Preprocessing Steps

```python
from src.preprocessing import (
    to_lowercase,
    remove_punctuation,
    remove_stopwords,
    remove_short_tokens,
    stem_tokens,
    lemmatize_tokens,
    remove_numbers
)

# Lowercase
tokens = to_lowercase(['Hello', 'WORLD'])
# Result: ['hello', 'world']

# Remove punctuation (works on text before tokenization)
text = remove_punctuation("Hello, world!")
# Result: "Hello world"

# Remove stopwords
tokens = remove_stopwords(['this', 'is', 'a', 'test'])
# Result: ['test']

# Remove short tokens
tokens = remove_short_tokens(['I', 'am', 'a', 'developer'], min_length=2)
# Result: ['am', 'developer']

# Stemming
tokens = stem_tokens(['running', 'runs', 'runner'])
# Result: ['run', 'run', 'runner']

# Lemmatization
tokens = lemmatize_tokens(['running', 'runs', 'runner'], pos='v')
# Result: ['run', 'run', 'runner']

# Remove numbers
tokens = remove_numbers(['hello', '123', 'world', '42'])
# Result: ['hello', 'world']
```

### 3. Complete Preprocessing Pipeline

The `preprocess_text()` function combines all preprocessing steps:

```python
from src.preprocessing import preprocess_text

text = "The Dogs are Running quickly! They ran 123 times today."

# Basic preprocessing (default settings)
tokens = preprocess_text(text)
# Result: ['dogs', 'running', 'quickly', 'ran', '123', 'times', 'today']

# With stemming
tokens = preprocess_text(text, stem=True)
# Result: ['dog', 'run', 'quick', 'ran', '123', 'time', 'today']

# With lemmatization
tokens = preprocess_text(text, lemmatize=True)
# Result: ['dog', 'run', 'quickly', 'run', '123', 'time', 'today']

# Remove numbers
tokens = preprocess_text(text, remove_nums=True)
# Result: ['dogs', 'running', 'quickly', 'ran', 'times', 'today']

# Custom configuration
tokens = preprocess_text(
    text,
    lowercase=True,       # Convert to lowercase (default: True)
    remove_punct=True,    # Remove punctuation (default: True)
    remove_stops=True,    # Remove stopwords (default: True)
    remove_nums=False,    # Remove numeric tokens (default: False)
    min_length=3,         # Minimum token length (default: 2)
    stem=False,           # Apply stemming (default: False)
    lemmatize=True        # Apply lemmatization (default: False)
)
```

### 4. DataFrame Preprocessing

Process entire DataFrame columns efficiently:

```python
import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.data.load_data import load_kaggle_dataset

# Load data
df = load_kaggle_dataset("data/raw/WELFake_Dataset.csv")

# Apply preprocessing to a column
df = preprocess_dataframe(
    df,
    text_column="text",
    output_column="clean_tokens",
    lowercase=True,
    remove_stops=True,
    lemmatize=True,
    min_length=3
)

# Now df has a new column 'clean_tokens' with preprocessed tokens
```

## Preprocessing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowercase` | bool | True | Convert all text to lowercase |
| `remove_punct` | bool | True | Remove punctuation marks |
| `remove_stops` | bool | True | Remove common stopwords (a, the, is, etc.) |
| `remove_nums` | bool | False | Remove purely numeric tokens |
| `min_length` | int | 2 | Minimum token length to keep |
| `stem` | bool | False | Apply Porter Stemmer (may produce non-words) |
| `lemmatize` | bool | False | Apply WordNet Lemmatizer (produces valid words) |
| `custom_stopwords` | set | None | Custom set of stopwords to use |

## Stemming vs Lemmatization

**Stemming** (Porter Stemmer):
- Faster but less accurate
- May produce non-words (e.g., "studies" → "studi")
- Good for: Information retrieval, search

**Lemmatization** (WordNet):
- Slower but more accurate
- Produces valid dictionary words (e.g., "studies" → "study")
- Good for: NLP models, text analysis

**Example:**
```python
words = ["running", "runs", "ran", "studies", "studying"]

# Stemming
stem_tokens(words)
# Result: ['run', 'run', 'ran', 'studi', 'study']

# Lemmatization
lemmatize_tokens(words, pos='v')
# Result: ['run', 'run', 'run', 'study', 'study']
```

## Running the Demo

To see all preprocessing functions in action:

```bash
python scripts/demo_preprocessing.py
```

## Preprocessing Pipeline Script

The main preprocessing script processes the entire dataset:

```bash
python scripts/preprocess_data.py
```

This will:
1. Load the raw WELFake dataset
2. Create basic tokens (preserves original)
3. Create processed tokens (cleaned for modeling)
4. Save both versions to `data/processed/cleaned_fake_news.csv`

## Usage in Notebooks

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

# Import preprocessing functions
from src.preprocessing import preprocess_text, preprocess_dataframe

# Use in your analysis
text = "Your text here..."
clean_tokens = preprocess_text(text, lemmatize=True)
```

## Custom Stopwords

You can provide custom stopwords if needed:

```python
from src.preprocessing import preprocess_text, STOPWORDS

# Use default stopwords
tokens = preprocess_text(text, remove_stops=True)

# Add custom stopwords
custom_stops = STOPWORDS | {'custom', 'word', 'list'}
tokens = preprocess_text(text, remove_stops=True, custom_stopwords=custom_stops)

# Use only custom stopwords
my_stops = {'only', 'these', 'words'}
tokens = preprocess_text(text, remove_stops=True, custom_stopwords=my_stops)
```

## Best Practices

1. **Keep original text**: Save both original and preprocessed versions for reference
2. **Choose stem OR lemmatize**: Don't use both (lemmatize takes priority if both are True)
3. **Experiment with parameters**: Different tasks may need different preprocessing
4. **Consider min_length**: Increase to 3 for cleaner results
5. **Numbers**: Keep them for certain tasks (dates, quantities matter)

## Common Preprocessing Pipelines

### For Classification Models
```python
tokens = preprocess_text(
    text,
    lowercase=True,
    remove_stops=True,
    lemmatize=True,
    min_length=3,
    remove_nums=False
)
```

### For Topic Modeling
```python
tokens = preprocess_text(
    text,
    lowercase=True,
    remove_stops=True,
    lemmatize=True,
    min_length=3,
    remove_nums=True
)
```

### For Word Embeddings
```python
tokens = preprocess_text(
    text,
    lowercase=True,
    remove_stops=False,  # Keep stopwords for context
    lemmatize=True,
    min_length=2,
    remove_nums=False
)
```

### Minimal Preprocessing
```python
tokens = preprocess_text(
    text,
    lowercase=True,
    remove_stops=False,
    remove_punct=True,
    min_length=1
)
```

