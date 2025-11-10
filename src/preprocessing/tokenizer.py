"""Text tokenization and preprocessing utilities."""

import re
import string
from typing import List, Callable, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only runs once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Initialize stemmers and lemmatizers
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

# Simple regex tokenizer - splits text into words, keeping apostrophes
WORD_RE = re.compile(r"[A-Za-z0-9']+")

# English stopwords
STOPWORDS = set(stopwords.words('english'))


def simple_tokenize(text: str) -> List[str]:
    """
    Convert a text string into a list of word-like tokens.
    
    Uses regex pattern to match words containing letters, numbers, and apostrophes.
    Examples:
        - "don't" -> ["don't"]
        - "Hello, world!" -> ["Hello", "world"]
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens (words with apostrophes preserved)
    """
    if text is None:
        return []
    return WORD_RE.findall(str(text))


def to_lowercase(tokens: List[str]) -> List[str]:
    """
    Convert all tokens to lowercase.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of lowercase tokens
        
    Example:
        >>> to_lowercase(["Hello", "WORLD"])
        ["hello", "world"]
    """
    return [token.lower() for token in tokens]


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with punctuation removed
        
    Example:
        >>> remove_punctuation("Hello, world!")
        "Hello world"
    """
    if text is None:
        return ""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[set] = None) -> List[str]:
    """
    Remove stopwords from token list.
    
    Args:
        tokens: List of tokens
        custom_stopwords: Optional set of custom stopwords (uses English stopwords by default)
        
    Returns:
        List of tokens with stopwords removed
        
    Example:
        >>> remove_stopwords(["this", "is", "a", "test"])
        ["test"]
    """
    stopwords_set = custom_stopwords if custom_stopwords is not None else STOPWORDS
    return [token for token in tokens if token.lower() not in stopwords_set]


def remove_short_tokens(tokens: List[str], min_length: int = 2) -> List[str]:
    """
    Remove tokens shorter than minimum length.
    
    Args:
        tokens: List of tokens
        min_length: Minimum token length to keep
        
    Returns:
        List of tokens meeting minimum length
        
    Example:
        >>> remove_short_tokens(["I", "am", "a", "developer"], min_length=2)
        ["am", "developer"]
    """
    return [token for token in tokens if len(token) >= min_length]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply Porter Stemmer to tokens.
    
    Stemming reduces words to their root form by removing suffixes.
    Note: May produce non-words (e.g., "running" -> "run", "studies" -> "studi")
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of stemmed tokens
        
    Example:
        >>> stem_tokens(["running", "runs", "runner"])
        ["run", "run", "runner"]
    """
    return [_stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: List[str], pos: str = 'n') -> List[str]:
    """
    Apply WordNet Lemmatizer to tokens.
    
    Lemmatization reduces words to their base dictionary form.
    Produces valid words (e.g., "running" -> "run", "studies" -> "study")
    
    Args:
        tokens: List of tokens
        pos: Part of speech ('n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb)
        
    Returns:
        List of lemmatized tokens
        
    Example:
        >>> lemmatize_tokens(["running", "runs", "runner"], pos='v')
        ["run", "run", "runner"]
    """
    return [_lemmatizer.lemmatize(token, pos=pos) for token in tokens]


def remove_numbers(tokens: List[str]) -> List[str]:
    """
    Remove tokens that are purely numeric.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of non-numeric tokens
        
    Example:
        >>> remove_numbers(["hello", "123", "world", "42"])
        ["hello", "world"]
    """
    return [token for token in tokens if not token.isdigit()]


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stops: bool = True,
    remove_nums: bool = False,
    min_length: int = 2,
    stem: bool = False,
    lemmatize: bool = False,
    custom_stopwords: Optional[set] = None
) -> List[str]:
    """
    Complete text preprocessing pipeline.
    
    Applies multiple preprocessing steps in sequence:
    1. Remove punctuation (optional)
    2. Tokenize
    3. Lowercase (optional)
    4. Remove stopwords (optional)
    5. Remove numbers (optional)
    6. Remove short tokens
    7. Stem or lemmatize (optional, mutually exclusive - lemmatize takes priority)
    
    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation before tokenizing
        remove_stops: Remove stopwords
        remove_nums: Remove numeric tokens
        min_length: Minimum token length to keep
        stem: Apply stemming
        lemmatize: Apply lemmatization (takes priority over stem)
        custom_stopwords: Optional custom stopwords set
        
    Returns:
        List of preprocessed tokens
        
    Example:
        >>> preprocess_text("The dogs are running!", lowercase=True, remove_stops=True)
        ["dogs", "running"]
    """
    if text is None:
        return []
    
    # Convert to string
    text = str(text)
    
    # Remove punctuation
    if remove_punct:
        text = remove_punctuation(text)
    
    # Tokenize
    tokens = simple_tokenize(text)
    
    # Lowercase
    if lowercase:
        tokens = to_lowercase(tokens)
    
    # Remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens, custom_stopwords)
    
    # Remove numbers
    if remove_nums:
        tokens = remove_numbers(tokens)
    
    # Remove short tokens
    tokens = remove_short_tokens(tokens, min_length)
    
    # Lemmatize or stem (lemmatize takes priority)
    if lemmatize:
        tokens = lemmatize_tokens(tokens, pos='v')
    elif stem:
        tokens = stem_tokens(tokens)
    
    return tokens


def tokenize_dataframe(df, text_column: str = "text", output_column: str = "tokens"):
    """
    Apply tokenization to a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text to tokenize
        output_column: Name of new column to store tokens
        
    Returns:
        DataFrame with new tokens column
    """
    df = df.copy()
    df[output_column] = df[text_column].astype(str).apply(simple_tokenize)
    
    return df


def preprocess_dataframe(
    df,
    text_column: str = "text",
    output_column: str = "tokens",
    **preprocess_kwargs
):
    """
    Apply full preprocessing pipeline to a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text to preprocess
        output_column: Name of new column to store processed tokens
        **preprocess_kwargs: Additional arguments passed to preprocess_text()
        
    Returns:
        DataFrame with new preprocessed tokens column
        
    Example:
        >>> df = preprocess_dataframe(
        ...     df,
        ...     text_column="text",
        ...     output_column="clean_tokens",
        ...     lowercase=True,
        ...     remove_stops=True,
        ...     lemmatize=True
        ... )
    """
    df = df.copy()
    df[output_column] = df[text_column].astype(str).apply(
        lambda text: preprocess_text(text, **preprocess_kwargs)
    )
    
    return df

