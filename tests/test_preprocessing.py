"""Unit tests for text preprocessing functions."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
from src.preprocessing import (
    simple_tokenize,
    to_lowercase,
    remove_punctuation,
    remove_stopwords,
    remove_short_tokens,
    stem_tokens,
    lemmatize_tokens,
    remove_numbers,
    preprocess_text,
    preprocess_dataframe,
)


class TestBasicTokenization:
    """Tests for basic tokenization."""
    
    def test_simple_tokenize(self):
        """Test basic tokenization."""
        text = "Hello, world! Don't stop."
        result = simple_tokenize(text)
        assert result == ["Hello", "world", "Don't", "stop"]
    
    def test_tokenize_none(self):
        """Test tokenization with None input."""
        result = simple_tokenize(None)
        assert result == []
    
    def test_tokenize_empty(self):
        """Test tokenization with empty string."""
        result = simple_tokenize("")
        assert result == []
    
    def test_tokenize_with_numbers(self):
        """Test tokenization preserves numbers."""
        text = "I have 123 apples"
        result = simple_tokenize(text)
        assert "123" in result


class TestIndividualSteps:
    """Tests for individual preprocessing steps."""
    
    def test_to_lowercase(self):
        """Test lowercase conversion."""
        tokens = ["Hello", "WORLD", "Test"]
        result = to_lowercase(tokens)
        assert result == ["hello", "world", "test"]
    
    def test_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, world! How are you?"
        result = remove_punctuation(text)
        assert result == "Hello world How are you"
    
    def test_remove_punctuation_none(self):
        """Test punctuation removal with None."""
        result = remove_punctuation(None)
        assert result == ""
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["this", "is", "a", "test", "sentence"]
        result = remove_stopwords(tokens)
        assert "test" in result
        assert "sentence" in result
        assert "is" not in result
        assert "a" not in result
    
    def test_remove_stopwords_custom(self):
        """Test stopword removal with custom stopwords."""
        tokens = ["hello", "world", "test"]
        custom_stops = {"hello", "world"}
        result = remove_stopwords(tokens, custom_stopwords=custom_stops)
        assert result == ["test"]
    
    def test_remove_short_tokens(self):
        """Test removal of short tokens."""
        tokens = ["I", "am", "a", "developer"]
        result = remove_short_tokens(tokens, min_length=2)
        assert "am" in result
        assert "developer" in result
        assert "I" not in result
        assert "a" not in result
    
    def test_remove_numbers(self):
        """Test number removal."""
        tokens = ["hello", "123", "world", "42", "test"]
        result = remove_numbers(tokens)
        assert result == ["hello", "world", "test"]
    
    def test_stem_tokens(self):
        """Test stemming."""
        tokens = ["running", "runs", "runner"]
        result = stem_tokens(tokens)
        # Porter Stemmer should reduce to common stem
        assert result[0] == result[1]  # running and runs -> run
    
    def test_lemmatize_tokens(self):
        """Test lemmatization."""
        tokens = ["running", "runs"]
        result = lemmatize_tokens(tokens, pos='v')
        # Both should lemmatize to 'run'
        assert result == ["run", "run"]


class TestPreprocessText:
    """Tests for complete preprocessing pipeline."""
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing with defaults."""
        text = "The dogs are running!"
        result = preprocess_text(text)
        # Should lowercase, remove stopwords, remove punctuation
        assert "dogs" in result
        assert "running" in result
        assert "the" not in result  # stopword
        assert "are" not in result  # stopword
    
    def test_with_stemming(self):
        """Test preprocessing with stemming."""
        text = "The dogs are running"
        result = preprocess_text(text, stem=True, remove_stops=True)
        # Should stem 'running' to 'run' and 'dogs' to 'dog'
        assert "run" in result
        assert "dog" in result
    
    def test_with_lemmatization(self):
        """Test preprocessing with lemmatization."""
        text = "The dogs are running"
        result = preprocess_text(text, lemmatize=True, remove_stops=True)
        # Should lemmatize properly
        assert "dog" in result
        assert "run" in result
    
    def test_remove_numbers_option(self):
        """Test number removal option."""
        text = "I have 123 apples"
        result = preprocess_text(text, remove_nums=True)
        assert "123" not in result
        assert "apples" in result
    
    def test_minimal_preprocessing(self):
        """Test minimal preprocessing."""
        text = "Hello World"
        result = preprocess_text(
            text,
            lowercase=False,
            remove_stops=False,
            remove_punct=False
        )
        # Should preserve case and all words
        assert "Hello" in result
        assert "World" in result
    
    def test_none_input(self):
        """Test preprocessing with None input."""
        result = preprocess_text(None)
        assert result == []
    
    def test_min_length_filter(self):
        """Test minimum length filtering."""
        text = "I am a developer"
        result = preprocess_text(text, min_length=3, remove_stops=False)
        assert "developer" in result
        assert "I" not in result
        assert "am" not in result


class TestDataFramePreprocessing:
    """Tests for DataFrame preprocessing functions."""
    
    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame({
            'text': ['Hello world!', 'The dogs are running', 'Test 123']
        })
        
        result = preprocess_dataframe(
            df,
            text_column='text',
            output_column='tokens',
            lowercase=True,
            remove_stops=True
        )
        
        assert 'tokens' in result.columns
        assert isinstance(result.iloc[0]['tokens'], list)
        assert len(result.iloc[0]['tokens']) > 0
    
    def test_preprocess_dataframe_with_lemmatization(self):
        """Test DataFrame preprocessing with lemmatization."""
        df = pd.DataFrame({
            'text': ['The dogs are running']
        })
        
        result = preprocess_dataframe(
            df,
            text_column='text',
            output_column='clean_tokens',
            lemmatize=True,
            remove_stops=True
        )
        
        tokens = result.iloc[0]['clean_tokens']
        assert "dog" in tokens
        assert "run" in tokens


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_string(self):
        """Test with empty string."""
        result = preprocess_text("")
        assert result == []
    
    def test_only_punctuation(self):
        """Test with only punctuation."""
        result = preprocess_text("!!!")
        assert result == []
    
    def test_only_stopwords(self):
        """Test with only stopwords."""
        text = "the a an is"
        result = preprocess_text(text, remove_stops=True)
        assert result == []
    
    def test_unicode_text(self):
        """Test with unicode characters."""
        text = "Hello 世界"
        result = preprocess_text(text)
        # Should handle unicode gracefully
        assert isinstance(result, list)
    
    def test_very_long_text(self):
        """Test with very long text."""
        text = "word " * 10000
        result = preprocess_text(text)
        # Should handle without errors
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

