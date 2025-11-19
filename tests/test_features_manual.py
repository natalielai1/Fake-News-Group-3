import sys
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from the file path if module import fails
import importlib.util
spec = importlib.util.spec_from_file_location("make_features", os.path.join(os.path.dirname(__file__), '../src/features/make_features.py'))
make_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_features)

train_word2vec = make_features.train_word2vec
get_sentence_vector = make_features.get_sentence_vector
create_word2vec_features = make_features.create_word2vec_features

def test_word2vec_implementation():
    print("Testing Word2Vec implementation...")
    
    # Mock data
    texts = [
        "this is a test sentence",
        "another test sentence here",
        "word embeddings are useful",
        "machine learning is fun"
    ]
    
    # Test training
    print("1. Testing model training...")
    model = train_word2vec(texts, vector_size=10, window=2, min_count=1)
    assert isinstance(model, Word2Vec)
    assert model.vector_size == 10
    print("   Model trained successfully.")
    
    # Test sentence vector
    print("2. Testing sentence vector calculation...")
    vec = get_sentence_vector("this is a test", model)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (10,)
    print("   Sentence vector calculated successfully.")
    
    # Test empty sentence
    print("3. Testing empty sentence handling...")
    vec_empty = get_sentence_vector("", model)
    assert np.all(vec_empty == 0)
    print("   Empty sentence handled correctly.")
    
    # Test full pipeline function
    print("4. Testing create_word2vec_features...")
    X, model_out = create_word2vec_features(texts, vector_size=10)
    assert X.shape == (4, 10)
    assert model_out is not None
    print("   Feature creation successful.")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_word2vec_implementation()
