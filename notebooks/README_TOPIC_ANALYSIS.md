# Topic Analysis Notebook Guide

## Overview

The `topic_analysis.ipynb` notebook performs unsupervised topic modeling on the preprocessed fake news dataset using Latent Dirichlet Allocation (LDA).

## What It Does

1. **Loads Preprocessed Data** - Uses the cleaned and tokenized dataset
2. **Creates Document-Term Matrix** - Vectorizes text using CountVectorizer
3. **Trains LDA Model** - Discovers latent topics in the corpus
4. **Visualizes Topics** - Shows word clouds and distributions
5. **Analyzes Patterns** - Compares fake vs real news by topic
6. **Saves Results** - Exports topic assignments and keywords

## Prerequisites

### Required Data
- Preprocessed dataset: `data/processed/preprocessed_dataset.csv`

Run the preprocessing script first if you haven't:
```bash
python scripts/preprocess_data.py
```

### Required Packages
All packages should be in your `requirements.txt`:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- tqdm

## How to Run

1. **Activate your environment:**
```bash
conda activate fake-news-detection
```

2. **Open Jupyter:**
```bash
jupyter notebook notebooks/topic_analysis.ipynb
```

3. **Run all cells** (Cell → Run All) or execute cells sequentially

## Expected Runtime

- **Small dataset (<10K articles)**: 2-5 minutes
- **Medium dataset (10K-100K articles)**: 5-15 minutes
- **Large dataset (>100K articles)**: 15-30 minutes

LDA training time depends on:
- Number of documents
- Vocabulary size
- Number of topics
- Number of iterations

## Key Parameters

You can adjust these in the notebook:

### Topic Modeling
```python
N_TOPICS = 10        # Number of topics to extract
MAX_FEATURES = 5000  # Vocabulary size
MIN_DF = 5           # Min document frequency
MAX_DF = 0.7         # Max document frequency
```

### Tips for Tuning:
- **Too few topics** → Topics are too broad
- **Too many topics** → Topics overlap or are too specific
- Start with 10-15 topics and adjust based on coherence

## Outputs

### 1. Dataset with Topics
**File**: `data/processed/dataset_with_topics.csv`

Contains:
- Original columns (id, title, text, label)
- `dominant_topic`: Primary topic for each article
- `topic_0_prob`, `topic_1_prob`, ... : Probability distribution

### 2. Topic Keywords
**File**: `results/models/topic_keywords.pkl`

Python dictionary with top 20 keywords per topic:
```python
import pickle
with open('results/models/topic_keywords.pkl', 'rb') as f:
    keywords = pickle.load(f)
```

## Visualizations

The notebook generates:

1. **Topic Distribution Charts**
   - Overall distribution of articles across topics
   - Fake vs Real news by topic

2. **Word Clouds**
   - Visual representation of top words per topic
   - Larger words = higher importance

3. **Topic Similarity Heatmap**
   - Shows which topics are related
   - Based on word distribution similarity

4. **Sample Articles**
   - Representative examples from each topic
   - Helps interpret what each topic represents

## Interpreting Results

### Good Topic Model Signs:
✅ Topics have distinct, coherent keywords
✅ Sample articles in each topic make sense together
✅ Topics don't heavily overlap (low similarity scores)
✅ Distribution is not too skewed (no single dominant topic)

### Poor Topic Model Signs:
❌ Topics have very similar keywords
❌ Sample articles seem unrelated
❌ High similarity scores between many topics
❌ Most articles in 1-2 topics

### If Results Are Poor:
1. Try different number of topics
2. Adjust vocabulary size (MAX_FEATURES)
3. Change preprocessing (more/less aggressive stopword removal)
4. Use different min_df/max_df thresholds

## Key Findings to Look For

### 1. Topic-Label Association
- Are certain topics more common in fake news?
- Are certain topics more common in real news?
- Example: Topic 3 might be 80% fake news → useful feature!

### 2. Topic Characteristics
- Political topics
- Entertainment/celebrity topics
- Science/health topics
- Economic topics

### 3. Linguistic Patterns
- Do fake news topics use more emotional language?
- Are real news topics more specific/technical?

## Using Results for Classification

The topic probabilities can be powerful features:

```python
# Load data with topics
df = pd.read_csv('data/processed/dataset_with_topics.csv')

# Extract topic features
topic_features = df[[f'topic_{i}_prob' for i in range(10)]]

# Combine with other features for classification
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(topic_features, df['label'])
```

## Common Issues

### Issue: Memory Error
**Solution**: Reduce MAX_FEATURES or use a sample of data

### Issue: Topics Don't Make Sense
**Solution**: 
- Check preprocessing quality
- Adjust number of topics
- Tune min_df/max_df parameters

### Issue: Very Slow Training
**Solution**:
- Reduce max_iter (default is 20)
- Reduce MAX_FEATURES
- Use smaller sample for testing

### Issue: Import Errors
**Solution**:
```bash
pip install wordcloud tqdm
```

## Advanced Usage

### Experiment with Topic Numbers
```python
# Try different numbers of topics
for n_topics in [5, 10, 15, 20]:
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(doc_term_matrix)
    print(f"{n_topics} topics: perplexity={lda.perplexity(doc_term_matrix)}")
```

### Try NMF Instead of LDA
```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=10, random_state=42)
nmf_output = nmf.fit_transform(tfidf_matrix)  # Use TF-IDF for NMF
```

### Calculate Topic Coherence
```python
# Using gensim (requires separate install)
from gensim.models import CoherenceModel
# Calculate coherence score for model quality
```

## Next Steps After Topic Analysis

1. **Feature Engineering**: Add topic probabilities to your feature set
2. **Build Classifiers**: Train models using topic features
3. **Ensemble Models**: Combine topic-based and text-based features
4. **Interpretability**: Use topics to explain model predictions

## References

- [LDA Paper](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) - Blei et al. 2003
- [Scikit-learn LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [Topic Modeling Best Practices](https://towardsdatascience.com/beginners-guide-to-lda-topic-modelling-with-python-2dce9f5f6d16)

## Questions?

Check the notebook comments and docstrings for detailed explanations of each step.


