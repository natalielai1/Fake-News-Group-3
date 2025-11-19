# Topic Analysis Notebook - Summary

## ✅ What Was Created

### Main Notebook
**File**: `notebooks/topic_analysis.ipynb`

A comprehensive Jupyter notebook that performs unsupervised topic modeling on fake news articles.

### Documentation
**File**: `notebooks/README_TOPIC_ANALYSIS.md`

Complete guide with usage instructions, parameter tuning, and troubleshooting.

---

## 📊 What the Notebook Does

### 1. Data Loading
- Loads preprocessed dataset with clean tokens
- Converts token lists to text format for vectorization
- Shows dataset statistics and label distribution

### 2. Document-Term Matrix Creation
- Uses CountVectorizer with configurable parameters
- Creates vocabulary of 5,000 most important terms
- Filters by document frequency to remove noise

### 3. LDA Topic Modeling
- Trains Latent Dirichlet Allocation model
- Extracts 10 latent topics from the corpus
- Assigns topic probabilities to each article

### 4. Topic Visualization
- **Word Clouds**: Visual representation of each topic
- **Distribution Charts**: Articles per topic, Fake vs Real comparison
- **Similarity Heatmap**: Shows relationships between topics
- **Sample Articles**: Representative examples for each topic

### 5. Analysis
- Identifies topics associated with fake news
- Identifies topics associated with real news
- Calculates topic coherence and similarity
- Provides statistical summaries

### 6. Results Export
- Saves dataset with topic assignments
- Exports topic keywords for reference
- Ready for downstream classification tasks

---

## 🎯 Key Features

### Sections (14 total):
1. ✅ Setup and Imports
2. ✅ Load Preprocessed Data
3. ✅ Prepare Text for Topic Modeling
4. ✅ Create Document-Term Matrix
5. ✅ Train LDA Topic Model
6. ✅ Display Top Words for Each Topic
7. ✅ Assign Topics to Documents
8. ✅ Visualize Topic Distributions
9. ✅ Word Clouds for Topics
10. ✅ Sample Articles from Each Topic
11. ✅ Topic Analysis by Label (Fake vs Real)
12. ✅ Topic Similarity Heatmap
13. ✅ Save Results
14. ✅ Summary Statistics

---

## 📂 Output Files

### 1. dataset_with_topics.csv
**Location**: `data/processed/dataset_with_topics.csv`

Contains:
- All original columns (id, title, text, label)
- `dominant_topic`: Primary topic for each article
- `topic_0_prob` through `topic_9_prob`: Probability distributions

**Use**: Ready for feature engineering and classification

### 2. topic_keywords.pkl
**Location**: `results/models/topic_keywords.pkl`

Contains:
- Dictionary mapping topics to top 20 keywords
- Used for topic interpretation and visualization

---

## 🚀 How to Use

### Quick Start
```bash
# 1. Activate environment
conda activate fake-news-detection

# 2. Ensure data is preprocessed
python scripts/preprocess_data.py

# 3. Open notebook
jupyter notebook notebooks/topic_analysis.ipynb

# 4. Run all cells
```

### Expected Runtime
- **Small dataset** (<10K): 2-5 minutes
- **Medium dataset** (10K-100K): 5-15 minutes  
- **Large dataset** (>100K): 15-30 minutes

---

## 🔧 Configurable Parameters

### Easy to Adjust:
```python
N_TOPICS = 10        # Number of topics to extract
MAX_FEATURES = 5000  # Vocabulary size
MIN_DF = 5           # Minimum document frequency
MAX_DF = 0.7         # Maximum document frequency
```

### Tuning Tips:
- **More topics** → More specific groupings
- **Fewer topics** → Broader themes
- **Higher MIN_DF** → Remove rare words
- **Lower MAX_DF** → Remove common words

---

## 📈 Visualizations Included

### 1. Topic Distribution Bar Chart
- Shows how articles are distributed across topics
- Identifies dominant vs minor topics

### 2. Fake vs Real Comparison
- Stacked bar chart showing topic composition by label
- Highlights which topics lean fake or real

### 3. Word Clouds (6 topics shown)
- Beautiful visual of most important words per topic
- Larger words = more important

### 4. Similarity Heatmap
- Color-coded matrix showing topic relationships
- Helps identify overlapping themes

---

## 🎓 What You'll Learn

### From the Analysis:

1. **Content Themes**
   - What topics appear in the dataset?
   - Political, entertainment, health, etc.

2. **Fake vs Real Patterns**
   - Which topics have more fake news?
   - Which topics have more real news?

3. **Topic Quality**
   - Are topics coherent and distinct?
   - Do sample articles make sense?

4. **Feature Engineering**
   - Topic probabilities as ML features
   - Better than bag-of-words alone

---

## 💡 Next Steps After Running

### 1. Interpret Topics
- Review top words for each topic
- Read sample articles
- Name topics based on content

### 2. Analyze Patterns
- Which topics are fake-news heavy?
- Which topics are real-news heavy?
- Are there topic clusters?

### 3. Use for Classification
```python
# Topics as features for ML models
df = pd.read_csv('data/processed/dataset_with_topics.csv')
features = df[[f'topic_{i}_prob' for i in range(10)]]
labels = df['label']

# Train classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(features, labels)
```

### 4. Experiment
- Try different numbers of topics (5, 15, 20)
- Adjust preprocessing parameters
- Compare LDA vs NMF algorithms

---

## 🔍 Example Insights You Might Find

### Hypothetical Findings:

**Topic 0: Politics**
- Keywords: president, election, vote, government
- 65% fake news, 35% real news
- **Insight**: Political topics skew toward fake news

**Topic 3: Health**
- Keywords: health, medical, doctor, study
- 40% fake news, 60% real news
- **Insight**: Health news more reliable

**Topic 7: Celebrity**
- Keywords: actor, movie, celebrity, star
- 80% fake news, 20% real news
- **Insight**: Entertainment gossip heavily fake

---

## 📚 Technical Details

### Algorithm: Latent Dirichlet Allocation (LDA)
- Unsupervised probabilistic topic model
- Each document is a mixture of topics
- Each topic is a mixture of words

### Why LDA?
- ✅ Interpretable results (word distributions)
- ✅ Works well for text data
- ✅ Widely used and validated
- ✅ Scikit-learn implementation is robust

### Alternative: NMF
Non-negative Matrix Factorization:
- Often produces more distinct topics
- Faster than LDA
- Try both and compare!

---

## 🛠️ Troubleshooting

### Topics Don't Make Sense?
- Try different number of topics
- Adjust min_df/max_df
- Check preprocessing quality

### Too Slow?
- Reduce MAX_FEATURES
- Lower max_iter
- Use data sample for testing

### Memory Issues?
- Reduce MAX_FEATURES
- Process in batches
- Use sparse matrices (already default)

---

## 📖 Files Created

```
notebooks/
├── topic_analysis.ipynb           ← Main notebook
└── README_TOPIC_ANALYSIS.md      ← Usage guide

data/processed/
└── dataset_with_topics.csv       ← Output data

results/models/
└── topic_keywords.pkl            ← Topic keywords

TOPIC_ANALYSIS_SUMMARY.md         ← This file
```

---

## ✨ Benefits of Topic Analysis

### For This Project:

1. **Feature Engineering**
   - Topic probabilities as features
   - Better performance than bag-of-words

2. **Interpretability**
   - Understand what fake news is about
   - Identify problematic themes

3. **Data Understanding**
   - Discover hidden patterns
   - Group similar articles

4. **Model Improvement**
   - Combine with other features
   - Topic-specific models

---

## 🎯 Success Criteria

### Good Results:
✅ Topics have coherent, distinct keywords
✅ Sample articles make sense within topics
✅ Clear fake/real patterns emerge
✅ Topic similarity shows logical relationships

### Need Improvement:
❌ Topics overlap significantly
❌ Keywords are too generic
❌ No clear fake/real distinction
❌ All articles in 1-2 topics

---

## 📊 Expected Outputs

When you run the notebook, you'll see:

1. **Console Output**
   - Progress messages
   - Statistics
   - Topic keywords
   - Sample article titles

2. **Visualizations**
   - 6-8 charts and plots
   - Word clouds
   - Distribution charts
   - Heatmaps

3. **Saved Files**
   - Enhanced CSV with topics
   - Pickle file with keywords

---

## 🔬 Research Applications

This analysis enables:
- **Content Analysis**: What themes appear in fake news?
- **Comparative Studies**: How do fake and real news differ by topic?
- **Temporal Analysis**: How do topics change over time? (if dates available)
- **Classification**: Use topics as features for ML models

---

## 📝 Citation

If using this in research:
```
Latent Dirichlet Allocation
Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).
Journal of Machine Learning Research, 3, 993-1022.
```

---

## ✅ Summary

**Created**: Comprehensive topic modeling notebook with:
- ✅ 32+ cells of analysis code
- ✅ Complete documentation
- ✅ Multiple visualizations
- ✅ Export functionality
- ✅ Usage guide

**Ready to**: 
- ✅ Discover topics in your data
- ✅ Analyze fake vs real patterns
- ✅ Generate features for ML models
- ✅ Understand your dataset deeply

**Time to complete**: 
- ✅ Setup: 2 minutes
- ✅ Execution: 5-30 minutes (depending on data size)
- ✅ Analysis: As long as you want!

---

## 🚀 Get Started Now!

```bash
conda activate fake-news-detection
jupyter notebook notebooks/topic_analysis.ipynb
```

Happy topic modeling! 🎉


