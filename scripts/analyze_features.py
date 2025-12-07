import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_feature_importance():
    print("Loading data and models...")
    
    # Load data
    try:
        train_data = np.load("data/processed/features_train_tfidf.npz")
        X_train = train_data['X'].item() # Sparse matrix
        y_train = train_data['y']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load vectorizer to get feature names
    try:
        vectorizer = joblib.load("data/processed/tfidf_vectorizer.pkl")
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return

    print(f"Loaded {len(feature_names)} features.")

    # Train Logistic Regression for interpretation (easier than Ridge for prob)
    print("Training Logistic Regression for analysis...")
    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # Get coefficients
    # Binary classification: coef_ is shape (1, n_features)
    # Positive coef -> Class 1 (Fake)
    # Negative coef -> Class 0 (Real) -- WAIT, check mapping!
    
    # Check class mapping
    print(f"Classes: {model.classes_}")
    # Usually [0, 1]. If 1 is Fake, then positive coefs predict Fake.
    
    coefs = model.coef_[0]
    
    # Create dataframe
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': coefs,
        'abs_importance': np.abs(coefs)
    })
    
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("TOP PREDICTORS FOR CLASS 1 (FAKE NEWS)")
    print("="*60)
    print(feat_imp.head(20).to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP PREDICTORS FOR CLASS 0 (REAL NEWS)")
    print("="*60)
    # Negative coefficients predict Class 0
    print(feat_imp.tail(20).sort_values('importance', ascending=True).to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Top 15 Fake
    top_fake = feat_imp.head(15)
    # Top 15 Real
    top_real = feat_imp.tail(15).sort_values('importance', ascending=True)
    
    combined = pd.concat([top_fake, top_real])
    
    colors = ['red' if x > 0 else 'blue' for x in combined['importance']]
    
    sns.barplot(x='importance', y='feature', data=combined, palette=colors)
    plt.title('Top Feature Importance (Logistic Regression)\nBlue=Real Predictors, Red=Fake Predictors')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png')
    print("\nSaved plot to results/figures/feature_importance.png")

if __name__ == "__main__":
    analyze_feature_importance()


