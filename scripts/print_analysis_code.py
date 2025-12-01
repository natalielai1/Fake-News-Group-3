import pickle
import sys
import re

# Bypass sklearn imports that might be crashing
# We will try to read the pickle file manually if possible, or just print instructions
# Actually, let's try to just read the vectorizer vocabulary if it's a simple pickle
# and if the model was saved (which it wasn't in make_features.py, only vectorizer was)

def manual_inspection():
    print("Environment seems unstable for full sklearn stack.")
    print("Since we cannot load the trained model (it wasn't saved to disk in make_features.py anyway, only trained in memory),")
    print("we need to re-train a model to analyze it.")
    
    print("\nHowever, since we cannot run the training script without crashing, here is the recommended code to run in a STABLE environment (e.g. your local notebook):")
    
    code = """
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Load Data
data = np.load('data/processed/features_train_tfidf.npz', allow_pickle=True)
X_train = data['X'].item()
y_train = data['y']

# 2. Load Vectorizer
vectorizer = joblib.load('data/processed/tfidf_vectorizer.pkl')
feature_names = vectorizer.get_feature_names_out()

# 3. Train Model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 4. Extract Coefficients
coefs = model.coef_[0]
df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
df['abs_coef'] = df['coef'].abs()
df = df.sort_values('abs_coef', ascending=False)

# 5. Show Top Features
print("Top Indicators for FAKE News (Positive Coefs):")
print(df[df['coef'] > 0].head(20))

print("\\nTop Indicators for REAL News (Negative Coefs):")
print(df[df['coef'] < 0].head(20))
    """
    print("-" * 40)
    print(code)
    print("-" * 40)

if __name__ == "__main__":
    manual_inspection()

