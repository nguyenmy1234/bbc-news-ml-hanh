"""
BBC News Classification - Pipeline Comparison
Compare 240 different ML pipelines for text classification

Author: Hạnh (AI Learning Hub)
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#================================================
# STEP 1: Download Dataset
#================================================

def download_dataset():
    """Download BBC News dataset"""
    print("📥 Downloading BBC News dataset...")
    
    base_url = 'https://ltsach.github.io/AILearningHub/datasets/bbcnews/data/'
    files = ['train.csv', 'val.csv', 'test.csv']
    
    os.makedirs('data', exist_ok=True)
    
    for filename in files:
        filepath = f'data/{filename}'
        if not os.path.exists(filepath):
            import urllib.request
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)
            print(f"   ✅ Downloaded {filename}")
        else:
            print(f"   ⏭️  {filename} already exists")
    
    print("✅ Dataset ready!\n")

def load_data():
    """Load data"""
    print("📂 Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    X_train = train_df['text'].values
    y_train = train_df['category'].values
    X_test = test_df['text'].values
    y_test = test_df['category'].values
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔬 BBC NEWS PIPELINE COMPARISON - BY HẠNH")
    print("="*70 + "\n")
    
    download_dataset()
    X_train, y_train, X_test, y_test = load_data()
    print("Data successfully loaded!")
