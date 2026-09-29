"""
BBC News Text Classification - TF-IDF + Traditional Machine Learning
Ready-to-run code with automatic dataset download

Author: Hạnh (AI Learning Hub)
Dataset: BBC News (2225 articles, 5 categories)
- Business, Entertainment, Politics, Sport, Tech
- Train: 1557 samples | Val: 334 samples | Test: 334 samples
"""

import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier

# XGBoost (auto-install if needed)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("📦 Installing XGBoost...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost installed")


# ============================================================================
# DATASET DOWNLOAD
# ============================================================================

def download_bbc_news():
    """Download BBC News dataset"""
    print("="*70)
    print("📥 DOWNLOADING BBC NEWS DATASET")
    print("="*70)
    
    base_url = 'https://raw.githubusercontent.com/datasets/bbc-news/main/data/'
    
    try:
        train_df = pd.read_csv('https://ltsach.github.io/AILearningHub/datasets/bbcnews/data/train.csv')
        val_df = pd.read_csv('https://ltsach.github.io/AILearningHub/datasets/bbcnews/data/val.csv')
        test_df = pd.read_csv('https://ltsach.github.io/AILearningHub/datasets/bbcnews/data/test.csv')
        
        print(f"✓ Train: {len(train_df):,} samples")
        print(f"✓ Val: {len(val_df):,} samples")
        print(f"✓ Test: {len(test_df):,} samples")
        print(f"✓ Categories: {sorted(train_df['category'].unique().tolist())}")
        print()
        
        return train_df, val_df, test_df
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return None, None, None


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_tfidf_features(train_texts, test_texts):
    """Extract TF-IDF features"""
    print("="*70)
    print("🔢 TF-IDF FEATURE EXTRACTION")
    print("="*70)
    
    start = time.time()
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    elapsed = time.time() - start
    
    print(f"✓ Vocabulary: {len(vectorizer.get_feature_names_out()):,} features")
    print(f"✓ Train shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")
    print(f"✓ Time: {elapsed:.2f}s")
    print()
    
    return X_train, X_test, vectorizer


# ============================================================================
# CLASSIFIERS
# ============================================================================

def train_classifier(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier"""
    from sklearn.metrics import precision_recall_fscore_support
    
    print("="*70)
    print(f"{name.upper()}")
    print("="*70)
    
    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predict
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"⏱️  Training: {train_time:.2f}s")
    print(f"⏱️  Inference: {inference_time:.3f}s ({inference_time/len(y_test)*1000:.2f}ms/sample)")
    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"📊 Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"📊 F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print()
    
    return {
        'name': name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'inference_time': inference_time,
        'inference_speed': len(y_test) / inference_time if inference_time > 0 else 0,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all experiments"""
    print()
    print("="*70)
    print("🚀 BBC NEWS CLASSIFICATION - BY HẠNH")
    print("="*70)
    print()
    
    # Download
    train_df, val_df, test_df = download_bbc_news()
    if train_df is None:
        return
    
    # Combine train + val
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    
    # Extract features
    X_train, X_test, vectorizer = extract_tfidf_features(
        train_full['text'], test_df['text']
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_full['category'])
    y_test_encoded = label_encoder.transform(test_df['category'])
    
    y_train = train_full['category']
    y_test = test_df['category']
    
    # Train classifiers
    classifiers = [
        ('Naive Bayes', MultinomialNB()),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=20, random_state=42)),
        ('SVM', LinearSVC(max_iter=2000, random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
    ]
    
    if XGBOOST_AVAILABLE:
        classifiers.append(('XGBoost', XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0), True))
    
    results = []
    for item in classifiers:
        if len(item) == 3:
            name, model, use_encoded = item
            result = train_classifier(name, model, X_train, y_train_encoded, X_test, y_test_encoded)
            result['y_pred'] = label_encoder.inverse_transform(result['y_pred'])
        else:
            name, model = item
            result = train_classifier(name, model, X_train, y_train, X_test, y_test)
        results.append(result)
        print()
    
    print("="*70)
    print("✅ EXPERIMENT COMPLETED BY HẠNH")
    print("="*70)

if __name__ == '__main__':
    main()