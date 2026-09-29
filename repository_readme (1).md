# 📰 BBC News Text Classification & ML Pipeline Benchmark

Welcome to the **BBC News Text Classification** project maintained by **Hạnh**. This repository provides an end-to-end Machine Learning workspace for classifying news articles into five distinct categories: **Business**, **Entertainment**, **Politics**, **Sport**, and **Tech**.

---

## 📌 Project Overview

- **Author:** Hạnh
- **Dataset:** BBC News Dataset (2,225 news articles).
- **Features:** TF-IDF Vectorization, Bag of Words (BoW), Chi-Squared Selection, PCA.
- **Models:** Naive Bayes, Logistic Regression, Linear SVC, Random Forest, AdaBoost, Gradient Boosting, MLP Classifier, XGBoost, and Voting Ensemble.
- **Pipeline Benchmark:** Automated grid benchmark testing **240 different combinations** of feature extractors, reducers, and classifiers.

---

## 🛠️ Installation & Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn plotly xgboost
```

### 3. Run Models & Generate Interactive Report
```bash
python bbc_news_tfidf_ml.py
```

### 4. Run 240 Pipeline Benchmark
```bash
python bbc_pipeline_comparison.py
```

---

## 📊 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Voting Ensemble** | **98.20%** | **0.9822** | **0.9820** | **0.9820** | 4.20s |
| **Logistic Regression** | 97.60% | 0.9762 | 0.9760 | 0.9759 | 0.42s |
| **MLP Classifier** | 97.30% | 0.9735 | 0.9730 | 0.9731 | 3.80s |
| **Linear SVC** | 97.00% | 0.9701 | 0.9700 | 0.9698 | 0.18s |
| **Multinomial Naive Bayes** | 96.10% | 0.9620 | 0.9610 | 0.9608 | **0.05s** |

---

## 🌐 How to Enable GitHub Pages (Web View Interactive)

1. Push your code (including `index.html`) to your GitHub Repository.
2. Go to **Settings** -> **Pages** in your GitHub repository menu.
3. Under **Build and deployment** -> **Source**, select `Deploy from a branch`.
4. Choose `main` branch and `/ (root)` folder, then click **Save**.
5. Wait ~1 minute, your interactive Web View dashboard will be live!

---
*Created with ❤️ by **Hạnh**.*