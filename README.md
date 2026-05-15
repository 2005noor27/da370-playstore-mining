#  DA 370 — Google Play Store Data Mining

**Course:** DA 370 — Data Warehouse & Data Mining (Spring 2026)  
**Instructor:** Dr. Enas Khashashneh  
**Student:** Noor

---

##  Project Overview

A complete data mining pipeline applied to the Google Play Store dataset (10,841 apps + 64,295 user reviews). This project demonstrates four core data mining techniques and deploys an interactive dashboard built with Streamlit.

---

## Techniques Applied

| Technique | Models Used | Best Result |
|---|---|---|
| **Regression** | Linear, Random Forest, Gradient Boosting, Decision Tree, KNN | R² = 0.1143 (Gradient Boosting) |
| **Clustering** | K-Means, Hierarchical, DBSCAN, Gaussian Mixture | Silhouette = 0.4054 (K-Means, K=3) |
| **Association Rules** | FP-Growth | 174 rules, Max Lift = 1.49 |
| **Sentiment Analysis** ⭐ | TF-IDF + Logistic Regression | 92% Accuracy |

---

##  Key Findings

- **3 distinct app clusters discovered:** Underperforming (11%), Niche High-Quality (24%), Mainstream Popular (65%)
- **Gaming Paradox:** Games have high install counts and ratings but the highest negative sentiment rate (38%) in written reviews
- **Health & Fitness apps** achieve the highest positive sentiment (76.5%)
- **Basic app metadata explains only ~11% of rating variance** — user-facing quality factors matter more

---

## Interactive Dashboard

Built with **Streamlit**, featuring 5 sections:

1. 🏠 Overview — Project statistics
2. 🎯 Rating Predictor — Predict app ratings interactively
3. 📊 App Clusters — Explore the 3 discovered segments
4. 🔗 Association Rules — Browse discovered patterns
5. 💬 Sentiment Analyzer — Analyze any review's sentiment

---

##  Repository Structure
├── app.py                          # Streamlit dashboard
├── predect the rating.ipynb        # Full data mining notebook
├── requirements.txt                # Python dependencies
└── 370_models/                     # Trained model files (.pkl)

---

##  Tech Stack

- **Python 3.14**
- **scikit-learn** — ML models
- **mlxtend** — Association rules (FP-Growth)
- **Streamlit** — Interactive dashboard
- **Pandas, NumPy, Matplotlib, Seaborn**

---

##  Dataset

- [Google Play Store Apps on Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
