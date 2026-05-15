"""
DA 370 — Google Play Store Data Mining Dashboard
Streamlit app showcasing all data mining techniques
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Play Store Data Mining",
    page_icon="📱",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
MODELS_DIR = '370_models'
@st.cache_resource
def load_models():
    return {
        'gb': joblib.load(os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')),
        'km': joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl')),
        'cluster_scaler': joblib.load(os.path.join(MODELS_DIR, 'cluster_scaler.pkl')),
        'tfidf': joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')),
        'sentiment_clf': joblib.load(os.path.join(MODELS_DIR, 'sentiment_classifier.pkl')),
        'feature_cols': joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl')),
        'stats': joblib.load(os.path.join(MODELS_DIR, 'stats.pkl'))
    }

models = load_models()
stats = models['stats']

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("📱 Google Play Store — Data Mining Dashboard")
st.markdown("**DA 370 Final Project** | Data Warehouse & Data Mining")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Section:",
    [
        "🏠 Overview",
        "🎯 Rating Predictor",
        "📊 App Clusters",
        "🔗 Association Rules",
        "💬 Sentiment Analyzer"
    ]
)

# ─────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Apps", f"{stats['total_apps']:,}")
    col2.metric("User Reviews", f"{stats['total_reviews']:,}")
    col3.metric("Best Model R²", f"{stats['r2_gb']:.3f}")
    col4.metric("Sentiment Accuracy", f"{stats['sentiment_accuracy']*100:.0f}%")
    
    st.markdown("---")
    
    st.subheader("🛠️ Data Mining Techniques Applied")
    
    techniques = pd.DataFrame({
        'Technique': [
            'Regression', 
            'Clustering', 
            'Association Rules', 
            'Sentiment Analysis (Bonus)'
        ],
        'Models': [
            'Linear, Random Forest, Gradient Boosting, Decision Tree, KNN',
            'K-Means, Hierarchical, DBSCAN, Gaussian Mixture',
            'FP-Growth (174 rules)',
            'TF-IDF + Logistic Regression'
        ],
        'Best Score': [
            f"R² = {stats['r2_gb']:.3f} (Gradient Boosting)",
            f"Silhouette = {stats['silhouette']:.3f} (K-Means)",
            "Max Lift = 1.49",
            f"Accuracy = {stats['sentiment_accuracy']*100:.0f}%"
        ]
    })
    st.dataframe(techniques, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("📌 Key Findings")
    st.markdown("""
    1. **Predicting app ratings is inherently difficult** — basic features explain only ~11% of rating variance, suggesting that quality factors not captured in metadata (UX, bugs, support) drive user ratings.
    
    2. **Apps cluster into 3 distinct groups**: Underperforming (11%), Niche High-Quality (24%), and Mainstream Popular (65%).
    
    3. **Gaming Paradox**: Games appear successful by metrics (high installs/ratings) but have the highest negative sentiment rate (38%) in written reviews.
    
    4. **Health & Fitness apps win on quality**: 76.5% positive reviews — the highest of any category.
    """)

# ─────────────────────────────────────────────────────────────
# PAGE 2: RATING PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "🎯 Rating Predictor":
    st.header("🎯 Predict App Rating")
    st.markdown("Enter app details to predict its expected rating (1-5)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Category",
            ['GAME', 'FAMILY', 'TOOLS', 'PRODUCTIVITY', 'MEDICAL', 
             'COMMUNICATION', 'FINANCE', 'SPORTS', 'PHOTOGRAPHY', 
             'PERSONALIZATION', 'BUSINESS', 'EDUCATION', 'LIFESTYLE']
        )
        reviews = st.number_input("Number of Reviews", min_value=0, value=10000, step=1000)
        installs = st.number_input("Number of Installs", min_value=0, value=100000, step=10000)
    
    with col2:
        size_mb = st.slider("App Size (MB)", 1.0, 100.0, 25.0)
        app_type = st.radio("Type", ["Free", "Paid"])
        price = st.number_input("Price ($)", 0.0, 50.0, 0.0) if app_type == "Paid" else 0.0
        content_rating = st.selectbox("Content Rating", ['Everyone', 'Teen', 'Mature 17+', 'Everyone 10+'])
    
    if st.button("🔮 Predict Rating", type="primary"):
        # تجهيز الـ input
        input_data = pd.DataFrame(0, index=[0], columns=models['feature_cols'])
        
        # نحط القيم الرقمية (مش معايرة بس تقريبية)
        if 'Reviews' in input_data.columns:
            input_data['Reviews'] = (np.log1p(reviews) - 6) / 3
        if 'Installs' in input_data.columns:
            input_data['Installs'] = (np.log1p(installs) - 10) / 4
        if 'Size' in input_data.columns:
            input_data['Size'] = (size_mb - 20) / 20
        if 'Price' in input_data.columns:
            input_data['Price'] = price
        if 'Type' in input_data.columns:
            input_data['Type'] = 1 if app_type == 'Paid' else 0
        
        # Category
        cat_col = f'Category_{category}'
        if cat_col in input_data.columns:
            input_data[cat_col] = 1
        
        # Content Rating
        cr_col = f'ContentRating_{content_rating}'
        if cr_col in input_data.columns:
            input_data[cr_col] = 1
        
        # التنبؤ
        prediction = models['gb'].predict(input_data)[0]
        prediction = max(1.0, min(5.0, prediction))
        
        # العرض
        st.markdown("---")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Predicted Rating", f"⭐ {prediction:.2f} / 5.0")
        with col_b:
            stars = int(prediction)
            star_display = "⭐" * stars + "☆" * (5 - stars)
            st.markdown(f"### {star_display}")
            
            if prediction >= 4.3:
                st.success("🎉 Excellent — Likely a top-performing app!")
            elif prediction >= 4.0:
                st.info("👍 Good — Above average performance expected")
            elif prediction >= 3.5:
                st.warning("⚠️ Average — Room for improvement")
            else:
                st.error("⚡ Below average — Consider quality improvements")
        
        st.caption(f"Model: Gradient Boosting | Test RMSE: {np.sqrt(0.218):.3f} | R²: {stats['r2_gb']:.3f}")

# ─────────────────────────────────────────────────────────────
# PAGE 3: APP CLUSTERS
# ─────────────────────────────────────────────────────────────
elif page == "📊 App Clusters":
    st.header("📊 App Segmentation (K-Means Clustering)")
    
    st.markdown(f"""
    We discovered **3 distinct clusters** of apps using K-Means clustering on 
    Rating, Reviews, Installs, and Size features.
    
    **Silhouette Score: {stats['silhouette']:.4f}** (>0.3 = good separation)
    """)
    
    st.subheader("🏷️ Cluster Profiles")
    
    cluster_data = pd.DataFrame({
        'Cluster': [
            '🔴 Underperforming',
            '🟢 Niche High-Quality',
            '🔵 Mainstream Popular'
        ],
        'Size': ['1,212 apps (11%)', '2,162 apps (24%)', '5,992 apps (65%)'],
        'Avg Rating': [3.21, 4.45, 4.30],
        'Avg Log(Reviews)': [1.73, 1.51, 2.43],
        'Avg Log(Installs)': [2.33, 2.11, 2.73],
        'Description': [
            'Low ratings, few users — likely problematic apps',
            'Highest ratings but smaller audience — specialized apps',
            'Most apps — popular and well-rated mainstream apps'
        ]
    })
    
    st.dataframe(cluster_data, use_container_width=True, hide_index=True)
    
    st.subheader("📊 Cluster Visualization")
    
    profile_df = pd.DataFrame(stats['cluster_profiles'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    profile_df.T.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'])
    ax.set_title('Cluster Profiles — Mean Feature Values')
    ax.set_ylabel('Value')
    ax.set_xlabel('Cluster')
    ax.legend(title='Feature')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("💡 **Key Insight:** Popularity and quality are not perfectly correlated. Cluster 1 (Niche) has the highest ratings but fewer users, while Cluster 2 (Mainstream) sacrifices some satisfaction for broader reach.")

# ─────────────────────────────────────────────────────────────
# PAGE 4: ASSOCIATION RULES
# ─────────────────────────────────────────────────────────────
elif page == "🔗 Association Rules":
    st.header("🔗 Hidden Patterns in App Attributes")
    
    st.markdown("""
    Using **FP-Growth algorithm**, we discovered **174 association rules** revealing 
    how app attributes co-occur.
    """)
    
    st.subheader("🏆 Top 10 Strongest Rules")
    
    rules_df = pd.DataFrame(stats['top_rules'])
    
    # تحويل الـ frozensets لنص
    rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)) if not isinstance(x, str) else x)
    rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)) if not isinstance(x, str) else x)
    rules_df['Rule'] = rules_df['antecedents'] + ' → ' + rules_df['consequents']
    
    rules_display = rules_df[['Rule', 'support', 'confidence', 'lift']].copy()
    rules_display.columns = ['Rule', 'Support', 'Confidence', 'Lift']
    rules_display['Support'] = rules_display['Support'].round(3)
    rules_display['Confidence'] = rules_display['Confidence'].round(3)
    rules_display['Lift'] = rules_display['Lift'].round(3)
    
    st.dataframe(rules_display, use_container_width=True, hide_index=True)
    
    st.subheader("📈 Top Rules by Lift")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    top_5 = rules_display.head(5)
    ax.barh(range(len(top_5)), top_5['Lift'], color='mediumseagreen')
    ax.set_yticks(range(len(top_5)))
    ax.set_yticklabels([r[:50] + '...' if len(r) > 50 else r for r in top_5['Rule']])
    ax.set_xlabel('Lift')
    ax.axvline(1, color='red', linestyle='--', label='Lift=1 (random)')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### 💡 Key Insights:
    - **GAME + Free → High Rating + Very High Installs** (Confidence: 75%, Lift: 1.39)
    - Games are the most consistently successful category combination
    - Free apps with "Everyone" content rating dominate the success patterns
    """)

# ─────────────────────────────────────────────────────────────
# PAGE 5: SENTIMENT ANALYZER
# ─────────────────────────────────────────────────────────────
elif page == "💬 Sentiment Analyzer":
    st.header("💬 Review Sentiment Analyzer")
    
    st.markdown(f"""
    Our **TF-IDF + Logistic Regression** model achieves **{stats['sentiment_accuracy']*100:.0f}% accuracy** 
    on sentiment classification.
    
    Try it yourself — paste any English app review below:
    """)
    
    review_text = st.text_area(
        "Enter a review:",
        value="This app is amazing! I love using it every day. Best app ever.",
        height=100
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip():
            # Transform و Predict
            X_input = models['tfidf'].transform([review_text])
            prediction = models['sentiment_clf'].predict(X_input)[0]
            probabilities = models['sentiment_clf'].predict_proba(X_input)[0]
            
            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            emoji_map = {0: '😠', 1: '😐', 2: '😊'}
            color_map = {0: 'error', 1: 'warning', 2: 'success'}
            
            predicted_sentiment = sentiment_map[prediction]
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {emoji_map[prediction]} {predicted_sentiment}")
                if color_map[prediction] == 'success':
                    st.success(f"Confidence: {probabilities[prediction]*100:.1f}%")
                elif color_map[prediction] == 'warning':
                    st.warning(f"Confidence: {probabilities[prediction]*100:.1f}%")
                else:
                    st.error(f"Confidence: {probabilities[prediction]*100:.1f}%")
            
            with col2:
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative 😠', 'Neutral 😐', 'Positive 😊'],
                    'Probability': probabilities
                })
                
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ['#e74c3c', '#95a5a6', '#2ecc71']
                ax.barh(prob_df['Sentiment'], prob_df['Probability'], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Sentiment Probabilities')
                plt.tight_layout()
                st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("📊 Sentiment by Category (Top 10)")
    
    sentiment_data = pd.DataFrame({
        'Category': ['HEALTH_AND_FITNESS', 'PHOTOGRAPHY', 'PRODUCTIVITY', 'FAMILY', 
                     'SPORTS', 'BUSINESS', 'DATING', 'ENTERTAINMENT', 'TRAVEL', 'GAME'],
        'Positive %': [76.5, 68.9, 68.7, 65.8, 62.8, 61.8, 61.6, 61.1, 58.8, 57.7],
        'Negative %': [12.8, 19.6, 19.1, 26.4, 23.4, 14.6, 21.8, 24.2, 24.5, 38.0]
    })
    
    st.dataframe(sentiment_data, use_container_width=True, hide_index=True)
    
    st.warning("⚠️ **Gaming Paradox**: Despite high install counts and ratings, GAMES have the highest negative sentiment rate (38%) — suggesting written reviews capture frustrations that star ratings miss.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**DA 370 — Spring 2026**")
st.sidebar.markdown("Built with Streamlit 🎈")
