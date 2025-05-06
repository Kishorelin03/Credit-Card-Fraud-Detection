# 💳 Credit Card Fraud Detection Using Machine Learning

## 🧠 Objective
This project aims to accurately detect fraudulent credit card transactions using a machine learning approach. Due to the highly imbalanced nature of the dataset (frauds make up only 0.17% of all transactions), special attention is given to resampling techniques and evaluation metrics that prioritize recall and precision.

## 📊 Dataset Overview
- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**: 30 (V1 to V28, Time, Amount)
- **Target**: `Class` (0 = Non-Fraud, 1 = Fraud)

## 🛠️ Tools & Technologies Used
- **Language**: Python
- **Libraries**: pandas, NumPy, scikit-learn, imbalanced-learn, seaborn, matplotlib
- **Techniques**:
  - StandardScaler for feature normalization
  - SMOTE for oversampling minority class
  - Random Forest Classifier for fraud detection
  - Model evaluation via Confusion Matrix, ROC AUC, Precision-Recall Curve
  - Feature Importance Analysis

## 📈 Key Steps
1. **Data Preprocessing**
   - Removed missing values
   - Scaled `Amount` and `Time` using StandardScaler
2. **Train-Test Split**
   - Stratified split to preserve class distribution
3. **Class Balancing**
   - Applied SMOTE to oversample fraudulent transactions
4. **Modeling**
   - Trained a Random Forest model on a 10,000-row balanced sample
5. **Evaluation**
   - Achieved high ROC AUC
   - Visualized Precision-Recall tradeoff
   - Identified top fraud-predicting features

## ✅ Results
- **ROC AUC Score**: ~0.98  
- **Top Predictive Features**: V14, V10, V12 (based on feature importance)
- **Precision-Recall Curve** demonstrated the ability to capture the majority of fraud cases with acceptable false positives.

## ⚠️ Limitations
- Anonymized data limits interpretability of features (V1–V28)
- Random Forest is not suitable for real-time scoring without optimization
- No temporal validation (fraud may change over time)

## 🚀 Future Work
- Deploy the model using Streamlit for interactive predictions
- Incorporate cost-sensitive learning
- Experiment with unsupervised anomaly detection (Isolation Forest, One-Class SVM)