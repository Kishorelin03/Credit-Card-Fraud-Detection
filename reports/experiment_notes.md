# 🧪 Experiment Notes

- **SMOTE**: Applied to balance class distribution (0 vs 1)
- **Model**: Random Forest (faster with 25 estimators on 10K samples)
- **Sampling**: Used a subset of resampled data to reduce training time
- **Evaluation**: Focused on ROC AUC, Precision-Recall Curve, Confusion Matrix

## ✅ Observations
- V14, V10, V12 were consistently the top predictors
- SMOTE improved recall significantly
- Precision dropped slightly due to oversampling

## 📌 To Try Next
- Cost-sensitive learning
- XGBoost with stratified sampling
- Isolation Forest (unsupervised)