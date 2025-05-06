import streamlit as st
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_rf_model.pkl")

model = load_model()
expected_features = list(model.feature_names_in_)

st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Upload a credit card transaction CSV file and receive fraud predictions using a trained Random Forest model.
""")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # Preprocess if raw Kaggle format
        if 'Amount' in data.columns and 'Time' in data.columns:
            st.info("Preprocessing raw Kaggle dataset...")
            scaler = StandardScaler()
            data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
            data['scaled_time'] = scaler.fit_transform(data[['Time']])
            data.drop(['Amount', 'Time', 'Class'], axis=1, inplace=True, errors='ignore')
            data = data[['scaled_amount', 'scaled_time'] + [col for col in data.columns if col not in ['scaled_amount', 'scaled_time']]]

        # Validate column match
        if not set(expected_features).issubset(data.columns):
            st.error("❌ Uploaded CSV is missing one or more required columns.")
            missing_cols = list(set(expected_features) - set(data.columns))
            st.warning("🧾 Your file is missing the following column(s):")
            st.code("\n".join(missing_cols), language='text')
            st.info("✅ Required columns:")
            st.code(", ".join(expected_features), language='text')
        else:
            # Generate predictions
            st.success("✅ Data is valid. Generating predictions...")
            predictions = model.predict(data[expected_features])
            probs = model.predict_proba(data[expected_features])[:, 1]

            result = data.copy()
            result['Fraud Probability'] = probs
            result['Prediction'] = predictions
            result['Prediction'] = result['Prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

            # Rearranged column order
            cols = [col for col in result.columns if col not in ['Prediction', 'Fraud Probability']]
            result = result[cols + ['Prediction', 'Fraud Probability']]

            # Fraud count
            fraud_count = result['Prediction'].value_counts().get('Fraud', 0)
            st.markdown(f"### 🔍 Fraudulent Transactions Detected: `{fraud_count}`")

            # 2×2 Grid of Visualizations
            st.subheader("📊 Fraud Detection Visualizations")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔸 Fraud vs. Non-Fraud (Pie)")
                fraud_counts = result['Prediction'].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.2f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
                ax1.axis('equal')
                st.pyplot(fig1)

            with col2:
                st.markdown("#### 📈 Fraud Probability Distribution")
                fig2, ax2 = plt.subplots()
                sns.histplot(result['Fraud Probability'], bins=20, kde=True, ax=ax2, color="purple")
                ax2.set_xlabel("Probability")
                ax2.set_ylabel("Count")
                ax2.set_title("Fraud Probability")
                st.pyplot(fig2)

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### 🔍 Top 10 Important Features")
                importances = pd.Series(model.feature_importances_, index=expected_features)
                top_features = importances.sort_values(ascending=False).head(10)
                fig3, ax3 = plt.subplots()
                sns.barplot(x=top_features.values, y=top_features.index, ax=ax3, color='teal')
                ax3.set_xlabel("Importance")
                st.pyplot(fig3)

            with col4:
                st.markdown("#### 📊 Fraud Count (Bar)")
                fig4, ax4 = plt.subplots()
                sns.countplot(x='Prediction', data=result, ax=ax4, palette='pastel')
                ax4.set_ylabel("Transaction Count")
                ax4.set_title("Fraud vs. Not Fraud Count")
                st.pyplot(fig4)

            # Display table with filters
            show_fraud_only = st.checkbox("🔎 Show only Fraud transactions", value=False)
            row_limit = st.selectbox("🔢 How many rows to display?", [10, 50, 100, 'All'])

            display_df = result[result['Prediction'] == 'Fraud'] if show_fraud_only else result
            if row_limit != 'All':
                display_df = display_df.head(int(row_limit))

            st.subheader("📋 Prediction Table")
            st.dataframe(display_df)

            # CSV download
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file:\n\n{e}")
