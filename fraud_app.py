import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("Upload a CSV file with transactions:", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if 'Time' in data.columns:
            data = data.drop('Time', axis=1)

        if 'Class' in data.columns:
            data = data.drop('Class', axis=1)

        predictions = model.predict(data)
        data['Prediction'] = predictions

        st.success("✅ Predictions complete!")
        st.dataframe(data)

        fraud_count = data['Prediction'].sum()
        total = len(data)
        st.write(f"🔎 Detected **{fraud_count} frauds** out of **{total} transactions**")

        # Add fraud pie chart
        labels = ['Non-Fraud', 'Fraud']
        values = [total - fraud_count, fraud_count]
        colors = ['#4CAF50', '#FF5252']

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Draw pie as a circle
        st.pyplot(fig)

        if fraud_count > 0:
            st.error("⚠️ Fraudulent transactions detected!")
        else:
            st.success("✅ No fraud detected. All clear!")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
