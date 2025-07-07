# 💳 Credit Card Fraud Detection App

A real-time machine learning app to detect fraudulent credit card transactions, built using Python, Scikit-learn, and Streamlit.

## 🚀 Live App

👉 [Click here to try the app](https://credit-card-fraud-app-hyadqbrtc42npe3zfcd2vo.streamlit.app)

## 📷 Screenshot

![App Screenshot](https://raw.githubusercontent.com/Dineshyalamaddi/credit-card-fraud-app/main/screenshot.png)

## 🧠 Machine Learning

- ✅ Model: Random Forest Classifier
- 📊 Dataset: Kaggle's Credit Card Fraud Detection dataset
- ⚙️ Real-time predictions via Streamlit
- 📁 Model saved using `joblib`

## 🗂 Project Files

- `Train_Fraud_Model.ipynb`: Jupyter notebook to train and save the model
- `fraud_model.pkl`: Trained Random Forest model
- `fraud_app.py`: Streamlit frontend for prediction
- `requirements.txt`: All required libraries

## 🛠 How to Run Locally

```bash
git clone https://github.com/Dineshyalamaddi/credit-card-fraud-app.git
cd credit-card-fraud-app
pip install -r requirements.txt
streamlit run fraud_app.py
```
