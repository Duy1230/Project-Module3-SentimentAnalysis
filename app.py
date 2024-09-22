from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from preprocess import process_text
import pickle

# Load the models
models = {
    'Decision Tree': pickle.load(open("weights/DecisionTree_model.pkl", "rb")),
    'Random Forest': pickle.load(open("weights/RandomForest_model.pkl", "rb")),
    'Gradient Boosting': pickle.load(open("weights/GradientBoosting_model.pkl", "rb")),
    'AdaBoost': pickle.load(open("weights/AdaBoost_model.pkl", "rb")),
    'XGBoost': pickle.load(open("weights/XGBoost_model.pkl", "rb"))
}

# Function to get available models


def get_available_models():
    return list(models.keys())

# Create a function to predict the sentiment


def predict_sentiment(text, model_name):
    text = process_text(text)
    prediction = models[model_name].predict(text)
    if prediction[0] == 0:
        return "Negative"
    else:
        return "Positive"

# def predict_sentiment_with_confidence(text, model_name):
#     text = process_text(text)
#     prediction = models[model_name].predict_proba([text])
#     confidence = prediction[0][1] if prediction[0][1] > 0.5 else 1 - prediction[0][1]
#     sentiment = "Positive" if prediction[0][1] > 0.5 else "Negative"
#     return sentiment, confidence * 100
