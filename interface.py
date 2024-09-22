import streamlit as st
from app import predict_sentiment, get_available_models

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

st.title("📊 Sentiment Analysis App 🤖")

st.markdown("""
This app analyzes the sentiment of your text using various machine learning models.
Choose a model and enter your text to get started!
""")

# Model selection
available_models = get_available_models()
selected_model = st.selectbox("Select a model", available_models)

# Text input
text = st.text_area("Enter text to analyze", height=100)

# Analyze button
if st.button("Analyze Sentiment"):
    if text:
        sentiment = predict_sentiment(text, selected_model)

        # Display result
        st.markdown("### Results")
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment} 😊")
        else:
            st.error(f"Sentiment: {sentiment} 😔")

        # Add a confidence score (you'll need to modify your predict_sentiment function to return this)
        # confidence = predict_sentiment_with_confidence(text, selected_model)
        # st.progress(confidence)
        # st.text(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")

# Add some information about the project
st.markdown("""
---
### About this project
This sentiment analysis project uses various machine learning models to predict
the sentiment of input text. The models have been trained on a dataset of labeled
sentiments and can classify text as either positive or negative.

For more information, check out the [project repository](https://github.com/Duy1230/Project-Module3-SentimentAnalysis).
""")
