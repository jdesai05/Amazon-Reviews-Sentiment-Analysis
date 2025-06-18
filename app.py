# app.py

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Alexa Review Sentiment Analyzer", layout="centered")
st.title("🗣️ Alexa Sentiment Review Analyzer")
st.subheader("Enter a review and find out if it's Positive or Negative")

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][prediction]
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
    return sentiment, round(probability * 100, 2)

with st.form("review_form"):
    user_review = st.text_area("🔍 Paste your Alexa product review here:", height=150)
    submit = st.form_submit_button("Analyze Sentiment")

    if submit:
        if user_review.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            sentiment, confidence = predict_sentiment(user_review)
            st.success(f"**Predicted Sentiment:** {sentiment}")
            st.write(f"**Model Confidence:** {confidence}%")

