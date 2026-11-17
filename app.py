import streamlit as st
import joblib
import re
import string

# Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="💬",
    layout="centered"
)

# Load Saved Model
@st.cache_resource
def load_model():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_model()

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Sentiment Prediction
def predict_sentiment(review):
    cleaned_review = clean_text(review)
    review_vector = tfidf.transform([cleaned_review])
    prediction = model.predict(review_vector)[0]

    return "Positive" if prediction == 1 else "Negative"

# Application Interface
st.title("💬 Sentiment Analysis Tool")

st.write(
    "Analyze customer reviews using Machine Learning "
    "and Natural Language Processing."
)

st.markdown("---")

# User Input
review = st.text_area(
    "Enter your review:",
    placeholder="Example: This product is amazing and works perfectly!",
    height=150
)

# Analyze Button
if st.button("Analyze Sentiment", use_container_width=True):

    if review.strip():

        result = predict_sentiment(review)

        st.markdown("### Analysis Result")

        if result == "Positive":
            st.success("😊 Positive Sentiment")
            st.write("The review expresses a positive opinion.")

        else:
            st.error("😞 Negative Sentiment")
            st.write("The review expresses a negative opinion.")

    else:
        st.warning("Please enter a review first.")

st.markdown("---")

st.caption("Developed as an AI & Machine Learning Internship Project.")