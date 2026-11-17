import joblib
import re
import string

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Sentiment prediction function
def predict_sentiment(review):
    cleaned_review = clean_text(review)

    review_vector = tfidf.transform([cleaned_review])

    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"


# Main CLI interface
def main():
    print("\n===================================")
    print("     SENTIMENT ANALYSIS TOOL")
    print("===================================")
    print("Enter a review to analyze.")
    print("Type 'exit' to close the application.\n")

    while True:
        review = input("Enter your review: ").strip()

        if review.lower() == "exit":
            print("\nThank you for using Sentiment Analysis Tool!")
            break

        if not review:
            print("Please enter a valid review.\n")
            continue

        result = predict_sentiment(review)

        print("\nPredicted Sentiment:", result)
        print("-----------------------------------")


if __name__ == "__main__":
 main()