# Sentiment Analysis Tool

## Project Overview
This project is developed as part of the Syntecxhub AI Internship.

The Sentiment Analysis Tool is a machine learning application that analyzes text reviews and classifies them as Positive or Negative.

It uses Natural Language Processing (NLP) techniques and machine learning algorithms to understand the sentiment expressed in customer reviews.

## Project Objectives
- Analyze text reviews automatically.
- Clean and preprocess textual data.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple machine learning models.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Predict sentiment for new, unseen reviews.

## Dataset
**Dataset:** UCI Sentiment Labelled Sentences

- Total original reviews: 3,000
- Amazon reviews: 1,000
- IMDb reviews: 1,000
- Yelp reviews: 1,000
- Positive sentiment: 1
- Negative sentiment: 0

After removing duplicate records, 2,983 reviews remained.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Workflow
1. Data loading and exploration
2. Missing value and duplicate analysis
3. Text cleaning and preprocessing
4. Train-test splitting
5. TF-IDF feature extraction
6. Model training
7. Model evaluation
8. Best model selection
9. Model saving and prediction

## Machine Learning Models
Three machine learning algorithms were trained and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

**Best Performing Model:** Linear SVM

**Test Accuracy:** Approximately 83%

The dataset was divided into training and testing sets using an 80:20 split.

## Project Features
- Text sentiment classification
- Positive and Negative predictions
- TF-IDF-based text representation
- Machine learning model comparison
- Streamlit web application
- Command-line interface (CLI)
- Prediction on new reviews

## Project Structure

```text
Syntecxhub_Sentiment_Analysis/
│
├── data/
│   └── sentiment_dataset.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── 01_Sentiment_Analysis_EDA.ipynb
│
│   
├── app.py
├── cli.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Navigate to the project directory:

```bash
cd Syntecxhub_Sentiment_Analysis
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

## Run the CLI Application

```bash
python cli.py
```

Enter a review when prompted to receive its predicted sentiment.

Type `exit` to close the CLI application.

## Model Evaluation

The Linear SVM model achieved approximately 83% accuracy on the held-out test dataset.

Evaluation included:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Learning Outcomes
- Practical understanding of NLP.
- Text preprocessing and feature extraction.
- Machine learning model training and comparison.
- Model evaluation and performance analysis.
- Building interactive applications using Streamlit.
- Implementing command-line prediction systems.

## Internship
**Organization:** Syntecxhub  
**Project:** Sentiment Analysis Tool  
**Domain:** Artificial Intelligence / Machine Learning

---

Developed as part of the Syntecxhub AI Internship.