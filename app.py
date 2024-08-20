import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and TF-IDF vectorizer
model_filename = r'C:\Users\subra\streamlit_app\finalized_model.sav'
tfidf_filename = r'C:\Users\subra\streamlit_app\tfidf.pkl'
loaded_model = pickle.load(open(model_filename, 'rb'))
tfidf = pickle.load(open(tfidf_filename, 'rb'))

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Convert words to lowercase
    words = [word.lower() for word in words]

    # Remove stopwords
    stopwords_list = stopwords.words('english')
    words = [word for word in words if word not in stopwords_list]

    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]

    # Lemmatization of the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a string
    return ' '.join(words)

def predict_sentiment(review_text):
    # Preprocess the review
    preprocessed_review = preprocess_text(review_text)

    # Transform the preprocessed review using the TF-IDF vectorizer
    new_review_tfidf = tfidf.transform([preprocessed_review])

    # Predict the sentiment
    prediction = loaded_model.predict(new_review_tfidf)

    # Return the prediction (0 for negative, 1 for positive)
    return prediction[0]

# Streamlit app
st.title('Hotel Review Sentiment Analysis')

review_input = st.text_area("Enter a hotel review:")

if st.button('Predict Review'):
    if review_input:
        prediction = predict_sentiment(review_input)
        if prediction == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")
    else:
        st.warning("Please enter a review to analyze.")



