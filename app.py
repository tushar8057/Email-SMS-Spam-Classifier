import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to clean and preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Remove special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:  # Stemming
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
model_path = os.path.join(os.getcwd(), 'model.pkl')

tfidf = pickle.load(open(vectorizer_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message:")

if st.button('Predict'):

    # 1. Preprocess text
    transformed_sms = transform_text(input_sms)
    
    # 2. Convert to vector
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")
