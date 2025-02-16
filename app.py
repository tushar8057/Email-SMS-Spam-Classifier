import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import speech_recognition as sr

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Listening... Speak now")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  # Capture audio
            text = recognizer.recognize_google(audio).strip()  # Convert speech to text
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "⚠️ Could not request results, check internet connection."
        except Exception as e:
            return f"🚨 Error: {str(e)}"

# Function to clean and preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]  # Remove special characters
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]  # Apply stemming

    return " ".join(text)

# Load the vectorizer and model
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
model_path = os.path.join(os.getcwd(), 'model.pkl')

tfidf = pickle.load(open(vectorizer_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

# User input text box
input_sms = st.text_area("Enter the message:")

# Variable to store transcribed text
transcribed_text = ""

if st.button("🎙️ Start Recording"):
    transcribed_text = recognize_speech()

    if not transcribed_text:  # If no speech is detected
        st.warning("⚠️ No speech detected. Please try again.")
    else:
        st.success(f"📝 Transcription: {transcribed_text}")

if st.button('Predict'):
    if not input_sms and not transcribed_text:
        st.warning("⚠️ Please enter a message or use voice input.")
    else:
        text_to_predict = transcribed_text if transcribed_text else input_sms  # Use speech text if available
        transformed_sms = transform_text(text_to_predict)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display result
        st.header("🚨 Spam" if result == 1 else "✅ Not Spam")
