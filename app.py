import streamlit as st
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Auto download stopwords if not found
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words_set = set(stopwords.words('english'))

# ---------------------------
# Load Model & Tokenizer
# ---------------------------
model = tf.keras.models.load_model(r"C:/Users/elmnshawy/Downloads/lstm_model.h5")
with open(r"C:/Users/elmnshawy/Downloads/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Label mapping
label_map_rev = {0: "Negative", 2: "Neutral", 1: "Positive"}

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'https?://\S+|www\.\S+', ' ', sentence)
    sentence = re.sub(r'[^\x00-\x7F]', ' ', sentence)
    sentence = re.sub(r'\d+', ' ', sentence)
    sentence = re.sub(r'[^a-z\s]', ' ', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence).strip()

    contractions_map = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am"
    }
    for c_pattern, c_replacement in contractions_map.items():
        sentence = re.sub(c_pattern, c_replacement, sentence)

    sentence = re.sub(r'\bunk\b', '', sentence)
    words = [word for word in sentence.split() if word not in stop_words_set]
    return ' '.join(words)

# ---------------------------
# Prediction Function
# ---------------------------
def predict_sentiment(text):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)
    sentiment_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return label_map_rev[sentiment_class], confidence

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1DA1F2;'>🐦 Twitter Sentiment Analysis (BiLSTM)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Analyze the sentiment of tweets - Positive, Neutral, or Negative</p>", unsafe_allow_html=True)

tweet = st.text_area("✏️ Enter your tweet here:")

if st.button("🔍 Analyze Sentiment"):
    if tweet.strip():
        sentiment, confidence = predict_sentiment(tweet)

        # Color map for results
        color_map = {
            "Positive": "#4CAF50",
            "Neutral": "#2196F3",
            "Negative": "#F44336"
        }

        st.markdown(f"<h3 style='color:{color_map[sentiment]}; text-align:center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        st.progress(float(confidence))

        st.write(f"Confidence: **{confidence*100:.2f}%**")
    else:
        st.warning("⚠️ Please enter a tweet first.")
