# 🐦 Twitter Sentiment Analyzer

Twitter Sentiment Analyzer is an AI-powered system that classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments using a **Bidirectional LSTM** deep learning model.  
The project comes with an interactive **Streamlit** interface for real-time predictions.

---

## 📌 Project Overview
This project aims to provide an end-to-end sentiment analysis pipeline for Twitter data.  
It combines **advanced text preprocessing**, **deep learning architecture**, and **user-friendly deployment** to achieve high accuracy.

---

## ✨ Key Features
- 🧠 **Advanced AI Model**: Bidirectional LSTM with embedding layers
- 🧹 **Smart Preprocessing**: Removes URLs, special characters, stopwords, and expands contractions
- 📊 **Multi-class Classification**: Positive, Negative, and Neutral sentiment detection
- ⚡ **Real-time Predictions**: Interactive web app built with Streamlit
- 📂 **Reusable Model**: Can be integrated into other applications

---

## 📊 Dataset
The model was trained on a labeled Twitter dataset containing:
- **Training Data**: 74,682 tweets
- **Validation Data**: 1,000 tweets
- **Classes**: Positive, Negative, Neutral
- **Format**: CSV with entity, sentiment, and tweet text

---

## 🧠 Model Details
**Architecture:**
- Embedding Layer (128 dimensions)
- Bidirectional LSTM Layers (32, 16 units)
- Dense Layers with Dropout (64, 32 units)
- Softmax Output Layer (3 classes)

**Performance:**
- Accuracy: **91%**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Training: 15 epochs 

---

## 🔧 Text Preprocessing
The preprocessing pipeline includes:
- Lowercasing text
- Removing URLs and HTML tags
- Removing non-alphabetic characters
- Expanding contractions (e.g., can't → cannot)
- Removing stopwords (while keeping negation words)
- Tokenizing and padding sequences

---
## 📂 Project Structure

```plaintext
twitter_sentiment_project/
│
├── dataset/                  # Training and validation datasets
│   ├── twitter_training.csv
│   └── twitter_validation.csv
│
├── models/                   # Saved LSTM model and tokenizer
│   ├── lstm_model.h5
│   └── tokenizer.pkl
│
├── notebooks/                # Jupyter notebooks for model development and experiments
│   └── Twitter_Sentiment_Model.ipynb
│
├── app.py                     # Streamlit application for sentiment prediction
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation




