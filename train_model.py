import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import praw
from collections import Counter
import matplotlib.pyplot as plt
import os
import pickle
from typing import List

# --- Text Cleaning (No Stopword Removal) ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# --- Load and preprocess dataset ---
def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)
    df = df[['review', 'sentiment']]
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['review'] = df['review'].apply(clean_text)
    return df

# --- Tokenizer and Padding Setup ---
def prepare_tokenizer(texts: List[str], num_words=10000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_padded_sequences(tokenizer, texts: List[str], maxlen=200):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)

# --- Build and compile the model ---
def build_model(vocab_size=10000, embedding_dim=64, max_length=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Train the model ---
def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
    return model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

# --- Initialize Reddit API ---
def init_reddit(client_id: str, client_secret: str, user_agent: str):
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# --- Fetch Reddit posts ---
def fetch_reddit_posts(reddit, keyword: str, limit=20):
    posts = []
    try:
        for submission in reddit.subreddit("all").search(keyword, limit=limit):
            posts.append(submission.title + " " + submission.selftext)
    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
    return posts

# --- Predict sentiment ---
def predict_sentiment(model, tokenizer, texts: List[str], maxlen=200):
    cleaned_texts = [clean_text(text) for text in texts]
    padded = texts_to_padded_sequences(tokenizer, cleaned_texts, maxlen)
    preds = model.predict(padded)

    # FIX: Use .item() to avoid NumPy deprecation warnings
    labels = ["Positive" if p.item() >= 0.5 else "Negative" for p in preds]
    confidences = [p.item() if p.item() >= 0.5 else 1 - p.item() for p in preds]
    return list(zip(texts, labels, confidences))

# --- Analyze Reddit sentiment ---
def analyze_reddit_sentiment(reddit, model, tokenizer, keyword: str, limit=20):
    posts = fetch_reddit_posts(reddit, keyword, limit)
    if not posts:
        print("No posts fetched.")
        return

    results = predict_sentiment(model, tokenizer, posts)

    for i, (post, label, conf) in enumerate(results):
        print(f"\nPost {i+1}: {post[:100]}...")
        print(f"Sentiment: {label} (Confidence: {conf:.2f})")

    count = Counter(label for _, label, _ in results)
    plt.figure(figsize=(5, 5))
    plt.pie(count.values(), labels=count.keys(), autopct="%1.1f%%", startangle=140)
    plt.title(f"Reddit Sentiment on '{keyword}'")
    plt.axis("equal")
    plt.show()

# --- Analyze single text sentiment ---
def analyze_text_sentiment(model, tokenizer, text: str):
    result = predict_sentiment(model, tokenizer, [text])[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result[1]} (Confidence: {result[2]:.2f})")

# --- Paths ---
MODEL_DIR = r"C:\Users\PRECISION 5520\Documents\PY files\Sentiment analysis - School Final yaer project\models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# --- Main ---
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading existing model and tokenizer...")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        print("Training new model...")
        df = load_and_preprocess_data(r"C:\Users\PRECISION 5520\Documents\PY files\Sentiment analysis - School Final yaer project\data\IMDB Dataset.csv")

        tokenizer = prepare_tokenizer(df['review'].tolist())
        X = texts_to_padded_sequences(tokenizer, df['review'].tolist())
        y = df['sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_model()
        model.summary()
        train_model(model, X_train, y_train, X_test, y_test)

        model.save(MODEL_PATH)
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"Model and tokenizer saved in:\n{MODEL_DIR}")

    reddit = init_reddit(
        client_id="D2CdPqs52i0d685fPAT34A",
        client_secret="mJymLTbzXvU7hODGGIGvubmIVjdLCA",
        user_agent="sentimentApp by /u/BedroomSubject5844"
    )

    print("Reddit read-only mode:", reddit.read_only)

    analyze_reddit_sentiment(reddit, model, tokenizer, "artificial intelligence", limit=20)
    analyze_text_sentiment(model, tokenizer, "I really love the advances in AI technology!")

    texts = [
        "I really love the advances in AI technology!",
        "This movie was terrible and boring.",
        "The food was amazing and delicious.",
        "I hate waiting in long lines."
    ]
    results = predict_sentiment(model, tokenizer, texts)
    for text, label, conf in results:
        print(f"Text: {text}\nSentiment: {label} (Confidence: {conf:.2f})\n")
