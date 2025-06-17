# pyright: reportMissingImports=false
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .text_cleaner import clean_text


# Clean text function directly included
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@w+|\#','', text)  # remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Load the model and tokenizer
def load_model_tokenizer():
    model = tf.keras.models.load_model("models/model.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Predict sentiment of input text
def predict_text_sentiment(model, tokenizer, text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    label = "Positive" if pred >= 0.5 else "Negative"
    confidence = float(pred if pred >= 0.5 else 1 - pred)
    return label, confidence
