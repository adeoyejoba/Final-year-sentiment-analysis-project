import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore[import]
from .text_cleaner import clean_text  # ✅ Make sure this exists

def load_model_tokenizer():
    model = tf.keras.models.load_model("models/model.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_text_sentiment(model, tokenizer, text):
    try:
        print("Raw input:", text)

        # Optional: Override for clearly negative expressions
        if "hate" in text.lower() and "love" not in text.lower():
            print("⚠️ Strong negative phrase detected, overriding model prediction.")
            return "Negative", 0.95

        # Clean and tokenize input
        cleaned = clean_text(text)
        print("Cleaned text:", cleaned)

        seq = tokenizer.texts_to_sequences([cleaned])
        print("Tokenized sequence:", seq)

        if not seq or not seq[0]:
            raise ValueError("Input text could not be tokenized.")

        padded = pad_sequences(seq, maxlen=200)
        print("Padded sequence shape:", padded.shape)

        pred = model.predict(padded)[0][0]
        print("Raw model prediction score:", pred)

        # Adjusted threshold logic
        if pred >= 0.6:
            label = "Positive"
        elif pred <= 0.4:
            label = "Negative"
        else:
            label = "Neutral"

        confidence = float(max(pred, 1 - pred))
        print("Label:", label, "| Confidence:", confidence)

        return label, confidence

    except Exception as e:
        print("🔥 predict_text_sentiment() failed:", e)
        raise
