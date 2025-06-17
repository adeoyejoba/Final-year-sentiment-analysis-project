import praw
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # So matplotlib works without GUI
import matplotlib.pyplot as plt
from .text_cleaner import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

def init_reddit():
    return praw.Reddit(
        client_id="G-f4vfrqpcjsABMLNdqxmw",
        client_secret="TzYfgjlVgUe0IYE5ev2p1k_k7cA4HA",
        user_agent="sentimentApp by /u/BedroomSubject5844"
    )

def fetch_posts(reddit, keyword, limit=20):
    posts = []
    for submission in reddit.subreddit("all").search(keyword, limit=limit):
        posts.append(submission.title + " " + submission.selftext)
    return posts

def analyze_reddit_sentiment(model, tokenizer, reddit, keyword, limit=20):
    posts = fetch_posts(reddit, keyword, limit)
    cleaned = [clean_text(p) for p in posts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=200)
    preds = model.predict(padded)
    
    labels = ["Positive" if p >= 0.5 else "Negative" for p in preds]
    return list(zip(posts, labels))
