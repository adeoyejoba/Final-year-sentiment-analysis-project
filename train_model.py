import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import praw
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import pickle
from typing import List

# --- Text Cleaning ---
def clean_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    return text.strip()

# --- Load and preprocess dataset ---
def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)
    df = df[['review', 'sentiment']].dropna()  # Remove rows with missing values
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df.dropna()  # Remove rows where sentiment mapping failed
    df['review'] = df['review'].apply(clean_text)
    df = df[df['review'].str.len() > 0]  # Remove empty reviews
    return df

# --- Tokenizer and Padding ---
def prepare_tokenizer(texts: List[str], num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_padded_sequences(tokenizer, texts: List[str], maxlen=200):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)

# --- Build Model ---
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

# --- Train Model ---
def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    return history

# --- Reddit Integration ---
def init_reddit(client_id: str, client_secret: str, user_agent: str):
    try:
        reddit = praw.Reddit(
            client_id=client_id, 
            client_secret=client_secret, 
            user_agent=user_agent
        )
        reddit.read_only = True  # Fix: This should be an assignment, not just a property access
        # Test the connection
        reddit.user.me()
        return reddit
    except Exception as e:
        print("Reddit init failed:", e)
        return None

def fetch_reddit_posts_with_timestamps(reddit, keyword: str, limit=50):
    """Fetch Reddit posts with timestamps for timeline analysis"""
    posts_data = []
    if reddit is None:
        print("🟡 Using fallback offline posts.")
        # Generate fallback data with realistic timestamps
        now = datetime.utcnow()
        for i in range(min(limit, 20)):
            posts_data.append({
                'text': f"Sample offline comment about {keyword} - post {i+1}",
                'timestamp': now - timedelta(days=i % 7, hours=i % 24, minutes=i % 60)
            })
        return posts_data
    
    try:
        print(f"🔍 Searching Reddit for: '{keyword}'")
        search_results = list(reddit.subreddit("all").search(keyword, limit=limit, sort='new'))
        print(f"📊 Found {len(search_results)} posts")
        
        for submission in search_results:
            post_text = submission.title
            if hasattr(submission, 'selftext') and submission.selftext:
                post_text += " " + submission.selftext
            
            if post_text.strip():
                posts_data.append({
                    'text': post_text.strip(),
                    'timestamp': datetime.utcfromtimestamp(submission.created_utc)
                })
        
        if not posts_data:  # If no posts found, return fallback
            print(f"🟡 No posts found for '{keyword}', using fallback.")
            now = datetime.utcnow()
            for i in range(min(limit, 10)):
                posts_data.append({
                    'text': f"Sample fallback: couldn't fetch live Reddit post for '{keyword}' - post {i+1}",
                    'timestamp': now - timedelta(days=i % 7, hours=i % 24)
                })
            
    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        now = datetime.utcnow()
        for i in range(min(limit, 10)):
            posts_data.append({
                'text': f"Sample fallback: couldn't fetch live Reddit post for '{keyword}' - post {i+1}",
                'timestamp': now - timedelta(days=i % 7, hours=i % 24)
            })
    
    return posts_data

def fetch_reddit_posts(reddit, keyword: str, limit=20):
    """Legacy function for backward compatibility"""
    posts_data = fetch_reddit_posts_with_timestamps(reddit, keyword, limit)
    return [post['text'] for post in posts_data]

# --- Sentiment Prediction ---
def predict_sentiment(model, tokenizer, texts: List[str], maxlen=200):
    if not texts:
        return []
    
    cleaned_texts = [clean_text(text) for text in texts]
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(cleaned_texts) if text.strip()]
    
    if not valid_indices:
        return [(text, "Neutral", 0.5) for text in texts]
    
    valid_texts = [cleaned_texts[i] for i in valid_indices]
    padded = texts_to_padded_sequences(tokenizer, valid_texts, maxlen)
    preds = model.predict(padded, verbose=0)

    results = []
    valid_pred_idx = 0
    
    for i, original_text in enumerate(texts):
        if i in valid_indices:
            score = preds[valid_pred_idx].item()
            valid_pred_idx += 1
            
            if score >= 0.6:
                label = "Positive"
                confidence = score
            elif score <= 0.4:
                label = "Negative"
                confidence = 1 - score
            else:
                label = "Neutral"
                confidence = max(score, 1 - score)
        else:
            label = "Neutral"
            confidence = 0.5
            
        results.append((original_text, label, confidence))

    return results

def plot_sentiment_timeline(analyzed_posts, keyword, save_path=None):
    """Create and display sentiment over time plot"""
    try:
        print(f"📊 Creating sentiment timeline plot for '{keyword}'")
        
        if not analyzed_posts:
            print("❌ No posts to plot")
            return None
        
        # Sort posts by timestamp
        sorted_posts = sorted(analyzed_posts, key=lambda x: x['timestamp'])
        
        # Group posts by time period (hours for recent data, days for older data)
        time_groups = {}
        timestamps = [post['timestamp'] for post in sorted_posts]
        time_range = max(timestamps) - min(timestamps)
        
        # Determine grouping strategy based on time range
        if time_range <= timedelta(hours=24):
            # Group by hour
            time_format = '%Y-%m-%d %H:00'
            display_format = '%H:00'
            time_delta = timedelta(hours=1)
        elif time_range <= timedelta(days=7):
            # Group by day
            time_format = '%Y-%m-%d'
            display_format = '%b %d'
            time_delta = timedelta(days=1)
        else:
            # Group by week
            time_format = '%Y-W%U'
            display_format = 'Week %U'
            time_delta = timedelta(weeks=1)
        
        # Initialize time groups
        current_time = min(timestamps)
        end_time = max(timestamps)
        while current_time <= end_time:
            time_key = current_time.strftime(time_format)
            time_groups[time_key] = {
                'positive': 0, 'neutral': 0, 'negative': 0, 
                'total': 0, 'datetime': current_time
            }
            current_time += time_delta
        
        # Populate time groups
        for post in sorted_posts:
            time_key = post['timestamp'].strftime(time_format)
            if time_key in time_groups:
                sentiment = post['label'].lower()
                if sentiment in ['positive', 'neutral', 'negative']:
                    time_groups[time_key][sentiment] += 1
                    time_groups[time_key]['total'] += 1
        
        # Prepare data for plotting
        dates = []
        positive_counts = []
        neutral_counts = []
        negative_counts = []
        
        for time_key in sorted(time_groups.keys()):
            data = time_groups[time_key]
            dates.append(data['datetime'])
            positive_counts.append(data['positive'])
            neutral_counts.append(data['neutral'])
            negative_counts.append(data['negative'])
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Stacked bar chart
        width = 0.8
        ax1.bar(dates, positive_counts, width, label='Positive', color='#2E8B57', alpha=0.8)
        ax1.bar(dates, neutral_counts, width, bottom=positive_counts, label='Neutral', color='#FFD700', alpha=0.8)
        ax1.bar(dates, negative_counts, width, 
                bottom=[p + n for p, n in zip(positive_counts, neutral_counts)], 
                label='Negative', color='#DC143C', alpha=0.8)
        
        ax1.set_title(f'Sentiment Distribution Over Time for "{keyword}"', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Posts', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Sentiment ratio over time
        total_counts = [p + n + neg for p, n, neg in zip(positive_counts, neutral_counts, negative_counts)]
        positive_ratios = [p / t if t > 0 else 0 for p, t in zip(positive_counts, total_counts)]
        negative_ratios = [n / t if t > 0 else 0 for n, t in zip(negative_counts, total_counts)]
        
        ax2.plot(dates, positive_ratios, marker='o', color='#2E8B57', linewidth=2, markersize=6, label='Positive %')
        ax2.plot(dates, negative_ratios, marker='s', color='#DC143C', linewidth=2, markersize=6, label='Negative %')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% line')
        
        ax2.set_title('Sentiment Ratios Over Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sentiment Ratio', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Format x-axis based on time range
        if time_range <= timedelta(hours=24):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_range <= timedelta(days=7):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        total_posts = len(analyzed_posts)
        pos_count = sum(1 for post in analyzed_posts if post['label'].lower() == 'positive')
        neg_count = sum(1 for post in analyzed_posts if post['label'].lower() == 'negative')
        neu_count = total_posts - pos_count - neg_count
        
        print(f"\n📈 Timeline Analysis Summary for '{keyword}':")
        print(f"   Total posts analyzed: {total_posts}")
        print(f"   Positive: {pos_count} ({pos_count/total_posts*100:.1f}%)")
        print(f"   Negative: {neg_count} ({neg_count/total_posts*100:.1f}%)")
        print(f"   Neutral: {neu_count} ({neu_count/total_posts*100:.1f}%)")
        print(f"   Time range: {min(timestamps).strftime('%Y-%m-%d %H:%M')} to {max(timestamps).strftime('%Y-%m-%d %H:%M')}")
        
        return fig
        
    except Exception as e:
        print(f"❌ Error creating timeline plot: {e}")
        return None

def analyze_reddit_sentiment_with_timeline(reddit, model, tokenizer, keyword, limit=50, save_plot=False):
    """Complete pipeline for Reddit sentiment analysis with timeline visualization"""
    print(f"🚀 Starting sentiment timeline analysis for '{keyword}'")
    
    # Fetch posts with timestamps
    posts_data = fetch_reddit_posts_with_timestamps(reddit, keyword, limit)
    
    if not posts_data:
        print("❌ No posts found to analyze")
        return None
    
    # Extract texts for sentiment prediction
    texts = [post['text'] for post in posts_data]
    
    # Predict sentiments
    print("🧠 Analyzing sentiment...")
    sentiment_results = predict_sentiment(model, tokenizer, texts)
    
    # Combine results with timestamps
    analyzed_posts = []
    for i, (text, label, confidence) in enumerate(sentiment_results):
        analyzed_posts.append({
            'text': text,
            'label': label,
            'confidence': confidence,
            'timestamp': posts_data[i]['timestamp']
        })
    
    # Create timeline plot
    save_path = f"sentiment_timeline_{keyword.replace(' ', '_')}.png" if save_plot else None
    plot_sentiment_timeline(analyzed_posts, keyword, save_path)
    
    return analyzed_posts

# --- Main Execution Function ---
def main():
    """Main function to demonstrate the complete pipeline"""
    print("🎯 Reddit Sentiment Analysis with Timeline Visualization")
    print("=" * 60)
    
    # Reddit credentials (replace with your own)
    REDDIT_CLIENT_ID = "5gTPi8UyNTHCaFFXoVeBxw"
    REDDIT_CLIENT_SECRET = "vTpC0vIaQtbMSEiDETqtMkBlD2njyw"
    REDDIT_USER_AGENT = "BedroomSubject5844"
    
    # Initialize Reddit (will use fallback data if credentials not provided)
    reddit = init_reddit(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
    
    # For demonstration, we'll create a simple model (in practice, load your trained model)
    print("🔧 Setting up model (demo mode)...")
    
    # Demo tokenizer setup (in practice, load your trained tokenizer)
    sample_texts = ["This is great", "This is terrible", "This is okay"]
    tokenizer = prepare_tokenizer(sample_texts)
    
    # Demo model (in practice, load your trained model)
    model = build_model()
    
    # Example usage
    keywords = ["bitcoin", "tesla", "climate change"]
    
    for keyword in keywords:
        print(f"\n🔍 Analyzing sentiment timeline for: '{keyword}'")
        print("-" * 40)
        
        results = analyze_reddit_sentiment_with_timeline(
            reddit=reddit,
            model=model,
            tokenizer=tokenizer,
            keyword=keyword,
            limit=30,
            save_plot=True
        )
        
        if results:
            print(f"✅ Analysis complete for '{keyword}'")
        else:
            print(f"❌ Analysis failed for '{keyword}'")
        
        print()

if __name__ == "__main__":
    main()