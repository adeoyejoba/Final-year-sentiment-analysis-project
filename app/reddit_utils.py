import praw
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Ensures compatibility without GUI
import matplotlib.pyplot as plt
from .text_cleaner import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime, timedelta
import json

def init_reddit():
    try:
        reddit = praw.Reddit(
            client_id="G-f4vfrqpcjsABMLNdqxmw",
            client_secret="TzYfgjlVgUe0IYE5ev2p1k_k7cA4HA",
            user_agent="sentimentApp by /u/BedroomSubject5844"
        )
        reddit.read_only = True
        # Test the connection
        reddit.user.me()
        return reddit
    except Exception as e:
        print(f"Reddit initialization failed: {e}")
        return None

def fetch_posts_with_timestamps(reddit, keyword, limit=100):
    """Fetch posts with their timestamps for timeline analysis"""
    posts_data = []
    if reddit is None:
        print("🟡 Reddit not available, using fallback posts")
        # Generate fallback data with timestamps
        now = datetime.utcnow()
        for i in range(min(limit, 20)):
            posts_data.append({
                'text': f"Sample post about {keyword}",
                'timestamp': now - timedelta(days=i % 7, hours=i % 24)
            })
        return posts_data
    
    try:
        print(f"🔍 Searching Reddit for: '{keyword}'")
        search_results = list(reddit.subreddit("all").search(keyword, limit=limit, sort='new'))
        print(f"📊 Found {len(search_results)} posts")
        
        for submission in search_results:
            # Combine title and selftext, handle None values
            post_text = submission.title or ""
            if hasattr(submission, 'selftext') and submission.selftext:
                post_text += " " + submission.selftext
            
            # Only add non-empty posts with valid timestamps
            if post_text.strip():
                posts_data.append({
                    'text': post_text.strip(),
                    'timestamp': datetime.utcfromtimestamp(submission.created_utc)
                })
        
        # If no posts found, provide fallback
        if not posts_data:
            print(f"🟡 No posts found for '{keyword}', using fallback")
            now = datetime.utcnow()
            for i in range(10):
                posts_data.append({
                    'text': f"Sample fallback post about {keyword}",
                    'timestamp': now - timedelta(days=i % 7, hours=i % 24)
                })
            
    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        # Fallback data
        now = datetime.utcnow()
        for i in range(10):
            posts_data.append({
                'text': f"Sample fallback post about {keyword}",
                'timestamp': now - timedelta(days=i % 7, hours=i % 24)
            })
    
    print(f"📝 Final posts count: {len(posts_data)}")
    return posts_data

def fetch_posts(reddit, keyword, limit=20):
    """Legacy function for compatibility"""
    posts_data = fetch_posts_with_timestamps(reddit, keyword, limit)
    return [post['text'] for post in posts_data]

def generate_timeline_data(analyzed_posts):
    """Generate sentiment timeline data from analyzed posts"""
    try:
        # Group posts by day for the last 7 days
        now = datetime.utcnow()
        timeline_data = {}
        
        # Initialize timeline for last 7 days
        for i in range(7):
            date_key = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            timeline_data[date_key] = {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'total': 0
            }
        
        # Group analyzed posts by date
        for post in analyzed_posts:
            if 'timestamp' in post:
                date_key = post['timestamp'].strftime('%Y-%m-%d')
                if date_key in timeline_data:
                    sentiment = post['label'].lower()
                    if sentiment in ['positive', 'neutral', 'negative']:
                        timeline_data[date_key][sentiment] += 1
                        timeline_data[date_key]['total'] += 1
        
        # Convert to percentages and format for frontend
        formatted_timeline = []
        for i in range(6, -1, -1):  # Last 7 days, chronologically
            date_key = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            day_data = timeline_data[date_key]
            
            if day_data['total'] > 0:
                percentages = {
                    'date': (now - timedelta(days=i)).strftime('%b %d'),
                    'positive': round((day_data['positive'] / day_data['total']) * 100, 1),
                    'neutral': round((day_data['neutral'] / day_data['total']) * 100, 1),
                    'negative': round((day_data['negative'] / day_data['total']) * 100, 1)
                }
            else:
                # No data for this day, use average from other days or reasonable defaults
                percentages = {
                    'date': (now - timedelta(days=i)).strftime('%b %d'),
                    'positive': 33.3,
                    'neutral': 33.3,
                    'negative': 33.4
                }
            
            formatted_timeline.append(percentages)
        
        print(f"📈 Generated timeline data: {len(formatted_timeline)} days")
        return formatted_timeline
        
    except Exception as e:
        print(f"Error generating timeline data: {e}")
        # Return fallback timeline data
        now = datetime.utcnow()
        fallback_timeline = []
        for i in range(6, -1, -1):
            fallback_timeline.append({
                'date': (now - timedelta(days=i)).strftime('%b %d'),
                'positive': 40.0 + (i * 2),  # Some variation
                'neutral': 35.0,
                'negative': 25.0 - (i * 2)
            })
        return fallback_timeline

def analyze_reddit_comments(model, tokenizer, keyword, limit=50):
    try:
        print(f"🚀 Starting Reddit analysis for: '{keyword}'")
        reddit = init_reddit()
        posts_data = fetch_posts_with_timestamps(reddit, keyword, limit)
        
        if not posts_data:
            print("❌ No posts to analyze")
            return [], []

        print(f"📊 Analyzing {len(posts_data)} posts...")
        
        # Clean the posts
        cleaned_posts = []
        valid_posts_data = []
        
        for i, post_data in enumerate(posts_data):
            cleaned = clean_text(post_data['text'])
            if cleaned.strip():  # Only keep non-empty cleaned posts
                cleaned_posts.append(cleaned)
                valid_posts_data.append(post_data)
            else:
                print(f"⚠️ Post {i+1} became empty after cleaning, skipping")
        
        if not cleaned_posts:
            print("❌ No valid posts after cleaning")
            return [], []
        
        print(f"✅ {len(cleaned_posts)} valid posts after cleaning")
        
        # Tokenize and predict
        sequences = tokenizer.texts_to_sequences(cleaned_posts)
        
        # Filter out empty sequences
        valid_sequences = []
        final_posts_data = []
        
        for i, seq in enumerate(sequences):
            if seq:  # Only keep non-empty sequences
                valid_sequences.append(seq)
                final_posts_data.append(valid_posts_data[i])
            else:
                print(f"⚠️ Post {i+1} produced empty sequence, skipping")
        
        if not valid_sequences:
            print("❌ No valid sequences for prediction")
            return [], []
        
        print(f"🔢 {len(valid_sequences)} valid sequences for prediction")
        
        # Pad sequences
        padded = pad_sequences(valid_sequences, maxlen=200)
        print(f"📐 Padded sequences shape: {padded.shape}")
        
        # Make predictions
        preds = model.predict(padded, verbose=0)
        print(f"🎯 Predictions shape: {preds.shape}")
        
        # Process results
        results = []
        analyzed_posts = []
        
        for i, score in enumerate(preds):
            score_val = float(score.item() if hasattr(score, 'item') else score[0])
            
            # Determine sentiment label
            if score_val >= 0.6:
                label = "Positive"
                confidence = score_val
            elif score_val <= 0.4:
                label = "Negative"
                confidence = 1 - score_val
            else:
                label = "Neutral"
                confidence = max(score_val, 1 - score_val)
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            post_result = {
                "text": final_posts_data[i]['text'][:200] + "..." if len(final_posts_data[i]['text']) > 200 else final_posts_data[i]['text'],
                "label": label,
                "confidence": round(confidence * 100, 2)
            }
            
            analyzed_post = {
                "text": final_posts_data[i]['text'],
                "label": label,
                "confidence": confidence,
                "timestamp": final_posts_data[i]['timestamp']
            }
            
            results.append(post_result)
            analyzed_posts.append(analyzed_post)
        
        print(f"✅ Analysis complete: {len(results)} results")
        
        # Debug: Print sentiment distribution
        sentiment_counts = Counter(r["label"] for r in results)
        print(f"📊 Sentiment distribution: {dict(sentiment_counts)}")
        
        # Generate timeline data
        timeline_data = generate_timeline_data(analyzed_posts)
        
        return results, timeline_data
        
    except Exception as e:
        print(f"🔥 analyze_reddit_comments() failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []