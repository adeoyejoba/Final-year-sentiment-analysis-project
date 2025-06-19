# 🔹 NEW: API route for Reddit keyword sentiment analysis (for frontend JS use)
from flask import Blueprint, request, render_template, jsonify
from .model_utils import load_model_tokenizer, predict_text_sentiment
from .reddit_utils import analyze_reddit_comments

routes = Blueprint('routes', __name__)

# Load the model and tokenizer once at the beginning
model, tokenizer = load_model_tokenizer()

@routes.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@routes.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@routes.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

# 🔹 API-style route (for frontend JavaScript use, returns JSON)
@routes.route('/analyze-text', methods=['POST'])
def analyze_text_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400

    text = data['text']
    try:
        label, confidence = predict_text_sentiment(model, tokenizer, text)
        
        # Convert single prediction to probability distribution
        if label == 'Positive':
            positive_prob = confidence
            negative_prob = (1 - confidence) / 2
            neutral_prob = (1 - confidence) / 2
        elif label == 'Negative':
            negative_prob = confidence
            positive_prob = (1 - confidence) / 2
            neutral_prob = (1 - confidence) / 2
        else:  # Neutral
            neutral_prob = confidence
            positive_prob = (1 - confidence) / 2
            negative_prob = (1 - confidence) / 2
        
        return jsonify({
            'prediction': label.lower(),
            'probabilities': {
                'positive': round(positive_prob, 3),
                'neutral': round(neutral_prob, 3),
                'negative': round(negative_prob, 3)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🔹 Form-based route for Reddit keyword analysis (for HTML form rendering)
@routes.route('/analyze', methods=['POST'])
def analyze_form():
    keyword = request.form.get('text')
    if not keyword:
        return render_template('result.html', error="Please enter a keyword.", source="reddit")

    try:
        comments, timeline_data = analyze_reddit_comments(model, tokenizer, keyword)
        sentiment_summary = {
            "Positive": sum(1 for c in comments if c["label"] == "Positive"),
            "Neutral": sum(1 for c in comments if c["label"] == "Neutral"),
            "Negative": sum(1 for c in comments if c["label"] == "Negative")
        }

        return render_template("result.html",
            source='reddit',
            keyword=keyword,
            reddit_comments=comments,
            sentiment_summary=sentiment_summary,
            timeline_data=timeline_data
        )
    except Exception as e:
        return render_template("result.html", error=str(e), source="reddit")

# 🔹 Optional: Form-based route for single text sentiment analysis (HTML form)
@routes.route('/analyze-text-form', methods=['POST'])
def analyze_text_form():
    text = request.form.get('text')
    if not text:
        return render_template("result.html", error="No text provided.", source="text")

    try:
        label, confidence = predict_text_sentiment(model, tokenizer, text)
        return render_template("result.html",
            source='text',
            input_text=text,
            result=label,
            confidence=round(confidence * 100, 2)
        )
    except Exception as e:
        return render_template("result.html", error=str(e), source="text")

# 🔹 NEW: API route for Reddit keyword sentiment analysis (for frontend JS use)
@routes.route('/analyze-reddit', methods=['POST'])
def analyze_reddit_api():
    data = request.get_json()
    if not data or 'keyword' not in data:
        return jsonify({'error': 'Missing "keyword" field'}), 400
    
    keyword = data['keyword']
    try:
        print(f"🔍 Received request for keyword: '{keyword}'")
        comments, timeline_data = analyze_reddit_comments(model, tokenizer, keyword)
        
        if not comments:
            return jsonify({
                "success": False,
                "message": "No comments found or analysis failed. Try a different keyword.",
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "dominant": "Unknown",
                "comments": [],
                "timeline": []
            })
        
        # Count sentiments
        sentiment_counts = {
            "positive": sum(1 for c in comments if c["label"] == "Positive"),
            "neutral": sum(1 for c in comments if c["label"] == "Neutral"),
            "negative": sum(1 for c in comments if c["label"] == "Negative")
        }
        
        total = sum(sentiment_counts.values())
        print(f"📊 Total comments analyzed: {total}")
        print(f"📊 Sentiment counts: {sentiment_counts}")
        
        if total == 0:
            return jsonify({
                "success": False,
                "message": "No valid sentiments detected.",
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "dominant": "Unknown",
                "comments": [],
                "timeline": []
            })
        
        # Calculate percentages
        percentages = {
            "positive": round((sentiment_counts["positive"] / total) * 100, 1),
            "neutral": round((sentiment_counts["neutral"] / total) * 100, 1),
            "negative": round((sentiment_counts["negative"] / total) * 100, 1)
        }
        
        # Determine dominant sentiment
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        dominant_label = dominant_sentiment.capitalize()
        
        print(f"📊 Percentages: {percentages}")
        print(f"🏆 Dominant sentiment: {dominant_label}")
        print(f"📈 Timeline data points: {len(timeline_data)}")
        
        # Format comments for response
        formatted_comments = []
        for comment in comments[:10]:  # Limit to first 10 comments
            formatted_comments.append({
                "text": comment["text"],
                "label": comment["label"],
                "confidence": comment["confidence"]
            })
        
        response_data = {
            "success": True,
            "positive": percentages["positive"],
            "neutral": percentages["neutral"],
            "negative": percentages["negative"],
            "dominant": dominant_label,
            "total_analyzed": total,
            "comments": formatted_comments,
            "timeline": timeline_data  # ✅ Added timeline data
        }
        
        print(f"✅ Sending response with timeline data")
        return jsonify(response_data)
        
    except Exception as e:
        print("🔥 Reddit sentiment analysis failed:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Could not fetch Reddit sentiment. Try again later.",
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "dominant": "Error",
            "comments": [],
            "timeline": []
        })