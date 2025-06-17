from flask import Flask, Blueprint, render_template, request
from app.model_utils import load_model_tokenizer, predict_text_sentiment
from app.reddit_utils import analyze_reddit_sentiment, init_reddit

# Create the Blueprint
main = Blueprint("main", __name__)

# Load model and tokenizer once at startup
model, tokenizer = load_model_tokenizer()
reddit = init_reddit()

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form.get("user_text")
        reddit_keyword = request.form.get("reddit_keyword")

        if user_text:
            label, confidence = predict_text_sentiment(model, tokenizer, user_text)
            return render_template("result.html", result=label, confidence=confidence, source="text")

        elif reddit_keyword:
            results = analyze_reddit_sentiment(model, tokenizer, reddit, reddit_keyword)
            return render_template("result.html", reddit_results=results, keyword=reddit_keyword, source="reddit")

    return render_template("index.html")

# Create and configure the Flask app
def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)
    return app
