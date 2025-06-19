from flask import Flask
from flask_cors import CORS
from app.routes import routes

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = '6b478708a8429e1806c17bd097ce188e'
    
    CORS(app, resources={r"/*": {"origins": "*"}})

    app.register_blueprint(routes)  # ✅ This is critical

    return app
