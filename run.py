from flask import Flask
from app import create_app
import os

# Explicitly tell Flask where to look for templates
template_dir = os.path.abspath("templates")
app = create_app()
app.template_folder = template_dir

if __name__ == "__main__":
    app.run(port=8000, debug=True)
