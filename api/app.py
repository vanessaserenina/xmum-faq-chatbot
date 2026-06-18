import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_cors import CORS
from api.routes import chat_bp
from nlp.pipeline import load_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "templates"),
        static_folder=os.path.join(BASE_DIR, "static"),
        static_url_path="/static"
    )

    CORS(app)

    load_pipeline()

    app.register_blueprint(chat_bp)

    return app
