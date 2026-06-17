import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_cors import CORS
from api.routes import chat_bp
from nlp.pipeline import load_pipeline


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    )

    CORS(app)

    load_pipeline()

    app.register_blueprint(chat_bp)

    return app
