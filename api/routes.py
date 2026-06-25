from flask import Blueprint, request, jsonify, render_template
from nlp.pipeline import get_response

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/")
def index():
    return render_template("index.html")


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing message field"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    result = get_response(message)
    filtered_result = {k: v for k, v in result.items() if k not in ('intent', 'confidence')}
    return jsonify(filtered_result)


@chat_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})
