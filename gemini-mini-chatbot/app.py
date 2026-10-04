import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import errors
from google.genai import types


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL_NAME = "gemini-3.5-flash-lite"
MAX_HISTORY_MESSAGES = 40
MAX_MESSAGE_LENGTH = 8000


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return jsonify({"error": "Add your Gemini API key to the .env file, then restart the app."}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "The request must contain a JSON object."}), 400

    history = data.get("history")
    if not isinstance(history, list) or not history:
        return jsonify({"error": "Enter a message before sending."}), 400
    if len(history) > MAX_HISTORY_MESSAGES:
        return jsonify({"error": "This conversation is too long. Clear the chat and try again."}), 400

    contents = []
    for message in history:
        if not isinstance(message, dict):
            return jsonify({"error": "The conversation contains an invalid message."}), 400

        role = message.get("role")
        text = message.get("text")
        if role not in {"user", "model"} or not isinstance(text, str):
            return jsonify({"error": "The conversation contains an invalid message."}), 400

        text = text.strip()
        if not text or len(text) > MAX_MESSAGE_LENGTH:
            return jsonify({"error": "Messages must be between 1 and 8,000 characters."}), 400

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            )
        )

    if contents[-1].role != "user":
        return jsonify({"error": "Send a new message to continue the conversation."}), 400

    client = None
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
        except errors.APIError as error:
            if error.code not in {429, 500, 503} or MODEL_NAME == FALLBACK_MODEL_NAME:
                raise

            app.logger.warning(
                "Gemini model %s returned HTTP %s; retrying with fallback model %s",
                MODEL_NAME,
                error.code,
                FALLBACK_MODEL_NAME,
            )
            response = client.models.generate_content(
                model=FALLBACK_MODEL_NAME,
                contents=contents,
            )
        answer = (response.text or "").strip()
        if not answer:
            return jsonify({"error": "Gemini returned an empty response. Please try again."}), 502
        return jsonify({"reply": answer})
    except errors.APIError as error:
        app.logger.exception("Gemini API request failed")
        if error.code in {429, 500, 503}:
            return jsonify({"error": "Gemini is temporarily busy. Please try again shortly."}), 503
        return jsonify({"error": "Gemini could not respond. Check your API key and try again."}), 502
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "Gemini could not respond right now. Check your API key and connection, then try again."}), 502
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
