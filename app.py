from flask import Flask, request, jsonify
import pickle
import os
import tempfile
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "scam_detector.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Whisper disabled - works 100% offline
whisper = None
print("Audio disabled - text analysis only (offline mode)")

def detect_scam(text):
    vec = vectorizer.transform([text])
    label = int(model.predict(vec)[0])
    prob = model.predict_proba(vec)[0]
    confidence = float(prob[label])
    return label, confidence

@app.route('/')
def index():
    return open(BASE_DIR / "templates" / "index.html", encoding="utf-8").read()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    label, confidence = detect_scam(text)
    return jsonify({'label': label, 'confidence': confidence})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    return jsonify({'error': 'Audio not available in offline mode.'}), 503

if __name__ == '__main__':
    print("Starting Anti-Scam Shield...")
    print("Open http://localhost:5000 in your browser!")
    app.run(debug=False, port=5000)