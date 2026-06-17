from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsp.feature_extraction import extract_features

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "saved_model", "noise_classifier.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

BUNDLE = None

def get_bundle():
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = load_model()
    return BUNDLE

SAMPLE_RATE = 10000
NUM_SAMPLES = 1000

def make_signal(noise_type):
    t = np.linspace(0, 0.1, NUM_SAMPLES)
    if noise_type == "impulsive":
        sig = np.random.normal(0, 0.1, NUM_SAMPLES)
        for pos in np.random.randint(0, NUM_SAMPLES, 5):
            sig[pos] += np.random.uniform(2, 5) * np.random.choice([-1, 1])
    elif noise_type == "narrowband":
        freq = np.random.uniform(1000, 4000)
        sig = np.random.uniform(0.5, 2.0) * np.sin(2 * np.pi * freq * t)
        sig += np.random.normal(0, 0.05, NUM_SAMPLES)
    else:
        f1 = np.random.uniform(200, 600)
        f2 = np.random.uniform(800, 1500)
        sig = (np.random.uniform(0.3, 1.0) * np.sin(2 * np.pi * f1 * t) +
               np.random.uniform(0.3, 1.0) * np.sin(2 * np.pi * f2 * t))
        sig += np.random.normal(0, 0.08, NUM_SAMPLES)
    return sig

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    noise_type = data.get("noise_type", "impulsive")
    signal = make_signal(noise_type)
    feats = extract_features(signal)
    bundle = get_bundle()
    pipeline = bundle["pipeline"]
    le = bundle["label_encoder"]
    feat_names = bundle["feature_names"]
    X = np.array([[feats[f] for f in feat_names]])
    pred_enc = pipeline.predict(X)[0]
    pred_proba = pipeline.predict_proba(X)[0]
    predicted_class = le.inverse_transform([pred_enc])[0]
    probabilities = {cls: float(prob) for cls, prob in zip(le.classes_, pred_proba)}
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(NUM_SAMPLES, d=1.0/SAMPLE_RATE)
    magnitude = np.abs(fft_vals)
    step = max(1, len(freqs)//200)
    return jsonify({
        "predicted": predicted_class, "actual": noise_type,
        "correct": predicted_class == noise_type,
        "probabilities": probabilities,
        "signal": signal[::5].tolist(),
        "freqs": freqs[::step].tolist(),
        "magnitude": magnitude[::step].tolist(),
        "features": {k: round(float(v), 4) for k, v in feats.items()}
    })

if __name__ == "__main__":
    print("Starting PLC Noise Classifier dashboard at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
