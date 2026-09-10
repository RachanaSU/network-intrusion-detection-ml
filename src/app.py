"""
app.py

Small Flask API that serves the trained intrusion-detection model.

Run:
    python src/app.py

Then:
    curl -X POST http://127.0.0.1:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"duration":0.1,"protocol_type":"tcp","service":"private","flag":"S0",
             "src_bytes":5000,"dst_bytes":20,"count":60,"srv_count":55,
             "num_failed_logins":2,"land":0,"wrong_fragment":1,"urgent":0}'
"""

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = "models/nids_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "count", "srv_count", "num_failed_logins", "land", "wrong_fragment", "urgent",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    row = pd.DataFrame([{f: payload[f] for f in REQUIRED_FIELDS}])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    confidence = float(proba[classes.index(pred)])

    return jsonify({"prediction": pred, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
