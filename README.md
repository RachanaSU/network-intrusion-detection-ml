# Network Intrusion Detection using Machine Learning

A small end-to-end ML project that classifies network connection records as
**normal** or **attack**, built on the connection-level feature schema used
by the well-known [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
intrusion-detection dataset (duration, protocol type, service, flag,
byte counts, failed-login counts, etc.).

Built to combine my networking background (from my MEng thesis on
data-centre routing) with practical machine learning — training, evaluating,
and serving a classifier as an API.

## What's in here

| File | Purpose |
|---|---|
| `data/generate_dataset.py` | Builds a labeled connection-record dataset with realistic statistical structure (normal vs. attack traffic patterns, plus label noise and feature jitter so it isn't trivially separable) |
| `src/train_model.py` | Trains a Random Forest classifier (with one-hot encoding for categorical fields) and evaluates it on a held-out test set |
| `src/predict.py` | CLI to classify a single connection record |
| `src/app.py` | Flask API exposing `/predict` and `/health` endpoints |
| `models/nids_model.joblib` | The trained pipeline (preprocessing + model) |
| `results/confusion_matrix.png`, `results/metrics.txt` | Evaluation output from the last training run |

## Results

Evaluated on a 25% held-out test split (1,250 records):

```
Accuracy: 0.97
F1 (attack class): 0.95

              precision    recall  f1-score   support
      attack       0.97      0.94      0.95       390
      normal       0.97      0.99      0.98       860
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python data/generate_dataset.py

# 2. Train the model
python src/train_model.py

# 3a. Classify a single record from the command line
python src/predict.py --duration 0.1 --protocol_type tcp --service private \
    --flag S0 --src_bytes 5000 --dst_bytes 20 --count 60 --srv_count 55 \
    --num_failed_logins 2 --land 0 --wrong_fragment 1 --urgent 0

# 3b. Or serve it as an API
python src/app.py
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
    -d '{"duration":0.1,"protocol_type":"tcp","service":"private","flag":"S0",
         "src_bytes":5000,"dst_bytes":20,"count":60,"srv_count":55,
         "num_failed_logins":2,"land":0,"wrong_fragment":1,"urgent":0}'
```

## About the data

The dataset generated here is **synthetic**, built with numpy to mirror the
statistical patterns of real intrusion-detection traffic (e.g. attacks
tend to have higher `src_bytes`, more repeated connections, more failed
logins, and SYN-style flags). This was necessary to build and test the
project without network access to a hosting service. The feature schema
matches NSL-KDD exactly, so the same code trains directly on the real
dataset — download it from the link above and point `train_model.py` at it
in place of `data/network_traffic.csv` to reproduce this pipeline on real
traffic captures.

## Notes on process

I used an AI tool (Claude) to help scaffold this project, debug the
preprocessing pipeline, and write this README — the design decisions,
testing, and final code are my own.
