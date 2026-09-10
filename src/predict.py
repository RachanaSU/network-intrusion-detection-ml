"""
predict.py

Command-line tool: load the trained model and classify one connection
record as "normal" or "attack".

Usage:
    python src/predict.py --duration 0.1 --protocol_type tcp --service private \
        --flag S0 --src_bytes 5000 --dst_bytes 20 --count 60 --srv_count 55 \
        --num_failed_logins 2 --land 0 --wrong_fragment 1 --urgent 0
"""

import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/nids_model.joblib"


def parse_args():
    p = argparse.ArgumentParser(description="Classify a network connection record.")
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--protocol_type", type=str, required=True, choices=["tcp", "udp", "icmp"])
    p.add_argument("--service", type=str, required=True)
    p.add_argument("--flag", type=str, required=True)
    p.add_argument("--src_bytes", type=float, required=True)
    p.add_argument("--dst_bytes", type=float, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--srv_count", type=int, required=True)
    p.add_argument("--num_failed_logins", type=int, default=0)
    p.add_argument("--land", type=int, default=0)
    p.add_argument("--wrong_fragment", type=int, default=0)
    p.add_argument("--urgent", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    model = joblib.load(MODEL_PATH)

    row = pd.DataFrame([{
        "duration": args.duration,
        "protocol_type": args.protocol_type,
        "service": args.service,
        "flag": args.flag,
        "src_bytes": args.src_bytes,
        "dst_bytes": args.dst_bytes,
        "count": args.count,
        "srv_count": args.srv_count,
        "num_failed_logins": args.num_failed_logins,
        "land": args.land,
        "wrong_fragment": args.wrong_fragment,
        "urgent": args.urgent,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    confidence = proba[classes.index(pred)]

    print(f"Prediction: {pred.upper()}  (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
