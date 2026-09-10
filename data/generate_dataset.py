"""
generate_dataset.py

Generates a labeled network-traffic dataset for intrusion detection,
using the same core feature schema as the well-known NSL-KDD dataset
(duration, protocol_type, service, flag, src_bytes, dst_bytes, etc.).

The data here is SYNTHETIC (built with numpy, with realistic statistical
structure so a model can actually learn from it) rather than downloaded,
because this environment has no internet access. To train on the real
public dataset instead, download NSL-KDD from
https://www.unb.ca/cic/datasets/nsl.html and point train_model.py at it —
the column names below match, so no other code needs to change.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_NORMAL = 3500
N_ATTACK = 1500

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "dns", "ssh", "private"]
FLAGS = ["SF", "S0", "REJ", "RSTO"]


def make_normal(n):
    return pd.DataFrame({
        "duration": RNG.exponential(scale=2.0, size=n).round(2),
        "protocol_type": RNG.choice(PROTOCOLS, size=n, p=[0.7, 0.25, 0.05]),
        "service": RNG.choice(SERVICES, size=n),
        "flag": RNG.choice(FLAGS, size=n, p=[0.85, 0.05, 0.05, 0.05]),
        "src_bytes": RNG.normal(300, 80, size=n).clip(0).round(0),
        "dst_bytes": RNG.normal(500, 150, size=n).clip(0).round(0),
        "count": RNG.integers(1, 15, size=n),
        "srv_count": RNG.integers(1, 15, size=n),
        "num_failed_logins": RNG.choice([0, 0, 0, 1], size=n),
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "label": "normal",
    })


def make_attack(n):
    # Attacks skew toward higher byte counts, more failed logins,
    # more repeated connections (count/srv_count), occasional wrong
    # fragments and SYN-flood-style flags — loosely modelled on
    # DoS / probe / R2L patterns in NSL-KDD.
    return pd.DataFrame({
        "duration": RNG.exponential(scale=0.3, size=n).round(2),
        "protocol_type": RNG.choice(PROTOCOLS, size=n, p=[0.5, 0.1, 0.4]),
        "service": RNG.choice(SERVICES, size=n, p=[0.4, 0.1, 0.1, 0.05, 0.05, 0.3]),
        "flag": RNG.choice(FLAGS, size=n, p=[0.1, 0.55, 0.25, 0.1]),
        "src_bytes": RNG.normal(4000, 1200, size=n).clip(0).round(0),
        "dst_bytes": RNG.normal(50, 40, size=n).clip(0).round(0),
        "count": RNG.integers(20, 80, size=n),
        "srv_count": RNG.integers(20, 80, size=n),
        "num_failed_logins": RNG.choice([0, 1, 2, 3], size=n, p=[0.3, 0.3, 0.25, 0.15]),
        "land": RNG.choice([0, 1], size=n, p=[0.9, 0.1]),
        "wrong_fragment": RNG.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        "urgent": RNG.choice([0, 1], size=n, p=[0.95, 0.05]),
        "label": "attack",
    })


def main():
    df = pd.concat([make_normal(N_NORMAL), make_attack(N_ATTACK)], ignore_index=True)

    # Add label noise (mislabel a small fraction) and feature jitter so the
    # problem isn't trivially separable — mirrors the ambiguity real traffic
    # captures have, and keeps the model's reported accuracy realistic.
    flip_idx = RNG.choice(df.index, size=int(0.04 * len(df)), replace=False)
    df.loc[flip_idx, "label"] = df.loc[flip_idx, "label"].map(
        {"normal": "attack", "attack": "normal"}
    )
    for col in ["src_bytes", "dst_bytes", "count", "srv_count"]:
        df[col] = (df[col] + RNG.normal(0, df[col].std() * 0.15, size=len(df))).clip(lower=0).round(0)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    out_path = "data/network_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({N_NORMAL} normal / {N_ATTACK} attack) to {out_path}")


if __name__ == "__main__":
    main()
