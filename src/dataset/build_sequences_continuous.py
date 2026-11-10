from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("data/states")
OUTPUT_DIR = Path("data/sequences_continuous")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 10
WINDOW = pd.Timedelta(seconds=30)

FEATURES = [
    "total_packets_mean",
    "total_packets_sum",
    "total_packets_max",
    "total_bytes_mean",
    "total_bytes_sum",
    "total_bytes_max",
    "avg_packet_size_mean",
    "avg_packet_size_max",
    "fwd_ratio_mean",
    "fwd_ratio_max",
    "bwd_ratio_mean",
    "bwd_ratio_max",
    "byte_rate_mean",
    "byte_rate_max",
    "packet_rate_mean",
    "packet_rate_max",
    "syn_rate_mean",
    "syn_rate_max",
    "ack_rate_mean",
    "ack_rate_max",
    "rst_rate_mean",
    "rst_rate_max",
    "flow_duration_mean",
    "flow_duration_max",
]


def build_sequences(df):
    X = []
    y_attack = []
    y_intensity = []
    last_attack = []

    values = df[FEATURES].to_numpy()
    attack_ratio = df["is_attack_mean"].to_numpy()

    for i in range(len(df) - SEQ_LEN):
        X.append(values[i:i + SEQ_LEN])

        future_attack = attack_ratio[i + SEQ_LEN]
        current_attack = attack_ratio[i + SEQ_LEN - 1]

        y_attack.append(int(future_attack > 0))
        y_intensity.append(future_attack)
        last_attack.append(int(current_attack > 0))

    return (
        np.asarray(X),
        np.asarray(y_attack),
        np.asarray(y_intensity),
        np.asarray(last_attack),
    )


for file in sorted(INPUT_DIR.glob("*.parquet")):
    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Start a new run whenever adjacent timestamps are not 30 seconds apart.
    new_run = df["timestamp"].diff().ne(WINDOW)
    df["run_id"] = new_run.cumsum()

    runs = [
        run.drop(columns="run_id")
        for _, run in df.groupby("run_id")
        if len(run) > SEQ_LEN
    ]

    results = [build_sequences(run) for run in runs]

    if results:
        X = np.concatenate([r[0] for r in results])
        y_attack = np.concatenate([r[1] for r in results])
        y_intensity = np.concatenate([r[2] for r in results])
        last_attack = np.concatenate([r[3] for r in results])
    else:
        X = np.empty((0, SEQ_LEN, len(FEATURES)))
        y_attack = np.empty(0, dtype=int)
        y_intensity = np.empty(0, dtype=float)
        last_attack = np.empty(0, dtype=int)

    output = OUTPUT_DIR / f"{file.stem}.npz"

    np.savez(
        output,
        X=X,
        y_attack=y_attack,
        y_intensity=y_intensity,
        last_attack=last_attack,
    )

    print("Continuous runs:", len(runs))
    print("X:", X.shape)
    print("y_attack:", y_attack.shape)
    print("y_intensity:", y_intensity.shape)
    print("last_attack:", last_attack.shape)
    print("Saved:", output)
