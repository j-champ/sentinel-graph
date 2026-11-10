from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/features_expanded")
OUTPUT_DIR = Path("data/states_expanded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGGREGATIONS = {
    "total_packets": ["mean", "sum", "max"],
    "total_bytes": ["mean", "sum", "max"],
    "avg_packet_size": ["mean", "max"],
    "fwd_ratio": ["mean", "max"],
    "bwd_ratio": ["mean", "max"],
    "byte_rate": ["mean", "max"],
    "packet_rate": ["mean", "max"],
    "syn_rate": ["mean", "max"],
    "ack_rate": ["mean", "max"],
    "rst_rate": ["mean", "max"],
    "flow_duration": ["mean", "max"],
    "is_attack": ["mean"],
}

for source in sorted(INPUT_DIR.glob("*.parquet")):
    output = OUTPUT_DIR / source.name

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    df = pd.read_parquet(source)
    df["is_attack"] = (df["label"] != "Benign").astype(int)

    states = (
        df.set_index("timestamp")
        .sort_index()
        .resample("30s")
        .agg(AGGREGATIONS)
    )

    states.columns = [
        f"{feature}_{stat}"
        for feature, stat in states.columns
    ]

    states = states.reset_index()

    before = len(states)
    states = states.dropna().reset_index(drop=True)

    states.to_parquet(output, index=False)

    print(f"\n{source.name}")
    print("Windows before dropping missing:", before)
    print("Valid windows:", len(states))
    print("Dropped windows:", before - len(states))
    print("Saved:", output)
