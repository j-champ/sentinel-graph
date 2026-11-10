from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/interim_expanded")
OUTPUT_DIR = Path("data/features_expanded")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for source in sorted(INPUT_DIR.glob("*.parquet")):
    output = OUTPUT_DIR / source.name

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    df = pd.read_parquet(source)

    df["total_packets"] = df["fwd_packets"] + df["bwd_packets"]
    df["total_bytes"] = df["fwd_bytes"] + df["bwd_bytes"]

    denominator = df["total_packets"].replace(0, float("nan"))

    df["avg_packet_size"] = df["total_bytes"] / denominator
    df["fwd_ratio"] = df["fwd_packets"] / denominator
    df["bwd_ratio"] = df["bwd_packets"] / denominator
    df["syn_rate"] = df["syn_count"] / denominator
    df["ack_rate"] = df["ack_count"] / denominator
    df["rst_rate"] = df["rst_count"] / denominator

    df.to_parquet(output, index=False)

    print(f"\n{source.name}")
    print("Rows:", len(df))
    print("Missing values:", df.isna().sum().sum())
    print("Saved:", output)
