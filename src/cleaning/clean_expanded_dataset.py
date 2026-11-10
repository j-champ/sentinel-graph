from pathlib import Path

import pandas as pd

from schema import CANONICAL_COLUMNS


RAW_DIR = Path("data/raw/CIC-IDS2018")
OUTPUT_DIR = Path("data/interim_expanded")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = [
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


NUMERIC_COLUMNS = [
    "dst_port",
    "flow_duration",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "byte_rate",
    "packet_rate",
    "flow_iat_mean",
    "flow_iat_std",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "packet_length_mean",
    "packet_length_std",
]


def clean_file(filename):

    source = RAW_DIR / filename
    output = OUTPUT_DIR / f"{source.stem}.parquet"

    if not source.exists():
        raise FileNotFoundError(source)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output}"
        )

    print(f"\nProcessing: {filename}")

    df = pd.read_csv(source, low_memory=False)

    original_rows = len(df)

    # Remove accidental repeated header rows.
    df = df[df["Label"] != "Label"].copy()

    repeated_headers = original_rows - len(df)

    # Rename original CIC-IDS2018 columns.
    df = df.rename(columns=CANONICAL_COLUMNS)

    # Keep the canonical columns available in this dataset.
    available = [
        column
        for column in CANONICAL_COLUMNS.values()
        if column in df.columns
    ]

    df = df[available].copy()

    # Preserve protocol and original attack labels.
    df["protocol"] = df["protocol"].astype("string")
    df["label"] = df["label"].astype("string")

    # Convert timestamps.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        dayfirst=True
    )

    invalid_timestamps = df["timestamp"].isna().sum()

    # Convert selected numerical features.
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = df[column].replace(
                [float("inf"), float("-inf")],
                float("nan")
            )

    # Remove invalid timestamps and invalid labels.
    df = df.dropna(
        subset=["timestamp", "label"]
    )

    # Retain only the expected CIC-IDS2018 date range.
    df = df[
        df["timestamp"].dt.year.between(2018, 2019)
    ].copy()

    # Chronological ordering.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save without median imputation.
    df.to_parquet(output, index=False)

    print("Original rows:", original_rows)
    print("Repeated headers removed:", repeated_headers)
    print("Invalid timestamps:", invalid_timestamps)
    print("Final rows:", len(df))
    print("Missing values:", df.isna().sum().sum())
    print("Labels:")
    print(df["label"].value_counts())
    print("Saved:", output)


if __name__ == "__main__":

    for filename in FILES:
        clean_file(filename)
