
import pandas as pd
import numpy as np
from pathlib import Path

STATE_DIR = Path("data/states")

TEST_DAYS = [
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet",
]


def main():
    for filename in TEST_DAYS:
        path = STATE_DIR / filename
        df = pd.read_parquet(path)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["attack_state"] = (
            df["is_attack_mean"] > 0
        ).astype(int)

        time_diff = df["timestamp"].diff()
        continuous = time_diff == pd.Timedelta(seconds=30)

        transitions = df.loc[
            continuous & (df["attack_state"] != df["attack_state"].shift()),
            ["timestamp", "attack_state", "is_attack_mean"]
        ]

        gaps = df.loc[
            time_diff.notna() & ~continuous,
            ["timestamp", "attack_state", "is_attack_mean"]
        ]

        print("\n" + "=" * 65)
        print(filename)
        print("=" * 65)

        print("Total states:", len(df))
        print("First timestamp:", df["timestamp"].min())
        print("Last timestamp:", df["timestamp"].max())

        print("\nAttack-state distribution:")
        print(df["attack_state"].value_counts().sort_index())

        print("\nTransitions within continuous runs:")
        print("Total:", len(transitions))
        print(transitions.to_string(index=False))

        print("\nTimestamp gaps:")
        print("Total:", len(gaps))

        if len(gaps):
            print("\nState immediately before and after each gap:")
            for idx in gaps.index:
                previous = df.loc[idx - 1]
                current = df.loc[idx]

                print(
                    f"{previous['timestamp']} "
                    f"(state={previous['attack_state']}) -> "
                    f"{current['timestamp']} "
                    f"(state={current['attack_state']})"
                )

        print("\nOverall intensity:")
        print("Mean:", df["is_attack_mean"].mean())
        print("Median:", df["is_attack_mean"].median())
        print("Maximum:", df["is_attack_mean"].max())


if __name__ == "__main__":
    main()

