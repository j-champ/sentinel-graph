from pathlib import Path

import pandas as pd


STATE_DIRS = [
    Path("data/states"),
    Path("data/states_expanded"),
]

WINDOW = pd.Timedelta(seconds=30)


def audit_file(path: Path) -> dict:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["attack"] = (df["is_attack_mean"] > 0).astype(int)

    # A new run begins whenever adjacent states are not exactly 30 seconds apart.
    gaps = df["timestamp"].diff()
    run_id = gaps.ne(WINDOW).cumsum()
    df["run_id"] = run_id

    onsets = 0
    terminations = 0
    isolated = 0
    sustained_episodes = 0
    attack_episodes = 0
    attack_states = int(df["attack"].sum())

    for _, run in df.groupby("run_id", sort=False):
        states = run["attack"].tolist()

        for i in range(1, len(states)):
            if states[i - 1] == 0 and states[i] == 1:
                onsets += 1
            elif states[i - 1] == 1 and states[i] == 0:
                terminations += 1

        # Count contiguous attack episodes within this continuous run.
        i = 0
        while i < len(states):
            if states[i] == 0:
                i += 1
                continue

            start = i
            while i < len(states) and states[i] == 1:
                i += 1

            length = i - start
            attack_episodes += 1

            if length == 1 and start > 0 and i < len(states):
                if states[start - 1] == 0 and states[i] == 0:
                    isolated += 1

            if length >= 2:
                sustained_episodes += 1

    return {
        "day": path.name,
        "states": len(df),
        "attack_states": attack_states,
        "attack_pct": round(100 * attack_states / len(df), 2) if len(df) else 0,
        "continuous_runs": int(df["run_id"].nunique()),
        "timestamp_gaps": int((gaps.notna() & gaps.ne(WINDOW)).sum()),
        "onsets": onsets,
        "terminations": terminations,
        "attack_episodes": attack_episodes,
        "isolated_episodes": isolated,
        "sustained_episodes": sustained_episodes,
    }


def main():
    files = []
    for directory in STATE_DIRS:
        if directory.exists():
            files.extend(directory.glob("*.parquet"))

    results = [audit_file(path) for path in sorted(files)]

    if not results:
        raise FileNotFoundError("No state Parquet files found.")

    report = pd.DataFrame(results)
    print("\nATTACK EPISODE AUDIT — 30-SECOND CONTINUITY")
    print("=" * 110)
    print(report.to_string(index=False))

    print("\nTOTALS")
    print("=" * 50)
    for column in [
        "states",
        "attack_states",
        "timestamp_gaps",
        "onsets",
        "terminations",
        "attack_episodes",
        "isolated_episodes",
        "sustained_episodes",
    ]:
        print(f"{column}: {report[column].sum()}")

    print(
        "\nNote: isolated episodes are single attack states with benign "
        "neighbors on both sides within the same continuous run."
    )


if __name__ == "__main__":
    main()
