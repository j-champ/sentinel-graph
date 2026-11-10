from pathlib import Path
import shutil

SEQUENCE_DIR = Path("data/sequences_continuous")
SPLIT_DIR = Path("data/splits_continuous")

SPLITS = {
    "train": [
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.npz",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "val": [
        "Friday-16-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "test": [
        "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.npz",
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
}


def main():
    for split, filenames in SPLITS.items():
        split_dir = SPLIT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            source = SEQUENCE_DIR / filename
            destination = split_dir / filename

            if not source.exists():
                raise FileNotFoundError(
                    f"Missing sequence file: {source}"
                )

            shutil.copy2(source, destination)
            print(f"{split.upper()}: {filename}")

    print("\nContinuous splits created successfully.")


if __name__ == "__main__":
    main()
