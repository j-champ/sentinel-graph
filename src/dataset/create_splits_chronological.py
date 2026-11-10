from pathlib import Path
import shutil
import numpy as np

SEQUENCE_DIR = Path("data/sequences_continuous")
SPLIT_DIR = Path("data/splits_chronological")

SPLITS = {
    "train": [
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.npz",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
        "Friday-16-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "val": [
        "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "test": [
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
}


def main():
    if SPLIT_DIR.exists():
        raise FileExistsError(
            f"{SPLIT_DIR} already exists. "
            "Inspect it before creating or modifying any files."
        )

    # Validate every source before creating the destination.
    for filenames in SPLITS.values():
        for filename in filenames:
            source = SEQUENCE_DIR / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing sequence file: {source}")

    for split, filenames in SPLITS.items():
        split_dir = SPLIT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            source = SEQUENCE_DIR / filename
            destination = split_dir / filename
            shutil.copy2(source, destination)
            print(f"{split.upper()}: {filename}")

    print("\n========== SPLIT SUMMARY ==========")

    for split in ("train", "val", "test"):
        X_list = []
        y_list = []

        for file in sorted((SPLIT_DIR / split).glob("*.npz")):
            with np.load(file) as data:
                X_list.append(data["X"])
                y_list.append(data["y_attack"])

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        print(f"{split.upper()}:")
        print(f"  X shape: {X.shape}")
        print(f"  Benign:  {int((y == 0).sum())}")
        print(f"  Attack:  {int((y == 1).sum())}")
        print(f"  Total:   {len(y)}")

    print("\nChronological splits created successfully.")


if __name__ == "__main__":
    main()
