from pathlib import Path
import shutil
import numpy as np

SEQUENCE_DIRS = [
    Path("data/sequences_continuous"),
    Path("data/sequences_expanded"),
]
SPLIT_DIR = Path("data/splits_expanded")

SPLITS = {
    "train": [
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.npz",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
        "Friday-16-02-2018_TrafficForML_CICFlowMeter.npz",
        "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "val": [
        "Friday-23-02-2018_TrafficForML_CICFlowMeter.npz",
        "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "test": [
        "Friday-02-03-2018_TrafficForML_CICFlowMeter.npz",
    ],
}

def find_source(filename):
    matches = [directory / filename for directory in SEQUENCE_DIRS
               if (directory / filename).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source for {filename}; found {len(matches)}"
        )
    return matches[0]

def main():
    if SPLIT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite: {SPLIT_DIR}")

    sources = {
        filename: find_source(filename)
        for filenames in SPLITS.values()
        for filename in filenames
    }

    for split, filenames in SPLITS.items():
        split_dir = SPLIT_DIR / split
        split_dir.mkdir(parents=True)

        for filename in filenames:
            shutil.copy2(sources[filename], split_dir / filename)

    for split in ("train", "val", "test"):
        X_list, y_list = [], []

        for file in sorted((SPLIT_DIR / split).glob("*.npz")):
            with np.load(file) as data:
                X_list.append(data["X"])
                y_list.append(data["y_attack"])

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        print(f"\n{split.upper()}")
        print("X shape:", X.shape)
        print("Benign:", int((y == 0).sum()))
        print("Attack:", int((y == 1).sum()))
        print("Total:", len(y))

    print("\nExpanded splits created.")

if __name__ == "__main__":
    main()
