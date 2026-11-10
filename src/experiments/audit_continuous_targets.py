
import numpy as np
from pathlib import Path


TEST_DIR = Path("data/splits_continuous/test")


def main():
    for file in sorted(TEST_DIR.glob("*.npz")):
        with np.load(file) as data:
            y = data["y_attack"]
            last_attack = data["last_attack"]
            intensity = data["y_intensity"]

        print("\n" + "=" * 60)
        print(file.name)
        print("=" * 60)

        print("Total sequences:", len(y))

        print("\nTarget distribution:")
        print("Benign:", np.sum(y == 0))
        print("Attack:", np.sum(y == 1))

        print("\nPrevious-state distribution:")
        print("Previous benign:", np.sum(last_attack == 0))
        print("Previous attack:", np.sum(last_attack == 1))

        print("\nTarget transitions:")
        print(
            "Benign -> Benign:",
            np.sum((last_attack == 0) & (y == 0))
        )
        print(
            "Benign -> Attack:",
            np.sum((last_attack == 0) & (y == 1))
        )
        print(
            "Attack -> Benign:",
            np.sum((last_attack == 1) & (y == 0))
        )
        print(
            "Attack -> Attack:",
            np.sum((last_attack == 1) & (y == 1))
        )

        print("\nAttack intensity:")
        print("Mean:", round(float(np.mean(intensity)), 4))
        print("Median:", round(float(np.median(intensity)), 4))
        print("Min:", round(float(np.min(intensity)), 4))
        print("Max:", round(float(np.max(intensity)), 4))


if __name__ == "__main__":
    main()

