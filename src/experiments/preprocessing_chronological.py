import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SPLIT_DIR = Path("data/splits_chronological")


def load_split(split):
    X_list = []
    y_attack_list = []
    y_intensity_list = []

    split_dir = SPLIT_DIR / split

    for file in sorted(split_dir.glob("*.npz")):
        with np.load(file) as data:
            X_list.append(data["X"])
            y_attack_list.append(data["y_attack"])
            y_intensity_list.append(data["y_intensity"])

    if not X_list:
        raise ValueError(f"No sequence files found in {split_dir}")

    X = np.concatenate(X_list, axis=0)
    y_attack = np.concatenate(y_attack_list, axis=0)
    y_intensity = np.concatenate(y_intensity_list, axis=0)

    return X, y_attack, y_intensity


def prepare_data():
    X_train, y_train, intensity_train = load_split("train")
    X_val, y_val, intensity_val = load_split("val")
    X_test, y_test, intensity_test = load_split("test")

    print("Original shapes:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    print("\nAttack target distribution:")
    for name, y in [
        ("Train", y_train),
        ("Val", y_val),
        ("Test", y_test),
    ]:
        print(f"{name}: {np.bincount(y, minlength=2)}")

    # Flatten temporal sequences for classical ML models.
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Fit scaler only on training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\nModel-ready shapes:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    return (
        X_train,
        y_train,
        intensity_train,
        X_val,
        y_val,
        intensity_val,
        X_test,
        y_test,
        intensity_test,
        scaler,
    )


if __name__ == "__main__":
    prepare_data()
