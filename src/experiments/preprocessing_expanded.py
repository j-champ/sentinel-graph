import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SPLIT_DIR = Path("data/splits_expanded")

def load_split(split):
    X_list, y_attack_list, y_intensity_list = [], [], []

    for file in sorted((SPLIT_DIR / split).glob("*.npz")):
        with np.load(file) as data:
            X_list.append(data["X"])
            y_attack_list.append(data["y_attack"])
            y_intensity_list.append(data["y_intensity"])

    if not X_list:
        raise ValueError(f"No sequence files found in {SPLIT_DIR / split}")

    return (
        np.concatenate(X_list),
        np.concatenate(y_attack_list),
        np.concatenate(y_intensity_list),
    )

def prepare_data():
    X_train, y_train, intensity_train = load_split("train")
    X_val, y_val, intensity_val = load_split("val")
    X_test, y_test, intensity_test = load_split("test")

    print("Original shapes:")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)
    X_test = X_test.reshape(len(X_test), -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\nModel-ready shapes:")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    return (
        X_train, y_train, intensity_train,
        X_val, y_val, intensity_val,
        X_test, y_test, intensity_test,
        scaler,
    )

if __name__ == "__main__":
    prepare_data()
