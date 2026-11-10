import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


BASE = Path("data/splits_chronological")

TRAIN_CONFIGS = {
    "Feb 15 only": [
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "Feb 14 + Feb 15": [
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.npz",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
    "Feb 14 + Feb 15 + Feb 16": [
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.npz",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
        "Friday-16-02-2018_TrafficForML_CICFlowMeter.npz",
    ],
}


def load_files(paths):
    X_parts = []
    y_parts = []

    for path in paths:
        with np.load(path) as data:
            X_parts.append(data["X"])
            y_parts.append(data["y_attack"])

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y


def evaluate(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    test_file = BASE / "test" / (
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.npz"
    )

    X_test_raw, y_test = load_files([test_file])
    X_test_raw = X_test_raw.reshape(len(X_test_raw), -1)

    print("FIXED TEST SET")
    print("Sequences:", len(y_test))
    print("Class counts:", np.bincount(y_test, minlength=2))

    results = []

    for config_name, filenames in TRAIN_CONFIGS.items():
        train_paths = [BASE / "train" / name for name in filenames]

        for path in train_paths:
            if not path.exists():
                raise FileNotFoundError(path)

        X_train_raw, y_train = load_files(train_paths)
        X_train_raw = X_train_raw.reshape(len(X_train_raw), -1)

        # Fit scaling only on the training data for this configuration.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, probabilities)
        results.append((config_name, len(y_train), metrics))

        print(f"\n{'=' * 65}")
        print(config_name)
        print("=" * 65)
        print("Training sequences:", len(y_train))
        print("Training class counts:",
              np.bincount(y_train, minlength=2))

        for key in [
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "fpr"
        ]:
            print(f"{key.upper():<12}: {metrics[key]:.4f}")

        print(
            f"TN={metrics['tn']}, FP={metrics['fp']}, "
            f"FN={metrics['fn']}, TP={metrics['tp']}"
        )

    print("\n\n========== FINAL ABLATION COMPARISON ==========")
    print(
        f"{'Configuration':<27} {'Train N':>8} "
        f"{'F1':>8} {'PR-AUC':>8} {'ROC-AUC':>8} "
        f"{'Recall':>8} {'FPR':>8}"
    )

    for name, n, m in results:
        print(
            f"{name:<27} {n:>8} "
            f"{m['f1']:>8.4f} {m['pr_auc']:>8.4f} "
            f"{m['roc_auc']:>8.4f} {m['recall']:>8.4f} "
            f"{m['fpr']:>8.4f}"
        )


if __name__ == "__main__":
    main()
