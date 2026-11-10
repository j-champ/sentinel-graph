from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_ROOT = Path("data")

TRAIN_DAYS = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.npz",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.npz",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.npz",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.npz",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.npz",
]

TEST_DAY = "Friday-23-02-2018_TrafficForML_CICFlowMeter.npz"


def load_days(folder, days):
    sequences, targets = [], []

    for day in days:
        path = DATA_ROOT / folder / day

        with np.load(path) as data:
            X = data["X"].astype(np.float32)
            y_attack = data["y_attack"].astype(int)
            last_attack = data["last_attack"].astype(int)

        y_onset = ((y_attack == 1) & (last_attack == 0)).astype(int)

        risk_set = last_attack == 0
        X = X[risk_set]
        y_onset = y_onset[risk_set]
        sequences.append(X)
        targets.append(y_onset)

        print(
            f"{day}: eligible_samples={len(y_onset)}, "
            f"onsets={int(y_onset.sum())}"
        )

    return np.concatenate(sequences), np.concatenate(targets)


def representations(X):
    return {
        "Last window": X[:, -1, :],
        "Five-minute mean": X.mean(axis=1),
        "First-to-last delta": X[:, -1, :] - X[:, 0, :],
    }


def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate(name, model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * 48)
    print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y, predictions, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y, probabilities):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, probabilities):.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main():
    print("Loading onset-forecasting data...")

    X_train_seq, y_train = load_days("sequences_continuous", TRAIN_DAYS)
    X_test_seq, y_test = load_days("sequences_expanded", [TEST_DAY])

    print("\nDATA SUMMARY")
    print(f"Train: {X_train_seq.shape}, onsets={int(y_train.sum())}")
    print(f"Test : {X_test_seq.shape}, onsets={int(y_test.sum())}")
    print("Target: attack onset in the next 30-second state")
    print("Prediction threshold: 0.5")

    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError("Both splits must contain positive onset examples.")

    train_representations = representations(X_train_seq)
    test_representations = representations(X_test_seq)

    for representation in train_representations:
        X_train = train_representations[representation]
        X_test = test_representations[representation]

        print("\n" + "=" * 60)
        print(f"FEATURE REPRESENTATION: {representation}")
        print(f"Feature count: {X_train.shape[1]}")
        print("=" * 60)

        for model_name, model in make_models().items():
            print(f"\nFitting {model_name}...")
            model.fit(X_train, y_train)
            evaluate(model_name, model, X_test, y_test)

    leave_one_day_out()



def leave_one_day_out():
    print("\\n" + "=" * 60)
    print("LEAVE-ONE-DAY-OUT ANALYSIS")
    print("Model: Logistic Regression")
    print("Representation: First-to-last delta")
    print("=" * 60)

    day_data = {}

    for day in TRAIN_DAYS:
        X, y = load_days("sequences_continuous", [day])
        day_data[day] = (X[:, -1, :] - X[:, 0, :], y)

    for held_out_day, (X_test, y_test) in day_data.items():
        training_days = [
            day for day in TRAIN_DAYS if day != held_out_day
        ]

        X_train = np.concatenate([
            day_data[day][0] for day in training_days
        ])
        y_train = np.concatenate([
            day_data[day][1] for day in training_days
        ])

        print(f"\\nHeld-out day: {held_out_day}")
        print(
            f"Training samples={len(y_train)}, "
            f"training onsets={int(y_train.sum())}"
        )
        print(
            f"Test samples={len(y_test)}, "
            f"test onsets={int(y_test.sum())}"
        )

        if len(y_test) == 0:
            print("Skipped: no eligible benign-state sequences.")
            continue

        if len(np.unique(y_train)) < 2:
            print("Skipped: training data contains only one class.")
            continue

        model = make_models()["Logistic Regression"]
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
        print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.4f}")
        print(f"PR-AUC:    {average_precision_score(y_test, probabilities):.4f}")

        if len(np.unique(y_test)) == 2:
            print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
        else:
            print("ROC-AUC:   undefined (held-out day has one class)")



if __name__ == "__main__":
    main()
