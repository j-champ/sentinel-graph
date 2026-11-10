
import numpy as np

from sklearn.dummy import DummyClassifier
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

from preprocessing_continuous import prepare_data


def evaluate(name, predictions, probabilities, y_test):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"FPR      : {fpr:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main():
    (
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
    ) = prepare_data()

    print("\nTraining baseline models...")

    # 1. Majority baseline
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)

    majority_pred = majority.predict(X_test)
    majority_prob = majority.predict_proba(X_test)[:, 1]

    evaluate(
        "Majority Baseline",
        majority_pred,
        majority_prob,
        y_test,
    )

    # 2. Persistence baseline
    from pathlib import Path

    split_dir = Path("data/splits_continuous/test")
    predictions = []
    targets = []

    for file in sorted(split_dir.glob("*.npz")):
        with np.load(file) as data:
            predictions.extend(data["last_attack"].astype(int))
            targets.extend(data["y_attack"])

    persistence_pred = np.asarray(predictions)
    persistence_y = np.asarray(targets)

    if not np.array_equal(persistence_y, y_test):
        raise ValueError("Persistence targets do not match test targets.")

    evaluate(
        "Persistence Baseline",
        persistence_pred,
        persistence_pred.astype(float),
        y_test,
    )

    # 3. Logistic Regression
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    logistic.fit(X_train, y_train)

    logistic_pred = logistic.predict(X_test)
    logistic_prob = logistic.predict_proba(X_test)[:, 1]

    evaluate(
        "Logistic Regression",
        logistic_pred,
        logistic_prob,
        y_test,
    )

    # 4. XGBoost
    xgb = XGBClassifier(
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

    xgb.fit(X_train, y_train)

    test_prob = xgb.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.50).astype(int)

    print("\nXGBoost classification threshold: 0.50")
    print("Validation metrics omitted: validation set has only one class.")

    evaluate(
        "XGBoost",
        test_pred,
        test_prob,
        y_test,
    )


if __name__ == "__main__":
    main()

