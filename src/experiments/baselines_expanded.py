import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing_expanded import prepare_data


def evaluate(name, predictions, probabilities, y_test):
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * 45)
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, probabilities):.4f}")
    print(f"FPR      : {fp / (fp + tn) if tn + fp else 0:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main():
    (
        X_train, y_train, _, X_val, y_val, _,
        X_test, y_test, _, _
    ) = prepare_data()

    print("\nTraining expanded baseline models...")

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    evaluate(
        "Majority Baseline",
        majority.predict(X_test),
        majority.predict_proba(X_test)[:, 1],
        y_test,
    )

    predictions, targets = [], []
    from pathlib import Path

    for file in sorted(Path("data/splits_expanded/test").glob("*.npz")):
        with np.load(file) as data:
            predictions.extend(data["last_attack"].astype(int))
            targets.extend(data["y_attack"])

    persistence_pred = np.asarray(predictions)
    if not np.array_equal(np.asarray(targets), y_test):
        raise ValueError("Persistence targets do not match test targets.")

    evaluate(
        "Persistence Baseline",
        persistence_pred,
        persistence_pred.astype(float),
        y_test,
    )

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )),
    ])
    logistic.fit(X_train, y_train)
    evaluate(
        "Logistic Regression",
        logistic.predict(X_test),
        logistic.predict_proba(X_test)[:, 1],
        y_test,
    )

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

    # Validation is reported separately; test remains untouched.
    val_prob = xgb.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.50).astype(int)
    evaluate("XGBoost — Validation (threshold 0.50)", val_pred, val_prob, y_val)

    test_prob = xgb.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.50).astype(int)
    evaluate("XGBoost — Test (threshold 0.50)", test_pred, test_prob, y_test)


if __name__ == "__main__":
    main()
