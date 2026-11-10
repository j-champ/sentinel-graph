import numpy as np

from pathlib import Path
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

from preprocessing_chronological import prepare_data


def evaluate(name, predictions, probabilities, y_true):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"FPR      : {fpr:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Ranking metrics require both classes.
    if len(np.unique(y_true)) == 2:
        print(f"ROC-AUC  : {roc_auc_score(y_true, probabilities):.4f}")
        print(f"PR-AUC   : {average_precision_score(y_true, probabilities):.4f}")
    else:
        print("ROC-AUC  : undefined (only one class)")
        print("PR-AUC   : undefined (only one class)")


def evaluate_probabilities(name, probabilities, y_true, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)
    evaluate(
        f"{name} (threshold={threshold:.2f})",
        predictions,
        probabilities,
        y_true,
    )


def main():
    (
        X_train, y_train, intensity_train,
        X_val, y_val, intensity_val,
        X_test, y_test, intensity_test,
        scaler,
    ) = prepare_data()

    print("\nTraining baseline models...")

    # 1. Majority baseline
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)

    for name, X, y in [
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        probabilities = majority.predict_proba(X)[:, 1]
        predictions = majority.predict(X)
        evaluate(f"Majority - {name}", predictions, probabilities, y)

    # 2. Persistence baseline
    split_dir = Path("data/splits_chronological")

    for split, y_true in [
        ("val", y_val),
        ("test", y_test),
    ]:
        predictions = []
        targets = []

        for file in sorted((split_dir / split).glob("*.npz")):
            with np.load(file) as data:
                predictions.extend(data["last_attack"].astype(int))
                targets.extend(data["y_attack"])

        persistence_pred = np.asarray(predictions)
        persistence_y = np.asarray(targets)

        if not np.array_equal(persistence_y, y_true):
            raise ValueError(
                f"Persistence targets do not match {split} targets."
            )

        evaluate(
            f"Persistence - {split.upper()}",
            persistence_pred,
            persistence_pred.astype(float),
            y_true,
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

    for name, X, y in [
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        probabilities = logistic.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        evaluate(f"Logistic Regression - {name}", predictions, probabilities, y)

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

    val_prob = xgb.predict_proba(X_val)[:, 1]
    test_prob = xgb.predict_proba(X_test)[:, 1]

    evaluate_probabilities("XGBoost - Validation", val_prob, y_val)
    evaluate_probabilities("XGBoost - Test", test_prob, y_test)

    print("\n========== PROBABILITY DISTRIBUTIONS ==========")

    for name, probabilities in [
        ("XGBoost validation", val_prob),
        ("XGBoost test", test_prob),
    ]:
        print(f"\n{name}")
        print(f"Min:    {probabilities.min():.4f}")
        print(f"Max:    {probabilities.max():.4f}")
        print(f"Mean:   {probabilities.mean():.4f}")
        print(f"Median: {np.median(probabilities):.4f}")


if __name__ == "__main__":
    main()
