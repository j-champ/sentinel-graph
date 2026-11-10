
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from preprocessing_continuous import load_split


def evaluate(name, predictions, probabilities, y):
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, probabilities):.4f}")
    print(f"PR-AUC   : {average_precision_score(y, probabilities):.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main():
    from preprocessing_continuous import prepare_data
    X_train, y_train, _, _, _, _, _, _, _, scaler = prepare_data()

    print("Training XGBoost...")
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

    test_dir = Path("data/splits_continuous/test")

    for file in sorted(test_dir.glob("*.npz")):
        with np.load(file) as data:
            X = scaler.transform(data["X"].reshape(len(data["X"]), -1))
            y = data["y_attack"]
            persistence = data["last_attack"].astype(int)

        print("\n" + "=" * 60)
        print(file.name)
        print("=" * 60)

        print("Sequences:", len(y))
        print("Class distribution:", np.bincount(y, minlength=2))

        if len(np.unique(y)) < 2:
            print("Skipping ROC-AUC and PR-AUC: only one class present.")
            continue

        xgb_prob = model.predict_proba(X)[:, 1]
        xgb_pred = (xgb_prob >= 0.50).astype(int)

        evaluate("Persistence", persistence, persistence.astype(float), y)
        evaluate("XGBoost", xgb_pred, xgb_prob, y)


if __name__ == "__main__":
    main()

