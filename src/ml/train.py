"""Entraînement du modèle hausse/baisse, comparé à la référence naïve, suivi dans MLflow."""
import logging

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier

from src.ml.dataset import FEATURES, TARGET, build_dataset

EXPERIMENT = "marketpulse-direction"
TEST_SHARE = 0.2  # 20 % des dates les plus récentes pour le test
PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, random_state=42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: round(float(v), 4) for k, v in metrics.items()}


def main() -> None:
    df = build_dataset().dropna(subset=FEATURES + [TARGET]).sort_values("date")

    # Découpage chronologique : jamais de futur dans l'entraînement
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    cutoff = dates.iloc[int(len(dates) * (1 - TEST_SHARE))]
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)
    log.info("Entraînement : %d lignes avant le %s | Test : %d lignes",
             len(train), cutoff.date(), len(test))

    # Référence naïve : toujours prédire la classe la plus fréquente de l'entraînement
    majority = int(y_train.mean() >= 0.5)
    baseline = evaluate(y_test, [majority] * len(y_test))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="xgboost_v1"):
        model = XGBClassifier(**PARAMS, eval_metric="logloss")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, (proba >= 0.5).astype(int), proba)

        mlflow.log_params({**PARAMS, "features": ",".join(FEATURES),
                           "cutoff_date": str(cutoff.date()),
                           "train_rows": len(train), "test_rows": len(test)})
        mlflow.log_metrics({f"model_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})
        importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        mlflow.log_dict(importance.round(4).to_dict(), "feature_importance.json")
        mlflow.xgboost.log_model(model, name="model")

    log.info("Référence naïve (toujours %s) : %s", "hausse" if majority else "baisse", baseline)
    log.info("Modèle XGBoost : %s", metrics)
    log.info("Importance des variables :\n%s", importance.round(3).to_string())


if __name__ == "__main__":
    main()
