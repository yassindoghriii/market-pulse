"""Enregistre un run dans le registre MLflow et lui donne l'alias 'production'."""
import argparse
import logging

import mlflow
from mlflow import MlflowClient

EXPERIMENT = "marketpulse-direction"
MODEL_NAME = "marketpulse-direction"
ALIAS = "production"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="xgboost_v2_sans_sentiment")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{args.run_name}'",
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    if not runs:
        raise SystemExit(f"Aucun run nommé {args.run_name}")
    run = runs[0]

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    log.info("Run %s (%s) enregistré comme %s version %s, alias '%s'",
             args.run_name, run.info.run_id[:8], MODEL_NAME, version.version, ALIAS)


if __name__ == "__main__":
    main()
