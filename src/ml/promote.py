"""Promotion en production : registre MLflow (alias) + export du modèle pour le déploiement."""
import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

EXPERIMENT = "marketpulse-direction"
MODEL_NAME = "marketpulse-direction"
ALIAS = "production"
EXPORT_DIR = Path("models/production")

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

    # 1. Registre : on ne crée une nouvelle version que si le run change
    version = None
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        if current.run_id == run.info.run_id:
            version = current.version
    except MlflowException:
        pass
    if version is None:
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME).version
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)

    # 2. Export : fichier du modèle + fiche d'identité, lus par l'API
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True)
    model.save_model(str(EXPORT_DIR / "model.json"))
    metadata = {
        "model_name": MODEL_NAME,
        "version": str(version),
        "run_name": args.run_name,
        "run_id": run.info.run_id,
        "features": run.data.params["features"].split(","),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (EXPORT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Modèle %s version %s (alias '%s') exporté dans %s",
             MODEL_NAME, version, ALIAS, EXPORT_DIR)


if __name__ == "__main__":
    main()
