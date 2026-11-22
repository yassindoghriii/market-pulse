"""Chaîne quotidienne MarketPulse : collecte -> chargement -> transformation -> sentiment."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# Chaque tâche lance un module du projet avec l'environnement Python isolé
RUN = "cd /opt/airflow/project && /opt/airflow/pipeline-venv/bin/python -m"

default_args = {
    "owner": "marketpulse",
    "retries": 2,                         # 2 nouvelles tentatives en cas d'échec
    "retry_delay": timedelta(minutes=5),  # 5 minutes d'attente entre deux tentatives
}

with DAG(
    dag_id="marketpulse_daily",
    description="Collecte, chargement, transformation et sentiment, chaque soir de bourse",
    schedule="0 22 * * 1-5",   # 22 h UTC du lundi au vendredi (après la clôture de New York)
    start_date=datetime(2026, 11, 1),
    catchup=False,             # ne pas rattraper les jours passés
    default_args=default_args,
    tags=["marketpulse"],
) as dag:
    collect_prices = BashOperator(task_id="collect_prices",
                                  bash_command=f"{RUN} src.collect.prices")
    collect_news = BashOperator(task_id="collect_news",
                                bash_command=f"{RUN} src.collect.news")
    load_raw = BashOperator(task_id="load_raw",
                            bash_command=f"{RUN} src.load.load_raw")
    build_price_features = BashOperator(task_id="build_price_features",
                                        bash_command=f"{RUN} src.transform.build_price_features")
    clean_news = BashOperator(task_id="clean_news",
                              bash_command=f"{RUN} src.transform.clean_news")
    score_sentiment = BashOperator(task_id="score_sentiment",
                                   bash_command=f"{RUN} src.ml.sentiment")

    # Ordre d'exécution
    [collect_prices, collect_news] >> load_raw
    load_raw >> [build_price_features, clean_news]
    clean_news >> score_sentiment
