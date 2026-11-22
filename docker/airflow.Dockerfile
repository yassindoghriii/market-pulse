# Airflow + environnement isolé pour les scripts MarketPulse
FROM apache/airflow:2.10.5-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Dossier du cache Hugging Face créé par l'utilisateur airflow (droits d'écriture)
RUN mkdir -p /home/airflow/.cache/huggingface

# Environnement séparé : nos bibliothèques ne modifient pas celles d'Airflow
COPY docker/requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN python -m venv /opt/airflow/pipeline-venv \
    && /opt/airflow/pipeline-venv/bin/pip install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cpu \
    && /opt/airflow/pipeline-venv/bin/pip install --no-cache-dir \
       -r /tmp/requirements-pipeline.txt
