"""Connexions aux bases de données, à partir des variables du fichier .env."""
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def pg_conninfo() -> str:
    """Chaîne de connexion à PostgreSQL."""
    return (
        f"host=localhost port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def mongo_db():
    """Base MongoDB du projet."""
    client = MongoClient(
        host="localhost", port=27017,
        username=os.environ["MONGO_USER"], password=os.environ["MONGO_PASSWORD"],
        authSource="admin",
    )
    return client["marketpulse"]
