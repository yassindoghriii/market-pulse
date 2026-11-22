"""Connexions aux bases de données, à partir des variables du fichier .env."""
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def pg_conninfo() -> str:
    """Chaîne de connexion à PostgreSQL."""
    return (
        f"host=127.0.0.1 port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def mongo_db():
    """Base MongoDB du projet."""
    client = MongoClient(
        host="127.0.0.1", port=int(os.environ["MONGO_PORT"]),
        username=os.environ["MONGO_USER"], password=os.environ["MONGO_PASSWORD"],
        authSource="admin",
    )
    return client["marketpulse"]


def pg_engine():
    """Connexion SQLAlchemy à PostgreSQL (utilisée par pandas.read_sql)."""
    from sqlalchemy import create_engine
    return create_engine(
        f"postgresql+psycopg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@127.0.0.1:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
