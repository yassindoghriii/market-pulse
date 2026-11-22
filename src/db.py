"""Connexions aux bases de données, à partir des variables d'environnement (.env)."""
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# En local : 127.0.0.1 ; dans Docker : nom du service (postgres, mongo)
PG_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
MONGO_HOST = os.getenv("MONGO_HOST", "127.0.0.1")


def pg_conninfo() -> str:
    """Chaîne de connexion à PostgreSQL (psycopg)."""
    return (
        f"host={PG_HOST} port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def pg_engine():
    """Connexion SQLAlchemy à PostgreSQL (utilisée par pandas.read_sql)."""
    from sqlalchemy import create_engine
    return create_engine(
        f"postgresql+psycopg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{PG_HOST}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )


def mongo_db():
    """Base MongoDB du projet."""
    client = MongoClient(
        host=MONGO_HOST, port=int(os.environ["MONGO_PORT"]),
        username=os.environ["MONGO_USER"], password=os.environ["MONGO_PASSWORD"],
        authSource="admin",
    )
    return client["marketpulse"]
