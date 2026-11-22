"""Tests de l'API (sans base de données ni modèle)."""
from fastapi.testclient import TestClient

from src.api.main import app

# Sans "with", le démarrage (chargement du modèle) n'est pas exécuté :
# on teste uniquement les routes et les contrôles qui n'en dépendent pas.
client = TestClient(app)


def test_health_repond():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_liste_des_actions():
    response = client.get("/tickers")
    assert response.status_code == 200
    assert len(response.json()) == 10


def test_action_inconnue_renvoie_404():
    response = client.get("/predict/XYZ")
    assert response.status_code == 404
    assert "Actions suivies" in response.json()["detail"]


def test_parametre_days_invalide_renvoie_422():
    response = client.get("/prices/MC.PA?days=0")
    assert response.status_code == 422


def test_metriques_exposees():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "marketpulse_last_price_age_days" in response.text
