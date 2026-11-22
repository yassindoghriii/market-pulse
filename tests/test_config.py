"""Tests de la configuration commune."""
from src.config import TICKERS


def test_dix_actions_suivies():
    assert len(TICKERS) == 10


def test_chaque_action_a_un_nom():
    assert all(isinstance(name, str) and name for name in TICKERS.values())


def test_six_actions_de_paris():
    assert len([t for t in TICKERS if t.endswith(".PA")]) == 6
