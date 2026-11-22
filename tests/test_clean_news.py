"""Tests du nettoyage des articles."""
import pytest

from src.transform.clean_news import NOISE, clean_title


def test_retire_le_nom_du_journal():
    assert clean_title("Apple beats estimates - Reuters", "Reuters") == "Apple beats estimates"


def test_garde_le_titre_si_la_source_ne_correspond_pas():
    assert clean_title("Apple - the comeback", "Reuters") == "Apple - the comeback"


def test_titre_sans_source():
    assert clean_title("  Tesla rallies  ", None) == "Tesla rallies"


@pytest.mark.parametrize("title", [
    "TotalEnergies SE (TTE.PA) Stock Price, News, Quote & History",
    "MC Stock Price and Chart — EURONEXT:MC",
])
def test_pages_de_cotation_detectees_comme_bruit(title):
    assert NOISE.search(title)


@pytest.mark.parametrize("title", [
    "Nvidia shares fall 14% after earnings",
    "Airbus raises delivery target",
])
def test_vrais_articles_conserves(title):
    assert not NOISE.search(title)
