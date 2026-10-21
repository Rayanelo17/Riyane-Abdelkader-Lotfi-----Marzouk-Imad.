"""Tests d'intégration : l'API telle qu'elle tourne vraiment (TP4, étape D).

Les tests du TP3 vérifient des réponses ; ceux-ci vérifient le démarrage de l'API
et son comportement quand le modèle manque ou tombe en panne.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
import predict as predict_module  # noqa: E402
from api.main import app  # noqa: E402

MODELE = RACINE / "models" / "sentiment.joblib"
pytestmark = pytest.mark.skipif(not MODELE.exists(), reason="modèle non produit : python src/train.py")


def test_demarrage_reel_charge_le_modele():
    """D.1 : un client ouvert par « with » exécute le lifespan."""
    with TestClient(app) as client:
        reponse = client.get("/health")
    assert reponse.status_code == 200
    assert "TP2" in reponse.json()["model"]


def test_modele_absent_refuse_demarrage(monkeypatch):
    """D.2 : modèle absent : chemin redirigé, cache vidé, l'API refuse de démarrer (FileNotFoundError)."""
    predict_module.charger_modele.cache_clear()
    monkeypatch.setattr(predict_module, "MODELE", RACINE / "models" / "inexistant.joblib")
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
    predict_module.charger_modele.cache_clear()


def test_modele_non_charge_503(monkeypatch):
    """D.3 : modèle non chargé : 503 sur /health et /predict."""
    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "classifier", None)
        r_health = client.get("/health")
        assert r_health.status_code == 503
        assert r_health.json()["detail"] == "Modèle non chargé"

        r_predict = client.post("/predict", json={"text": "Service impeccable"})
        assert r_predict.status_code == 503
        assert r_predict.json()["detail"] == "Modèle non chargé"


@pytest.mark.parametrize("exception", [RuntimeError("panne moteur"), ValueError("bug interne")])
def test_modele_en_panne_500(monkeypatch, exception):
    """D.4 : modèle en panne : 500 dans les deux cas, sans le message interne."""
    class PanneClassifier:
        model_name = "panne"

        def predict(self, text: str):
            raise exception

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "classifier", PanneClassifier())
        reponse = client.post("/predict", json={"text": "Très bon produit"})
        assert reponse.status_code == 500
        assert "panne moteur" not in reponse.text
        assert "bug interne" not in reponse.text
        assert reponse.json()["detail"] == "Erreur interne du serveur"


def test_classes_inattendues_refuse_demarrage(monkeypatch):
    """E.3 : classes inattendues dans le modèle -> l'API refuse de démarrer."""
    class DummyPipeline:
        classes_ = ["inconnu_1", "inconnu_2"]

    predict_module.charger_modele.cache_clear()
    monkeypatch.setattr(predict_module, "charger_modele", lambda: DummyPipeline())
    with pytest.raises(ValueError, match="Classe inattendue"):
        with TestClient(app):
            pass

