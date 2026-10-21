"""Tests aux limites du contrat d'entrée (TP4, étape B).

PredictRequest promet un champ « text », chaîne de 1 à 2000 caractères.
Ce fichier vérifie cette promesse à ses frontières, puis au hasard (hypothesis).
"""
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

MODELE = Path(__file__).resolve().parents[1] / "models" / "sentiment.joblib"
pytestmark = pytest.mark.skipif(not MODELE.exists(), reason="modèle non produit : python src/train.py")


@pytest.fixture(scope="module")
def client():
    """Portée « module » : l'API démarre une fois pour tout le fichier (hypothesis l'exige)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:  # le « with » exécute le lifespan : le modèle est chargé
        yield c


def en_json(valeur: object) -> bytes:
    """Corps de requête JSON, sous forme d'octets : permet aussi d'envoyer un corps invalide."""
    return json.dumps(valeur).encode("utf-8")


# Étape B.1 : huit cas au moins, chacun avec son code attendu.
CAS = [
    pytest.param(en_json({"text": ""}), 422, id="texte-vide"),
    pytest.param(en_json({"text": "   "}), 422, id="texte-espaces"),
    pytest.param(en_json({"text": "a"}), 200, id="un-seul-caractere"),
    pytest.param(en_json({"text": "a" * 2000}), 200, id="exactement-2000-caracteres"),
    pytest.param(en_json({"text": "a" * 2001}), 422, id="2001-caracteres"),
    pytest.param(en_json({"autre": "valeur"}), 422, id="champ-absent"),
    pytest.param(en_json({"text": 42}), 422, id="champ-autre-type"),
    pytest.param(b"pas du json", 422, id="corps-non-json"),
]


@pytest.mark.parametrize("corps, code_attendu", CAS)
def test_frontieres_du_contrat(client, corps, code_attendu):
    reponse = client.post("/predict", content=corps, headers={"Content-Type": "application/json"})
    assert reponse.status_code == code_attendu
    # Étape B.2 : pour chaque 422, vérifiez aussi la FORME de la réponse :
    # « detail » doit être la liste d'erreurs standard de FastAPI (objets avec loc, msg, type).
    if code_attendu == 422:
        data = reponse.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        for err in data["detail"]:
            assert "loc" in err
            assert "msg" in err
            assert "type" in err
    elif code_attendu == 200:
        data = reponse.json()
        assert data["label"] in {"POSITIVE", "NEGATIVE", "INDETERMINE"}
        assert 0.0 <= data["score"] <= 1.0


# Étape B.5 : deux tests de propriété avec hypothesis
@settings(deadline=None)
@given(st.text(min_size=1, max_size=2000).filter(lambda s: bool(s.strip())))
def test_propriete_texte_valide(client, texte: str):
    reponse = client.post("/predict", json={"text": texte})
    assert reponse.status_code == 200
    data = reponse.json()
    assert data["label"] in {"POSITIVE", "NEGATIVE", "INDETERMINE"}
    assert 0.0 <= data["score"] <= 1.0


@settings(deadline=None)
@given(st.text(alphabet=" \t\n\r", min_size=1, max_size=2000))
def test_propriete_texte_blancs(client, texte: str):
    reponse = client.post("/predict", json={"text": texte})
    assert reponse.status_code == 422
    data = reponse.json()
    assert isinstance(data["detail"], list)
    for err in data["detail"]:
        assert "loc" in err
        assert "msg" in err
        assert "type" in err
