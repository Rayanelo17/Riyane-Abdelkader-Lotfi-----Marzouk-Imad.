"""Tests du paquet installé."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Ces tests visent le paquet INSTALLÉ. Dans le dépôt, l'API ajoute src/ au chemin d'import :
# sentiment_app y serait trouvé dans les sources, ce qui ne prouve rien. On les ignore donc,
# sauf quand scripts/verifier_livrable.py les lance (VERIFICATION_LIVRABLE=1) : là, ils doivent passer.
_spec = importlib.util.find_spec("sentiment_app")
if not (_spec and "site-packages" in (_spec.origin or "")) and not os.environ.get("VERIFICATION_LIVRABLE"):
    pytest.skip("paquet non installé : python scripts/verifier_livrable.py", allow_module_level=True)

import sentiment_app  # noqa: E402

# La commande est installée à côté de l'interpréteur qui lance les tests (bin/ sous Linux, Scripts sous Windows).
# On ne compte pas sur le PATH : il dépend de l'activation de l'environnement.
COMMANDE = shutil.which("sentiment", path=str(Path(sys.executable).parent))


def test_importe_depuis_le_paquet_installe():
    assert "site-packages" in sentiment_app.__file__, "le test lit les sources, pas la roue installée"


def test_version():
    assert sentiment_app.__version__ == "1.0.0"


def test_predict_positif():
    r = sentiment_app.predict("Livraison rapide, produit conforme, très satisfait")
    assert r["label"] == "positive"
    assert 0.5 <= r["score"] <= 1.0


def test_predict_negatif():
    r = sentiment_app.predict("Produit cassé et service injoignable, à fuir")
    assert r["label"] == "negative"
    assert 0.5 <= r["score"] <= 1.0


def test_predict_texte_vide():
    with pytest.raises(ValueError, match="texte vide"):
        sentiment_app.predict("")


def test_commande_installee():
    assert COMMANDE is not None, "commande sentiment introuvable"
    res = subprocess.run(
        [COMMANDE, "--json", "Livraison rapide, produit conforme, très satisfait"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert res.returncode == 0
    corps = json.loads(res.stdout)
    assert corps["label"] == "positive"
