"""La fonction predict() du TP 1, devenue l'interface publique du paquet."""
import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

import joblib

RACINE = Path(__file__).resolve().parents[2]
MODELE_DEFAUT = RACINE / "models" / "sentiment.joblib"
MODELE = MODELE_DEFAUT


class Resultat(TypedDict):
    label: str
    score: float


def chemin_modele() -> Path:
    env = os.environ.get("SENTIMENT_MODELE")
    if env:
        return Path(env)
    if MODELE != MODELE_DEFAUT:
        return Path(MODELE)
    try:
        pkg_model = files("sentiment_app").joinpath("modele/sentiment.joblib")
        p = Path(str(pkg_model))
        if p.exists():
            return p
    except Exception:
        pass
    return Path(MODELE)


@lru_cache(maxsize=1)
def charger_modele():
    """Le modèle est lu dans le paquet installé, où qu'il soit sur le disque."""
    p = chemin_modele()
    if p != MODELE_DEFAUT and not p.exists():
        raise FileNotFoundError(f"Modèle introuvable : {p}")

    if not p.exists():
        try:
            with files("sentiment_app").joinpath("modele/sentiment.joblib").open("rb") as f:
                return joblib.load(f)
        except Exception:
            raise FileNotFoundError(f"Modèle introuvable : {p}") from None

    with open(p, "rb") as f:
        return joblib.load(f)


def predict(texte: str) -> Resultat:
    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("texte vide")
    modele = charger_modele()
    return {
        "label": str(modele.predict([texte])[0]),
        "score": round(float(modele.predict_proba([texte]).max()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    from sentiment_app.cli import main as _cli_main
    return _cli_main(argv)
