"""Compatibilité (TP6) : le code de prédiction vit désormais dans sentiment_app/predict.py.

Ce fichier garde valides « python src/predict.py "texte" » et « import predict » (tests, scripts).
À l'import, il se remplace par le vrai module : les deux noms désignent alors le même objet,
si bien qu'un test qui modifie predict.MODELE modifie bien celui que l'API utilise.
"""
import importlib
import sys

_module = importlib.import_module("sentiment_app.predict")

if __name__ == "__main__":
    from sentiment_app.cli import main
    raise SystemExit(main())

sys.modules[__name__] = _module
