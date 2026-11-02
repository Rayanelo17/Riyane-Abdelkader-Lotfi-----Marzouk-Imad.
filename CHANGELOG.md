# Journal des versions

Le numéro de version suit les TP : 0.N.0 à la fin du TP N. Chaque version publiée porte une étiquette Git `v0.N.0`.

## 0.6.0 (TP6) : le paquet

- **Pour celui qui installe** : l'application est distribuable sous forme d'une roue wheel Python (`sentiment_app-0.6.0-py3-none-any.whl`) installable avec `pip`. Elle fournit la commande `sentiment` (`--version`, `predict`, `serve`, `ui`). Le modèle externe est spécifié via la variable d'environnement `SENTIMENT_MODELE`.
- **Pour celui qui développe** : installation en mode éditable (`pip install -e .`). Intégration de `predict.py` dans le paquet `sentiment_app`, suppression des manipulations de `sys.path` dans l'API (`api/model.py`), et maintien de la rétrocompatibilité via le fichier pont `src/predict.py`.
- **Pour la CI** : scikit-learn est épinglé (`scikit-learn==1.5.*`) pour assurer la cohérence de chargement du modèle, et une étape de validation installe la roue construite dans un environnement virtuel isolé hors du dépôt et exécute un test de prédiction en ligne de commande.

## 0.5.0 (TP5) : l'interface

- Page Streamlit : un avis, ou un fichier d'avis ; le client HTTP traduit chaque réponse de l'API.

## 0.4.0 (TP4) : tests et qualité

- Contrat d'entrée testé, abstention `INDETERMINE`, jeu d'or, tests d'intégration, contrat typé, ruff, mypy, pre-commit.

## 0.3.0 (TP3) : l'API

- FastAPI : `/health`, `/predict`, contrat Pydantic ; la CI entraîne le modèle avant les tests.

## 0.2.0 (TP2) : le modèle

- TF-IDF et régression logistique ; les métriques sont versionnées, le modèle ne l'est pas.

## 0.1.0 (TP1) : le cadrage

- Environnement Python 3.11, structure du dépôt, trois fiches de cadrage, dépôt GitHub à deux auteurs.
