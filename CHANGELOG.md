# Journal des versions

Toutes les modifications notables de ce projet sont consignées dans ce fichier.

## [1.0.0] - 2026-11-09

### Ajouté
- Paquet Python standard `sentiment-app` prêt à être distribué (roue `wheel` et distribution source `sdist`).
- Modèle de classification TF-IDF entraîné et embarqué dans le paquet sous `src/sentiment_app/modele/sentiment.joblib`.
- Interface publique `sentiment_app.predict(texte: str) -> Resultat`.
- Commande CLI `sentiment "texte" [--json] [--version]`.
- Script d'audit et vérification automatique `scripts/verifier_livrable.py`.
- Suite de tests d'isolation `tests/test_paquet.py`.

### Limites connues
- Vocabulaire entraîné sur les données de démonstration du projet en français.
- Texte vide ou composé uniquement d'espaces rejeté avec `ValueError("texte vide")`.

## [0.5.0] - 2026-10-19

- Page Streamlit : un avis, ou un fichier d'avis ; le client HTTP traduit chaque réponse de l'API.

## [0.4.0] - 2026-10-19

- Contrat d'entrée testé, abstention `INDETERMINE`, jeu d'or, tests d'intégration, contrat typé, ruff, mypy, pre-commit.

## [0.3.0] - 2026-10-19

- FastAPI : `/health`, `/predict`, contrat Pydantic ; la CI entraîne le modèle avant les tests.

## [0.2.0] - 2026-10-19

- TF-IDF et régression logistique ; les métriques sont versionnées, le modèle ne l'est pas.

## [0.1.0] - 2026-10-05

- Environnement Python 3.11, structure du dépôt, trois fiches de cadrage, dépôt GitHub à deux auteurs.
