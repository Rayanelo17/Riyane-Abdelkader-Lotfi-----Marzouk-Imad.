# Semaine 3 : l'API, notes du binôme

**Module :** Développement et déploiement d'applications intelligentes · **Auteur :** A. Larhlimi  
**Binôme :** Rayanelo17 & Imad Marzouk  

---

## 1. Pourquoi l'API refuse-t-elle de démarrer sans `models/sentiment.joblib`, plutôt que de démarrer et de répondre 503 ?

Refuser de démarrer sans le fichier de modèle est une application directe du principe **« Fail Fast »** (échouer tôt). 

Le modèle est la ressource centrale indispensable au service : sans lui, l'API ne peut remplir sa fonction métier. En levant immédiatement une exception `FileNotFoundError` lors du `lifespan` avant même d'ouvrir le port d'écoute HTTP, l'API signale son indisponibilité aux outils d'orchestration et de déploiement (comme Kubernetes, Docker Compose, ou la CI). Ces outils peuvent ainsi bloquer un déploiement défectueux ou relancer l'étape de construction/entraînement.

À l'inverse, si l'API démarrait et acceptait les connexions pour renvoyer des 503, le conteneur ou le processus apparaîtrait faussement « actif » auprès du réseau, tout en échouant sur toutes les requêtes utilisateur. Le code 503 est réservé aux indisponibilités temporaires survenant à l'exécution (par exemple si le modèle est déchargé dynamiquement), pas à une erreur d'installation initiale.

---

## 2. Un texte vide est refusé deux fois : par Pydantic, puis par `predict()`. Est-ce redondant ?

Non, ce n'est pas une redondance inutile : il s'agit d'une **défense en profondeur** et d'une saine **séparation des responsabilités**.

1. **Le schéma Pydantic (`PredictRequest`)** est le garde-fou au point d'entrée HTTP. Il valide le format de la requête reçue du client et renvoie une réponse standard FastAPI avec code 422 et un corps structuré (`detail` sous forme de liste avec `loc`, `msg`, `type`). Il évite d'invoquer la logique interne avec des données non conformes.
2. **La fonction `predict()` dans `src/predict.py`** est le garde-fou du composant d'inférence lui-même. Elle doit pouvoir être utilisée en tant que bibliothèque autonome (par exemple en ligne de commande `python src/predict.py`, dans des scripts de batch ou des tests unitaires) sans supposer qu'un serveur FastAPI a préalablement nettoyé les données. Elle garantit ses propres préconditions en levant une exception `ValueError("texte vide")`.
