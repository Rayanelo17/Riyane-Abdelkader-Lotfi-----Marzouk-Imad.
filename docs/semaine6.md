# Semaine 6 : le paquet, notes du binôme

> Trois questions, quelques phrases chacune. Répondez avec ce que vous avez observé.

## 1. Pourquoi le modèle n'est-il pas dans la roue ? Comment le paquet le trouve-t-il ?

Le fichier de modèle (`sentiment.joblib`) n'est pas inclus dans la roue car il possède un cycle de vie distinct du code applicatif : il peut être réentraîné, optimisé ou remplacé fréquemment sans avoir à regénérer ni republier une nouvelle version du paquet logiciel. De plus, il s'agit d'un artefact volumineux binaire issu de l'apprentissage machine qui n'a pas sa place dans un paquet de code source (défini par `packages = ["sentiment_app", "api", "ui"]`).

Le paquet localise le modèle au moyen de la variable d'environnement `SENTIMENT_MODELE` (`os.environ.get("SENTIMENT_MODELE")`), ou replie par défaut sur le dossier local `models/sentiment.joblib` du projet si on travaille dans le dépôt. Lorsqu'il est exécuté en dehors du dépôt sans cette variable, il déclenche une exception claire (`FileNotFoundError`) avec code de retour 1 invitant l'utilisateur à renseigner le chemin du modèle dans `SENTIMENT_MODELE`.

## 2. Qu'est-ce qui a changé dans api/model.py quand predict() est entré dans le paquet, et pourquoi est-ce mieux ?

Dans `api/model.py`, toutes les altérations manuelles du chemin de recherche Python (`sys.path.insert(0, str(RACINE / "src"))`), les directives d'exclusion de linter (`# noqa: E402`) et les imports superflus de `sys` et `Path` ont été entièrement supprimés. L'API importe désormais directement le module via son nom de paquet officiel : `from sentiment_app import predict` et `from sentiment_app.predict import predict as _predict`.

C'est une amélioration architecturale majeure car le code se conforme aux standards d'empaquetage Python. Il n'y a plus de couplage fragile avec la structure arborescente locale des répertoires, plus d'effets de bord imprévus sur l'ordonnancement d'importation, et la base de code passe les vérifications strictes de `ruff` et `mypy` sans aucune exception de règle.

## 3. La roue installée hors du dépôt : qu'a-t-il fallu lui fournir, et pourquoi scikit-learn est-il épinglé dans pyproject.toml ?

Lors de l'installation de la roue dans un environnement vierge, `pip` installe de façon autonome l'ensemble des dépendances déclarées dans les métadonnées (FastAPI, uvicorn, Pydantic, joblib, scikit-learn). Cependant, le modèle n'étant pas inclus, il a fallu lui fournir explicitement le chemin absolu vers le fichier du modèle via la variable d'environnement `$env:SENTIMENT_MODELE = ".../sentiment.joblib"`.

`scikit-learn` est rigoureusement épinglé dans `pyproject.toml` (`scikit-learn>=1.5,<1.6`) et dans le workflow CI (`scikit-learn==1.5.*`) parce que les modèles sérialisés via `joblib` ou `pickle` dépendent étroitement de la représentation interne des classes de la version de scikit-learn utilisée lors de l'entraînement. Une variation de version majeure ou mineure entre la phase d'entraînement et l'inférence entraîne des erreurs critiques de chargement telles que `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`. Épingler la branche 1.5 garantit une parfaite compatibilité ascendante et descendante de l'artefact.
