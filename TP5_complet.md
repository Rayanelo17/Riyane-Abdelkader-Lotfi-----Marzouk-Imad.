# TP 5 : MLflow et DVC, version complète

Projet fil rouge : `sentiment-app`. Ce document réunit le déroulé du TP, les commandes et les explications. Il indique aussi, pour chaque opération, où elle est présentée dans les autres ressources du cours et où se trouve le code.

**Durée et travail.** Travail individuel. Le livrable est décrit à la section 8.

---


---

## 1. Carte des opérations

| # | Opération | Présentée dans | Code | Commande principale |
|---|---|---|---|---|
| 1 | Installer les dépendances | README §1 ; énoncé §2 | `requirements.txt` | `pip install -r requirements.txt` |
| 2 | Définir la base de suivi MLflow | README §1 ; énoncé §2 | variable `MLFLOW_TRACKING_URI`, lue par MLflow | `export MLFLOW_TRACKING_URI=sqlite:///mlflow.db` |
| 3 | Initialiser Git et DVC | README §2 ; énoncé §5.1 ; support, « Versionner les données avec DVC » ; présentation, diapositive 9 | dossier `.dvc/` | `git init`, `dvc init` |
| 4 | Générer le jeu de données | README §3 ; énoncé §5.2 | `src/make_demo.py` : `generer` (l. 43), `main` (l. 71) | `python src/make_demo.py` |
| 5 | Suivre le CSV avec DVC | README §3 ; énoncé §5.2 ; support, « Versionner les données avec DVC » | `data/demo_reviews.csv.dvc`, `data/.gitignore` | `dvc add data/demo_reviews.csv` |
| 6 | Configurer un remote et envoyer les données | README §3 ; énoncé §5.3 ; support, « Versionner les données avec DVC » | `.dvc/config` | `dvc remote add -d`, `dvc push` |
| 7 | Nettoyer, dédoublonner, découper | README §4 ; énoncé §6.1 ; support, « Pipelines reproductibles » | `src/prepare.py` : `nettoyer` (l. 19), `preparer` (l. 26) | `python src/prepare.py data/demo_reviews.csv` |
| 8 | Entraîner et journaliser un run | énoncé §3.1 et §3.2 ; README §5 ; support, « Suivi des expériences et registre de modèles » ; présentation, diapositive 5 | `src/train.py` : `construire_pipeline` (l. 33), `entrainer` (l. 43), `main` (l. 52), journalisation (l. 81 à 86) | `python src/train.py` |
| 9 | Enregistrer params, métrique, artefact et modèle | énoncé §3.1 ; README §5 ; support, « Tracking des hyperparamètres et des métriques » ; présentation, diapositive 5 | `src/train.py` l. 81 à 86 : `log_param`, `log_metric`, `log_artifact`, `log_model` | voir l'opération 8 |
| 10 | Faire varier un paramètre (expérience) | énoncé §3.2 et §6.3 ; README §5 ; support, « Expériences DVC » | `params.yaml`, option `--set-param` | `dvc exp run --set-param train.ngram_max=3` |
| 11 | Comparer les essais dans MLflow | énoncé §3.2 ; README §5 ; support, « Suivi des expériences » | interface MLflow ; script de comparaison (README §5) | `mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"` |
| 12 | Sélectionner le meilleur run et promouvoir | énoncé §4.1 ; README §6 ; support, « Le registre de modèles » ; présentation, diapositives 7 et 8 | `src/promote.py` : `promouvoir` (l. 21), `search_runs` (l. 32), `register_model` (l. 47), alias (l. 50) | `python src/promote.py` |
| 13 | Tester le refus sous le seuil | énoncé §4.2 ; README §6 | `src/promote.py`, `main` (l. 54) ; seuil dans `params.yaml`, section `promote` | modifier `min_f1`, puis `python src/promote.py` |
| 14 | Charger et prédire avec le modèle | README §7 ; support, « Charger le modèle servi » | `src/predict.py` : `charger` (l. 22), `charger_registre` (l. 27), `predire` (l. 34), `main` (l. 39) | `python src/predict.py --registre "..."` |
| 15 | Décrire le pipeline | énoncé §6.1 ; README §4 ; support, « Déclarer un pipeline » ; présentation, diapositive 10 | `dvc.yaml` (étapes `prepare` et `train`) | `dvc dag` |
| 16 | Relancer le pipeline sélectivement | énoncé §6.2 ; README §4 ; support, « Exécution sélective avec dvc repro » | `dvc.yaml`, `dvc.lock` | `dvc repro` |
| 17 | Lancer l'entraînement avec MLflow Projects | énoncé §8 bis ; README §10 ; support, « Dictionnaire » (entrées MLproject et python_env.yaml) ; présentation, diapositive 12 | `MLproject`, `python_env.yaml`, option `--ngram-max` de `src/train.py` | `mlflow run . -e train -P ngram_max=3 --experiment-name sentiment-app --env-manager local` |
| 18 | Tester le projet | énoncé §7 ; README §8 | `tests/` (30 tests) | `pytest -q` |
| 19 | Vérifier la cohérence des fichiers | README §8 ; `tests/test_pipeline_files.py` | `dvc.yaml`, `params.yaml`, `MLproject`, `python_env.yaml` | `pytest tests/test_pipeline_files.py` |
| 20 | Retrouver la définition d'un terme | support, « Dictionnaire » | aucun | aucune |

---

## 2. Déroulé du TP

### Étape 1 : préparer l'environnement

Créez un environnement virtuel, puis installez les dépendances. Le fichier `requirements.txt` fixe les versions minimales. MLflow doit être en version 3.x.

```bash
cd sentiment-app
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Définissez la base de suivi MLflow. Sans cette variable, MLflow écrit dans le dossier `mlruns/` et les commandes suivantes ne retrouvent pas les runs.

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db    # PowerShell : $env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

**Vérification.** `python -c "import mlflow, dvc, sklearn; print(mlflow.__version__, sklearn.__version__)"` affiche deux versions sans erreur.

### Étape 2 : initialiser Git et DVC

DVC s'appuie sur Git. Le dépôt doit donc exister avant `dvc init`.

```bash
git init
git add .
git commit -m "Projet initial"
dvc init
git add .dvc .dvcignore
git commit -m "Initialisation de DVC"
```

**Explication.** `dvc init` crée `.dvc/`, qui contient la configuration et le cache. La configuration est versionnée dans Git. Le cache ne l'est pas.

**Piège.** Si le CSV est déjà suivi par Git, `dvc add` refuse de le prendre en charge. Retirez-le de Git d'abord :

```bash
git rm --cached data/demo_reviews.csv
git commit -m "Retirer le CSV de Git"
```

### Étape 3 : générer et suivre les données

```bash
python src/make_demo.py
```

Sortie attendue : `290 avis écrits dans .../data/demo_reviews.csv`.

**Explication.** La graine fixée dans `src/make_demo.py` rend le jeu toujours identique. Les avis mixtes (un éloge et une critique) reçoivent le label `neutre`.

Suivez le fichier avec DVC, puis versionnez le petit fichier `.dvc` dans Git :

```bash
dvc add data/demo_reviews.csv
git add data/demo_reviews.csv.dvc data/.gitignore
git commit -m "Données versionnées avec DVC"
```

**Explication.** `dvc add` calcule le hash du CSV, le place dans le cache, et écrit le hash dans `data/demo_reviews.csv.dvc`. Git suit ce fichier, pas les données.

Configurez un stockage distant local, puis envoyez les données :

```bash
mkdir -p /tmp/dvc-remote
dvc remote add -d monstockage /tmp/dvc-remote
git add .dvc/config
git commit -m "Stockage distant DVC"
dvc push
```

**Explication.** `dvc push` copie le cache vers le remote. `dvc pull` fait l'inverse, sur un autre poste par exemple. Le remote est un dossier ici. En projet réel, ce serait un bucket S3, un serveur SSH ou un autre stockage partagé.

### Étape 4 : nettoyer, découper et entraîner

Le pipeline `dvc.yaml` décrit deux étapes. Lancez-les avec une seule commande :

```bash
dvc repro
```

DVC exécute `prepare`, puis `train`, dans l'ordre déduit des dépendances.

- `prepare` lance `python src/prepare.py data/demo_reviews.csv`. Il supprime les avis vides et les doublons, puis découpe en jeux d'entraînement et de test de façon stratifiée. Sortie attendue : `train : 217 avis, test : 73 avis -> data/prepared`.
- `train` lance `python src/train.py`. Il entraîne un pipeline TF-IDF suivi d'une régression logistique, puis journalise le run. Sortie attendue : `F1 macro (test) : 0.8368 | run MLflow : <identifiant>`.

**Explication du journal MLflow.** Le code de `src/train.py` (l. 81 à 86) enregistre les réglages (`log_param`), la métrique (`log_metric`), le fichier `metrics.json` (`log_artifact`) et le modèle (`log_model`). Chaque exécution crée un run dans l'expérience `sentiment-app`.

Vérifiez le graphe et la métrique :

```bash
dvc dag
dvc metrics show
```

Relancez sans rien changer :

```bash
dvc repro
```

Sortie attendue : `Data and pipelines are up to date.`

### Étape 5 : comparer trois essais

Faites varier `ngram_max`. Chaque valeur donne un run distinct :

```bash
dvc exp run --set-param train.ngram_max=1
dvc exp run --set-param train.ngram_max=3
```

Relancez aussi la configuration de référence, `ngram_max` égal à 2, avec `dvc repro` ou `python src/train.py`.

Résultats mesurés sur le jeu de démonstration :

| n-gramme | F1 macro |
|---|---|
| 1 | 0,8445 |
| 2 | 0,8368 |
| 3 | 0,7693 |

**Attention.** `dvc exp run` applique le résultat à l'espace de travail. Pour revenir à la configuration de référence, restaurez `params.yaml` depuis Git :

```bash
git checkout params.yaml
```

Affichez les expériences et les paramètres :

```bash
dvc exp show --only-changed
dvc params diff
```

Pour comparer les runs dans MLflow :

```bash
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

Ouvrez l'adresse affichée (par défaut `http://127.0.0.1:5000`), cochez les trois runs et ouvrez la comparaison.

**Question de réflexion.** Le meilleur score est obtenu avec `n = 1` sur ce petit jeu de 290 avis. Expliquez pourquoi un vocabulaire plus long ne gagne pas toujours.

### Étape 6 : promouvoir le meilleur run

Le script `src/promote.py` (fonction `promouvoir`, l. 21) fait trois choses. Il cherche le run terminé au meilleur `f1_macro` (l. 32, `search_runs`). Il compare ce score au seuil `promote.min_f1` de `params.yaml`. Si le seuil est atteint, il enregistre le modèle sous le nom `sentiment` (l. 47, `register_model`) et pose l'alias `champion` (l. 50).

```bash
python src/promote.py
echo "code de retour : $?"
```

Sortie attendue : `Promu : version 1 (F1 0.8368, run <identifiant>)`, avec le code de retour `0`.

Pendant l'enregistrement, MLflow 3 affiche un avertissement du type « has no artifacts at artifact path 'model' ». Il est attendu : le modèle est un modèle journalisé, et l'enregistrement réussit.

**Tester le refus.** Dans `params.yaml`, passez `min_f1` à `0.85`, puis relancez :

```bash
python src/promote.py
echo "code de retour : $?"
```

Sortie attendue : `Refusé : F1 0.8368 < seuil 0.85 (run <identifiant>)`, avec le code de retour `1`. Restaurez ensuite `0.80`.

**Explication de l'alias.** Le code client charge `models:/sentiment@champion` (`src/predict.py`, constante `URI_REGISTRE`). Déplacer l'alias vers une autre version change le modèle servi, sans modifier le code client.

### Étape 7 : prédire

```bash
python src/predict.py "Le film est excellent, je recommande." "Le service est décevant."
```

Sortie attendue :

```
positif	Le film est excellent, je recommande.
neutre	Le service est décevant.
```

La seconde prédiction est une limite à discuter. Le jeu d'entraînement compte 116 avis neutres sur 217, soit la classe la plus fréquente. Une phrase négative courte tombe donc souvent dans cette classe, même si la probabilité du négatif reste supérieure à celle du positif.

Pour prédire avec le modèle servi par le registre :

```bash
python src/predict.py --registre "Le film est excellent, je recommande."
```

**Explication.** La fonction `charger` lit le fichier local `models/sentiment.joblib`. La fonction `charger_registre` charge `models:/sentiment@champion` via `mlflow.pyfunc.load_model`. Les deux chemins donnent le même résultat tant que le modèle n'a pas changé.

### Étape 8 : décrire et rejouer le pipeline

Le fichier `dvc.yaml` décrit chaque étape. Une étape déclare sa commande (`cmd`), ses dépendances (`deps`), ses paramètres (`params`), ses sorties (`outs`) et ses métriques (`metrics`).

- `prepare` dépend de `src/prepare.py` et de `data/demo_reviews.csv`, utilise `prepare.seed` et `prepare.test_size`, et produit `data/prepared/`.
- `train` dépend de `src/train.py` et de `data/prepared/`, utilise `train.seed`, `train.ngram_max` et `train.min_df`, et produit `models/sentiment.joblib` et `models/metrics.json`.

Modifiez `min_df` de `2` à `3` dans `params.yaml`, puis lancez :

```bash
dvc repro
```

**Observation attendue.** Seule l'étape `train` est relancée, car `min_df` n'apparaît que dans `train`. Le F1 obtenu est `0.8312`. Relancez ensuite `dvc repro` : rien n'est recalculé. Restaurez `min_df: 2` avec `git checkout params.yaml` si vous voulez revenir à la référence.

Committez le pipeline, son état et les paramètres :

```bash
git add dvc.yaml dvc.lock params.yaml
git commit -m "Pipeline DVC"
```

**Explication.** `dvc.lock` fige les hashs des dépendances et des sorties. Un collègue qui clone le dépôt doit committer ou récupérer `dvc.lock` pour reproduire exactement les mêmes résultats.

### Étape 9 : tester

```bash
pytest -q
```

Sortie attendue : `30 passed`.

| Fichier | Ce qu'il vérifie |
|---|---|
| `tests/test_make_demo.py` | 290 avis, labels autorisés, avis uniques, reproductibilité |
| `tests/test_prepare.py` | nettoyage, retrait des doublons et des vides, découpage stratifié sans recouvrement |
| `tests/test_train.py` | F1 dans [0, 1], effet de `ngram_max` sur le vocabulaire, fichiers de sortie |
| `tests/test_mlflow_tracking.py` | run terminé, paramètres, métrique, modèle rechargeable, option `ngram_max` |
| `tests/test_promote.py` | promotion acceptée avec alias posé, refus sous le seuil sans alias |
| `tests/test_predict.py` | une étiquette par avis, orientation des probabilités |
| `tests/test_pipeline_files.py` | cohérence entre `dvc.yaml`, `params.yaml`, `MLproject` et `python_env.yaml` |

**Exercice.** Ajoutez un test qui vérifie qu'une valeur `ngram_max` inférieure à 1 est refusée. Constatez d'abord que `construire_pipeline(0, 2, 42)` ne lève aucune erreur, puis écrivez la validation.

### Étape 10 : lancer le projet avec MLflow Projects (bonus)

Le fichier `MLproject` déclare le point d'entrée `train`, avec le paramètre `ngram_max`. `python_env.yaml` décrit l'environnement.

```bash
dvc repro
mlflow run . -e train -P ngram_max=3 --experiment-name sentiment-app --env-manager local
```

Sortie attendue : `F1 macro (test) : 0.7693 | run MLflow : <identifiant>`.

**Points d'attention.**

- `--experiment-name sentiment-app` est nécessaire. Sans lui, le run part dans l'expérience `Default`.
- `--env-manager local` utilise l'environnement courant. Sans lui, MLflow crée un environnement isolé à partir de `python_env.yaml`.
- Le nom du run est généré par MLflow. Le nom `tfidf-logreg-ngram{n}` n'est donné que par `python src/train.py`.

**Question bonus.** Quelle est la différence de rôle entre `dvc.yaml` et `MLproject` ? Pourquoi `MLproject` ne peut-il pas remplacer `dvc.yaml` pour reproduire le pipeline complet ?

---

## 3. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `dvc add` refuse le fichier | le CSV est déjà suivi par Git | `git rm --cached data/demo_reviews.csv`, puis `dvc add` |
| `dvc repro` ne fait rien | aucune dépendance n'a changé | normal ; modifier `params.yaml` pour forcer un recalcul |
| Les runs n'apparaissent pas dans `mlflow ui` | `MLFLOW_TRACKING_URI` non défini | `export MLFLOW_TRACKING_URI=sqlite:///mlflow.db` |
| `Refusé : F1 ... < seuil` | le modèle est sous `promote.min_f1` | améliorer le modèle, ou justifier le seuil |
| `dvc exp run` a changé `params.yaml` | comportement attendu de DVC | `git checkout params.yaml` |
| `mlflow run` crée un run dans `Default` | option `--experiment-name` absente | ajouter `--experiment-name sentiment-app` |

---

## 4. Livrable

Remettez un dépôt Git contenant :

1. le code modifié, avec un commit par étape (initialisation, données, pipeline, tests) ;
2. `dvc.yaml`, `dvc.lock`, `params.yaml` et `data/demo_reviews.csv.dvc` ;
3. un fichier `RAPPORT.md` (une à deux pages) qui contient :
   - le tableau des trois essais (n-gramme, F1 macro, identifiant du run) ;
   - les réponses aux questions de l'énoncé et du bonus ;
   - la sortie de `pytest -q` ;
   - une justification du seuil `min_f1`.

4. rédiger un compte-rendu complet enrichi par des commentaires, explications, des schémas et des configurations additionnelles en cas d'incompatibilités.
