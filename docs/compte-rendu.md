# Compte-Rendu : Application Intelligente d'Analyse de Sentiment
## Fil Rouge — *Développement et Déploiement d'Applications Intelligentes*

---

### Informations Générales

- **Projet :** `sentiment-app`
- **Binôme :** Riyane Abdelkader Lotfi & Marzouk Imad
- **Dépôt GitHub :** `Rayanelo17/Riyane-Abdelkader-Lotfi-----Marzouk-Imad`
- **Date :** Octobre 2026
- **Objectif du jalon :** Cadrage du besoin, mise en place de l'environnement reproductible, entraînement du modèle de classification NLP, évaluation détaillée (matrices de confusion) et développement de l'interface utilisateur interactive.

---

## Sommaire

1. [Contexte et Objectifs Métier](#1-contexte-et-objectifs-métier)
2. [Cadrage des Cas d'Usage (Partie A)](#2-cadrage-des-cas-dusage-partie-a)
3. [Environnement Technique et Architecture du Dépôt (Partie B)](#3-environnement-technique-et-architecture-du-dépôt-partie-b)
4. [Modélisation et Pipeline de Machine Learning](#4-modélisation-et-pipeline-de-machine-learning)
5. [Évaluation et Analyse des Matrices de Confusion](#5-évaluation-et-analyse-des-matrices-de-confusion)
6. [Interface Utilisateur Interactive (Streamlit)](#6-interface-utilisateur-interactive-streamlit)
7. [Validation, Tests et Assurance Qualité](#7-validation-tests-et-assurance-qualité)
8. [Conclusion et Perspectives](#8-conclusion-et-perspectives)

---

## 1. Contexte et Objectifs Métier

Dans le cadre du cours *Développement et Déploiement d'Applications Intelligentes*, l'application fil rouge `sentiment-app` vise à répondre à une problématique e-commerce et relation client : **détecter et prioriser automatiquement les retours et avis clients négatifs**.

### Enjeux principaux :
- **Réactivité du support client :** Réduire le délai moyen de prise en charge (MTTR) des clients insatisfaits.
- **Minimisation des faux négatifs :** L'objectif primordial est de ne rater aucun avis négatif (exigence d'un rappel élevé $\ge 90\%$).
- **Sobriété et efficience technique :** Privilégier une solution légère, explicable et à faible coût d'inférence (pas de dépendance obligatoire à un LLM coûteux ou à un GPU).

---

## 2. Cadrage des Cas d'Usage (Partie A)

En appliquant la grille de décision du cours basée sur quatre critères fondamentaux :
- **(a)** Données étiquetées disponibles
- **(b)** Vérifiabilité nécessaire
- **(c)** Coût par requête / latence
- **(d)** Conséquence d'une erreur

Trois cas d'usage ont été analysés et formalisés dans `docs/cadrage/` :

| Fiche | Cas étudié | Approche retenue | Justification principale |
| :--- | :--- | :---: | :--- |
| `cas1-pharmacie.md` | Détection d'ordonnances incomplètes | **Règles métier** | Les champs obligatoires sont strictement normés par la loi ; le coût d'une erreur médicale est critique (d) et les règles garantissent une vérifiabilité totale (b) sans hallucination. |
| `cas2-reglement.md` | Questions sur le règlement intérieur | **RAG** | La connaissance est contenue dans des documents formels existants sans étiquetage préalable ; la réponse doit citer ses sources pour être vérifiable (b). |
| `cas3-avis-negatifs.md` | Priorisation des avis clients | **Machine Learning** | Disponibilité massive de données textuelles étiquetées (a), coût d'inférence quasi-nul à grande échelle (c) et vérifiabilité via score de probabilité / explicabilité (b). |

Tous les tests de cadrage sont validés :
```bash
pytest tests/test_cadrage.py -q
# 12 passed
```

---

## 3. Environnement Technique et Architecture du Dépôt (Partie B)

### 3.1. Structure du projet

Le projet est structuré selon les standards industriels MLOps :

```text
sentiment-app/
├── docs/                     # Documentation et fiches de cadrage
│   ├── cadrage/              # Cas 1, Cas 2, Cas 3, Grille et Exemples
│   └── COMPTE-RENDU.md       # Le présent compte-rendu
├── data/                     # Jeux de données (sample_reviews.csv)
├── models/                   # Modèles sérialisés (.joblib, exclu par .gitignore)
├── app/                      # Future API FastAPI (Semaine 3)
├── ui/                       # Interface utilisateur Streamlit
│   └── app.py                # Application principale Streamlit
├── exemples/                 # Scripts démonstrateurs (train_sentiment, app_streamlit)
├── tests/                    # Tests automatisés pytest (cadrage, environnement, Git)
├── scripts/                  # Scripts d'automatisation (setup, check_setup, verif)
├── Makefile                  # Cibles make d'automatisation
├── requirements.txt          # Dépendances du semestre aux versions figées
└── .gitignore                # Exclusion des artefacts lourds et sensibles
```

### 3.2. Hygiène logicielle et reproductibilité
- **Isolation :** Utilisation d'un environnement virtuel `.venv`.
- **Exclusion stricte :** Le fichier `.gitignore` protège le dépôt contre l'inclusion de `.venv/`, `models/*`, `.env`, `__pycache__/` et `mlruns/`.
- **Contrôle Git :** Historique vérifié avec des commits explicites des deux membres du binôme.

---

## 4. Modélisation et Pipeline de Machine Learning

### 4.1. Jeu de données
- **Source :** Corpus public Allociné (`tblard/allocine`), 20 000 avis en français issus du split de validation (données publiques sous licence MIT, exemptes de données personnelles).
- **Répartition :** 10 204 avis négatifs et 9 796 avis positifs.
- **Découpage :** 80% entraînement (16 000 avis) et 20% test (4 000 avis), stratifié selon la classe avec `random_state=42`. Le découpage est effectué **strictement avant toute vectorisation** pour empêcher toute fuite de données (*data leakage*).

### 4.2. Pipeline Scikit-Learn
Le pipeline encapsule la transformation textuelle et l'algorithme d'inférence dans un composant unifié :

1. **Vectorisation (`TfidfVectorizer`) :**
   - N-grammes : unigrammes et bigrammes (`ngram_range=(1, 2)`) pour capturer les nuances et négations (ex. : *"ne fonctionne pas"*, *"très déçu"*).
   - Fréquence minimale : `min_df=2` pour filtrer les fautes de frappe isolées.
   - Sous-linéarité : `sublinear_tf=True` ($1 + \log(tf)$).
   - Taille finale du vocabulaire : **149 529 termes**.

2. **Classification (`LogisticRegression`) :**
   - Régularisation L2.
   - Itérations maximales : `max_iter=1000`.
   - Équilibrage des classes : `class_weight='balanced'`.

L'objet complet est sauvegardé dans [`models/sentiment.joblib`](file:///c:/Users/c/Documents/sentiment-app/models/sentiment.joblib).

---

## 5. Évaluation et Analyse des Matrices de Confusion

### 5.1. Matrice de confusion sur le jeu de test (4 000 avis)

L'évaluation a été réalisée sur les 4 000 avis du jeu de test (jamais vus à l'entraînement) :

```text
Matrice de confusion (lignes = vérité terrain, colonnes = prédiction) :
[[1860  181]   <- Négatifs réels
 [ 137 1822]]  <- Positifs réels
```

Sous forme tabulaire explicite :

| | **Prédit : NÉGATIF** | **Prédit : POSITIF** | **Total Réel** |
| :--- | :---: | :---: | :---: |
| **Réel : NÉGATIF** | **1 860** *(Vrais Négatifs - VN)* | **181** *(Faux Positifs - FP)* | **2 041** |
| **Réel : POSITIF** | **137** *(Faux Négatifs - FN)* | **1 822** *(Vrais Positifs - VP)* | **1 959** |
| **Total Prédit** | **1 997** | **2 003** | **4 000** |

### 5.2. Métriques quantitatives détaillées

| Métrique | Formule | Résultat | Seuil cible | Statut |
| :--- | :---: | :---: | :---: | :---: |
| **Rappel Négatif (Recall)** | $\frac{VN}{VN + FP}$ | **91.1 %** | $\ge 90.0 \%$ | **ACCEPTÉ ✅** |
| **Précision Négative** | $\frac{VN}{VN + FN}$ | **93.1 %** | - | Élevée ✅ |
| **F1-Score Négatif** | $2 \cdot \frac{Prec \cdot Rappel}{Prec + Rappel}$ | **0.921** | $\ge 0.82$ | **ACCEPTÉ ✅** |
| **Exactitude Globale (Accuracy)** | $\frac{VN + VP}{Total}$ | **92.0 %** | - | Excellente ✅ |
| **Temps d'inférence unitaire** | - | **< 1 ms** | $< 50$ ms | Temps réel ✅ |

### 5.3. Interprétation des quadrants et risque métier

- **Vrais Négatifs ($VN = 1\,860$) :** Correspond aux clients mécontents correctement identifiés. Ils sont routés sans délai vers l'équipe support avec une priorité haute.
- **Faux Positifs ($FP = 181$) :** Avis négatifs classés positifs à tort ($8.9\%$ d'erreur). C'est le point de vigilance opérationnel. Le seuil de confiance peut être ajusté pour réduire encore ce taux si nécessaire.
- **Faux Négatifs ($FN = 137$) :** Avis positifs interprétés comme négatifs ($7.0\%$). L'impact est mineur : un opérateur lira un avis élogieux, sans préjudice pour le client.
- **Vrais Positifs ($VP = 1\,822$) :** Avis favorables confirmés ($93.0\%$).

---

## 6. Interface Utilisateur Interactive (Streamlit)

Une interface web ergonomique et réactive a été développée dans [`ui/app.py`](file:///c:/Users/c/Documents/sentiment-app/ui/app.py) et déployée localement sur le port `8501`.

```bash
python -m streamlit run ui/app.py
```

### Fonctionnalités intégrées :

1. **Onglet 1 : Analyse Unitaire**
   - Zone de texte libre avec saisie en direct ou sélection rapide parmi la banque d'exemples de `data/sample_reviews.csv`.
   - Verdict instantané avec code couleur (Vert pour Positif, Rouge pour Négatif).
   - Jauge dynamique de confiance et probabilités comparées ($P(\text{positif})$ vs $P(\text{négatif})$).
   - **Module d'explicabilité :** Extraction des mots-clés influents (tokens) et affichage de leurs poids TF-IDF positifs ou négatifs.

2. **Onglet 2 : Traitement par Lot (Batch CSV)**
   - Téléversement de fichiers CSV (nécessitant une colonne `review`) ou chargement direct du fichier de test du kit (`data/sample_reviews.csv`).
   - Calcul immédiat des prédictions et scores de confiance.
   - Indicateurs globaux (total, taux de négatifs, taux de positifs, confiance moyenne).
   - Graphique de répartition des sentiments.
   - **Matrice de confusion dynamique :** Calculée automatiquement si le fichier importé contient une colonne de vérité terrain `label`.
   - Tableau filtrable (Tous / Négatifs seulement / Positifs seulement) et bouton d'export des résultats enrichis en CSV.

3. **Onglet 3 : Performances du Modèle**
   - Tableau synthétique de la matrice de confusion sur les 4 000 avis Allociné.
   - Cartes d'indicateurs de rappel et d'exactitude.
   - Fiche récapitulative de l'architecture algorithmique et des critères de validation.

---

## 7. Validation, Tests et Assurance Qualité

Le projet intègre une suite de tests automatisés via `pytest` couvrant l'ensemble du périmètre :

```bash
pytest -q
```
- **Tests d'environnement (`test_environment.py`) :** Vérification de la version de Python, de la présence des répertoires requis (`ui/`, `app/`, `data/`, etc.), de l'intégrité de `.gitignore` et du bon pinning des dépendances dans `requirements.txt`.
- **Tests de cadrage (`test_cadrage.py`) :** Validation des rubriques obligatoires, de la sélection d'une approche unique et de la justification par au moins deux critères de la grille.
- **Tests Git (`test_git_history.py`) :** Validation de l'historique des commits et de la contribution effective des membres.

Le script de diagnostic `python scripts/check_setup.py` permet à tout moment de valider la conformité du poste de travail.

---

## 8. Conclusion et Perspectives

### Bilan
Le socle applicatif est pleinement opérationnel :
- Le cadrage méthodologique est validé.
- L'environnement de développement est reproductible et sous contrôle de version.
- Le modèle de Machine Learning supervisé atteint un **rappel de 91.1%** sur la classe négative, surpassant l'objectif contractuel de 90%.
- L'interface Streamlit permet aussi bien l'expérimentation unitaire que le traitement industriel par lot.

### Perspectives pour les prochaines étapes
1. **Semaine 3 (API REST) :** Exposition du modèle via des endpoints FastAPI (`/predict`, `/batch_predict`, `/health`) dans le dossier `app/`.
2. **Semaine 7 (Gestion des données avec DVC) :** Versionnage des jeux de données volumineux et des artefacts de modèle en stockage distant.
3. **Semaine 8-9 (MLOps & CI/CD) :** Tracking des hyperparamètres avec MLflow, conteneurisation Docker de l'API et déploiement continu automatisé.
