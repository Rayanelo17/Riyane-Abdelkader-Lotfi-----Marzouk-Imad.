# Compte rendu global des TP1 à TP3

**Binôme :** Abdelkader Lotfi Riyane (@Rayanelo17) et Imad Marzouk (@Imad-Marzouk)  
**Dépôt :** https://github.com/Rayanelo17/sentiment-app  
**Date :** 12/10/2026  

---

## 1. Le besoin et son cadrage

Le projet vise à concevoir et déployer une application d'analyse de sentiment appliquée aux retours et avis clients. Pour une équipe de relation client, le volume important d'avis rend le traitement manuel exhaustif trop lent, entraînant des retards critiques dans la prise en charge des insatisfactions. L'automatisation permet de catégoriser instantanément les messages pour isoler et prioriser les clients mécontents.

Pour le cas 3 (priorisation des avis négatifs), nous avons retenu une approche de **Machine Learning supervisé**. Cette décision s'appuie directement sur les critères de cadrage suivants :
- **(a) Données étiquetées :** Des jeux de données volumineux annotés en polarités (positif/négatif) sont immédiatement disponibles (notamment le corpus public Allociné de 20 000 avis), permettant d'entraîner un modèle spécialisé.
- **(b) Vérifiabilité :** Un modèle statistique fournit des probabilités calibrées et des scores de confiance explicites, permettant d'isoler les zones d'incertitude.
- **(c) Coût par requête :** L'inférence locale d'un pipeline scikit-learn (TF-IDF + régression logistique) a un coût marginal nul et une latence quasi instantanée (< 10 ms), contrairement à des appels répétitifs à des API de LLM propriétaires.
- **(d) Conséquence d'une erreur :** Une erreur ponctuelle n'a pas d'impact vital, d'autant qu'un routage vers un agent humain est prévu pour tout score sous le seuil d'incertitude.

---

## 2. L'architecture du dépôt

Le projet respecte une séparation stricte des responsabilités entre données, modélisation, interface et validation :
- `data/` : Jeux d'exemples et jeux de démonstration générés (`demo_reviews.csv`).
- `models/` : Artefacts de modélisation (`sentiment.joblib`) et mesures de qualité (`metrics.json`).
- `src/` : Scripts autonomes de génération de données (`make_demo.py`), d'entraînement (`train.py`), d'évaluation (`evaluate.py`) et d'inférence pure (`predict.py`).
- `ui/` : Interface utilisateur Streamlit (`app.py`) et fonctions métier pures testables (`helpers.py`).
- `tests/` : Suite de tests automatisés pytest couvrant l'environnement, le cadrage, l'historique Git, la qualité du modèle et l'interface (45 tests au total).
- `docs/` : Fiches de cadrage initiales, analyses d'erreurs (`semaine2.md`), arbitrage du seuil (`semaine3.md`) et le présent compte rendu.

### Pourquoi `models/metrics.json` est-il versionné alors que `models/sentiment.joblib` ne l'est pas ?
Le modèle binaire `sentiment.joblib` est un artefact volumineux qui peut être intégralement reconstruit de manière déterministe en exécutant une seule commande (`python src/train.py`). Le stocker dans Git alourdirait inutilement le dépôt.  
En revanche, `models/metrics.json` est un fichier texte léger qui documente la mesure objective des performances du modèle. Sa présence dans Git permet à l'intégration continue (GitHub Actions) de vérifier les seuils de qualité (par exemple F1 macro >= 0.85) sans nécessiter le transfert ni le réentraînement d'un modèle lourd en CI.

---

## 3. Le modèle

Le pipeline applique un découpage `train_test_split` avec une proportion 80/20 et un `random_state=42` strict pour garantir une reproductibilité parfaite. Le découpage est effectué avant toute vectorisation afin d'éviter toute fuite d'information (*data leakage*) du jeu de test vers l'entraînement. Vectoriseur et modèle sont encapsulés au sein d'un `Pipeline` scikit-learn combinant `TfidfVectorizer` et `LogisticRegression`.

### Résultats chiffrés

| Jeu | Avis | Vocabulaire | F1 macro | Rappel négatif | Exactitude (Accuracy) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Démonstration** | 290 | 316 termes | 0.879 | 0.931 | 0.879 |
| **Allociné** | 20 000 | 149 529 termes | **0.9205** | **0.9113** | **0.9205** |

### Matrice de confusion sur le jeu de démonstration (test : 58 avis)

```text
            negative  positive
  negative        27         2
  positive         5        24
```
- **Ligne 1 (Vérité = negative) :** 27 avis négatifs correctement classés, et 2 faux positifs (avis négatifs classés positifs). Ces 2 cas sont critiques sur le plan métier car ils correspondent à des clients mécontents non pris en charge à temps.
- **Ligne 2 (Vérité = positive) :** 5 faux négatifs (avis positifs étiquetés négatifs) et 24 vrais positifs.

---

## 4. L'analyse des erreurs

Sur les 58 avis du jeu de test de démonstration, nous comptabilisons exactement **7 erreurs** (2 faux positifs et 5 faux négatifs).

Toutes ces erreurs portent systématiquement sur des **avis mitigés avec retournement discursif**.  
*Exemple tiré de nos données réelles :*
> `[Vérité : positive | Prédiction : negative]` : *"Commande jamais reçue et remboursement en attente au début, sauf que ensuite je recommande."*

**Mécanisme de l'erreur :** Notre modèle s'appuie sur une représentation en sac de mots (TF-IDF). Les termes négatifs très forts du début (*"jamais reçue"*, *"remboursement en attente"*) cumulent un poids supérieur aux termes positifs de conclusion (*"je recommande"*). Le sac de mots ignore complètement l'ordre syntaxique et la conjonction pivot (*"sauf que ensuite"*), alors que le sentiment réel est dicté par la conclusion.

**Correction requise :** Pour dépasser cette limite structurelle du sac de mots, il est indispensable de capturer l'ordre et le contexte discursif au moyen d'embeddings de phrases (modèles de type Transformers / BERT), ce qui constitue le programme de la semaine 9.

---

## 5. L'interface et le seuil d'incertitude

Dans l'interface Streamlit (`ui/helpers.py`), nous retenons un seuil d'incertitude **`SEUIL_INCERTITUDE = 0.60` (60 %)**.

### Analyse de l'arbitrage
- À **0.50**, aucun avis n'est marqué comme incertain (0 avis à vérifier) : le modèle tranche aveuglément et commet des erreurs sans prévenir.
- À **0.75**, 234 avis sur 290 (80,7 %) sont renvoyés à l'équipe humaine, ce qui engorgerait le service client et annulerait le bénéfice de l'automatisation.
- À **0.60**, exactement **117 avis sur 290 (40,3 %)** sont signalés « à vérifier ». Ce compromis intercepte la majorité des 130 cas complexes et mitigés du jeu de données, protégeant ainsi l'expérience client tout en automatisant en toute sécurité 60 % des flux évidents.

---

## 6. Ce qui a échoué

Durant les séances de TP, plusieurs difficultés réelles ont été rencontrées et résolues :
1. **Conflit de dépendances pip sur Windows :** Lors de l'installation initiale avec Python 3.12, un conflit strict est survenu entre `pyarrow==21.0.*` et `mlflow==2.17.*` (ce dernier imposant `pyarrow<18`). Nous avons résolu l'incident en alignant `pyarrow==17.0.*` dans `requirements.txt`, garantissant ainsi la compatibilité totale de la chaîne ML.
2. **Gestion de l'historique multi-auteurs :** Le test `test_git_history.py` imposait au minimum 2 auteurs de commits distincts pour valider le travail en binôme. Nous avons harmonisé les configurations Git pour certifier les contributions d'Abdelkader Lotfi Riyane et d'Imad Marzouk.
3. **Absence initiale du modèle en CI :** Lors de la première mise en place, la distinction entre artefacts locaux (`sentiment.joblib`) et métriques versionnées (`metrics.json`) a été clarifiée grâce à l'implémentation de tests de CI conditionnels via `pytest.skip`.

---

## 7. Ce que vous feriez autrement

Avec le recul et dans le cadre d'un déploiement industriel :
- Nous introduirions un modèle d'embeddings pré-entraîné multilingue (ex. `CamemBERT` ou `MiniLM`) pour éliminer les erreurs sur les négations et retours de phrase.
- Nous mettrions en place un pipeline DVC dès la semaine 2 pour versionner les données volumineuses (Allociné 20 000 avis) avec stockage distant S3/MinIO.
- Nous ajouterions une surveillance continue du *data drift* et un mécanisme de *feedback loop* permettant aux conseillers de requalifier en un clic les cas étiquetés « à vérifier » pour réentraîner le modèle.

---

## 8. Répartition du travail

| Membre | Ce qu'il a pris en charge |
| --- | --- |
| **Abdelkader Lotfi Riyane** | Cadrage initial, scripts de pipeline d'entraînement et d'évaluation, gestion du dépôt GitHub. |
| **Imad Marzouk** | Conception et tests de validation de l'interface Streamlit, analyse des seuils d'incertitude et rédaction du compte rendu. |

---

## Outils d'aide à la rédaction

Ce travail a bénéficié de l'assistance de l'agent IA Antigravity pour la formalisation des scripts de test et la mise en forme de la documentation technique, chaque calcul métrique, résultat de test et analyse d'erreur ayant été systématiquement vérifié et validé sur machine locale par le binôme.
