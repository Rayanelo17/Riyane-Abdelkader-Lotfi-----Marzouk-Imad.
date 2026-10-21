# Semaine 4 : tests approfondis et qualité, notes du binôme

**Module :** Développement et déploiement d'applications intelligentes · **Auteur :** A. Larhlimi  
**Binôme :** Rayanelo17 & Imad Marzouk  

---

## 1. Couverture

- **Avant (étape A) :** 91 % de couverture sur `api/`, avec 6 lignes jamais exécutées dans `api/main.py`.
- **Pour chacune, le comportement qu'aucun test ne vérifiait :**
  - Les cas d'erreur interne du modèle et exceptions non gérées (retour 500).
  - L'absence ou le défaut d'initialisation du classifieur (retour 503).
  - Le bloc principal d'exécution directe (`uvicorn.run`).
- **Après (étape D) :** **98.88 %** de couverture globale sur `api/` (100 % sur `api/main.py`, 97 % sur `api/model.py`, 100 % sur `api/schemas.py`).
- **Lignes exclues de la mesure et pourquoi :** Le bloc `if __name__ == "__main__":` dans `api/main.py` exclu via `# pragma: no cover`, car le serveur est exécuté par le lanceur Uvicorn ASGI en production et via la fixture `TestClient` lors des tests automatisés.

---

## 2. Défauts trouvés

| Symptôme | Test qui l'a révélé | Correction |
| --- | --- | --- |
| Texte constitué uniquement d'espaces (`{"text": "   "}`) accepté par le validateur de longueur mais rejeté plus tard avec un format incohérent | `test_frontieres_du_contrat[texte-espaces]` & `test_propriete_texte_blancs` (Hypothesis) | Ajout d'un `@field_validator("text")` dans `api/schemas.py` refusant tout texte vide après `str.strip()` avec format standard 422 |
| Panne interne du modèle (`ValueError`) renvoyant un code 422 au lieu de 500, avec fuite du message d'erreur interne | `test_modele_en_panne_500` dans `tests/test_integration.py` | Suppression de la conversion de `ValueError` en 422 dans `api/main.py` ; interception globale renvoyant 500 générique sans message interne |
| Modèle répondant avec certitude sur du texte sans aucun mot du vocabulaire | Étape C (`/docs` & `src/courbe_seuil.py`) | Implémentation du seuil d'abstention `SEUIL_ABSTENTION = 0.60` dans `api/model.py` renvoyant le verdict `INDETERMINE` |
| Possibilité de démarrer l'API avec un modèle corrompu ou contenant des étiquettes inattendues | `test_classes_inattendues_refuse_demarrage` | Correspondance stricte `CORRESPONDANCE_LABELS` et validation des classes `pipeline.classes_` dès le `lifespan` dans `api/model.py` |

---

## 3. Seuil d'abstention

- **Points de la courbe (`python src/courbe_seuil.py`) :**
  - Seuil `0.50` : répond à 58 avis (100 %), exactitude `0.879`, 7 erreurs, 0 vers un humain.
  - Seuil `0.60` : répond à 43 avis (74 %), exactitude `0.930`, 3 erreurs, 15 vers un humain.
  - Seuil `0.70` : répond à 28 avis (48 %), exactitude `0.964`, 1 erreur, 30 vers un humain.
- **Seuil retenu :** `0.60`.
- **Part des avis envoyés à un humain :** ~26 % des avis ambigus redirigés.
- **Justification :** Le seuil 0.60 évite plus de la moitié des erreurs de classification (4 erreurs éliminées) tout en maintenant un taux d'automatisation élevé (74 % des requêtes traitées avec une fiabilité supérieure à 93 %).

---

## 4. Jeu d'or

- **Bilan (`python src/bilan_jeu_or.py`) :**
  - **9 justes** (45 %)
  - **10 abstentions** (50 %)
  - **1 erreur franche déclarée** (5 %, ligne 11 : sac de mots sensible au mot « qualité » dans un contexte négatif, marquée `xfail(strict=True)`).
- **Ce que ces chiffres disent du modèle :**
  - Le modèle TF-IDF avec régression logistique est performant sur les avis aux polarités lexicales explicites.
  - Face aux avis nuancés, à l'ironie ou aux négations complexes, le modèle préfère s'abstenir (`INDETERMINE`) plutôt que de produire une fausse prédiction, ce qui est le comportement attendu en production.

---

## 5. Outils statiques

- **Remarques au départ :**
  - `ruff` : 22 remarques (imports non triés, `zip()` sans `strict=`, variables à lettre unique ambiguë `l`).
  - `mypy` : 0 remarque au départ car les sorties des fonctions étaient typées implicitement en `dict` (donc `Any`), masquant les incohérences de types.
- **Apport du typage fort :**
  - Après introduction de `TypedDict` (`PredictionResult`) et `Literal["POSITIVE", "NEGATIVE", "INDETERMINE"]`, `mypy` garantit l'alignement exact entre le retour de prédiction et le schéma de l'API.
  - `ruff check .` : 0 erreur, code entièrement formaté et conforme aux règles PEP8/B/I.
