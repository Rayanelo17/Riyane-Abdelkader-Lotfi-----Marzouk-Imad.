# Semaine 5 : l'interface, notes du binôme

**Module :** Développement et déploiement d'applications intelligentes · **Auteur :** A. Larhlimi  
**Binôme :** Rayanelo17 & Imad Marzouk  

---

## 1. Pourquoi l'interface n'importe-t-elle jamais `predict()` ni le fichier `.joblib` ?

L'interface Streamlit est un pur client web découplé de l'API (architecture client-serveur / 3-tiers). Elle interagit exclusivement par requêtes HTTP via le module `ui.client`.

Ce découplage présente trois avantages fondamentaux :
- **Indépendance d'évolution :** Le modèle de machine learning peut être réentraîné, changé d'architecture ou optimisé sans modifier une seule ligne du code de l'interface.
- **Déploiement distribué :** L'interface et l'API peuvent être hébergées sur des machines ou conteneurs distincts (par exemple l'interface sur un serveur web léger et l'API sur une machine avec GPU ou ressources dédiées).
- **Légèreté du frontend :** L'interface n'a pas besoin de charger en mémoire les dépendances d'inférence (comme scikit-learn ou le binaire de 20 Mo du modèle), ce qui réduit son empreinte mémoire et accélère son cycle de vie.

---

## 2. Que montre l'interface quand le modèle hésite, et pourquoi est-ce mieux qu'un verdict forcé ?

Quand la confiance de la prédiction est inférieure au seuil d'abstention (`SEUIL_ABSTENTION = 0.60`), le client renvoie l'étiquette `INDETERMINE`. L'interface Streamlit affiche alors :
- Le verdict : **« Je ne sais pas : à faire vérifier par un humain »** avec le pourcentage de confiance calculé (ex: 52 %).
- Un bandeau informatif explicite : `st.info("La confiance est sous le seuil d'abstention : un humain doit trancher.")`.

C'est bien supérieur à un verdict forcé, car un texte ambigu, sarcastique ou fait de mots inconnus (tel que *"xqzvb wlkptr grombu"*) recevrait arbitrairement la classe négative avec une confiance marginale de ~51,6 %. Forcer une prédiction fausse dégrade la confiance des utilisateurs et peut causer des erreurs métier (par exemple classer un retour mitigé en insatisfaction grave). Transmettre ces cas au support humain fiabilise le traitement global.

---

## 3. Que voit l'utilisateur quand l'API tombe ? Et le fichier `data/avis_exemple.csv` : combien d'avis partent vers un humain ?

- **Quand l'API est arrêtée ou injoignable :** Grâce à la méthode `sante()` du client qui intercepte l'erreur réseau (`httpx.ConnectError`), l'interface ne plante pas. Elle affiche un bandeau rouge en haut de page : **« API injoignable : lancez d'abord uvicorn api.main:app »**. Si l'utilisateur tente une prédiction, un message d'erreur clair l'avertit de l'impossibilité de joindre le serveur.
- **Sur le fichier `data/avis_exemple.csv` :** L'analyse du lot de 6 avis donne la répartition `{'INDETERMINE': 1, 'NEGATIVE': 2, 'POSITIVE': 3}`. Exactement **1 avis sur 6** (soit environ 17 %) part vers un humain pour vérification.
