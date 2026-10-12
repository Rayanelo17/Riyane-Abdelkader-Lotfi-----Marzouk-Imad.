# Analyse des erreurs du modèle (Semaine 2)

Les erreurs observées lors de l'évaluation (notamment 2 faux positifs et 5 faux négatifs sur 58 avis de test) concernent toutes des avis mitigés comportant une première impression suivie d'un retournement d'opinion.
Ces erreurs surviennent parce que notre modèle s'appuie sur une approche en sac de mots (Bag-of-Words / TF-IDF) qui ne prend pas en compte l'ordre des termes ni la structure temporelle et discursive du texte, se contentant de sommer des polarités lexicales sans identifier la conclusion finale.
Pour corriger ces limitations, il faudrait utiliser des représentations contextualisées et des embeddings capables de préserver l'ordre des mots et la sémantique globale (tels que des modèles de type Transformers ou embeddings pré-entraînés prévus en semaine 9).
