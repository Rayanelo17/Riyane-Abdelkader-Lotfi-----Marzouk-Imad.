# sentiment-app

Classifieur de sentiment d'avis clients en français.

## Description

`sentiment-app` est un paquet Python permettant d'analyser le sentiment d'un avis client rédigé en français. Il attribue une étiquette (`positive` ou `negative`) et un score de confiance associé.

## Installation

Pour installer le paquet à partir de la roue :

```bash
pip install dist/sentiment_app-1.0.0-py3-none-any.whl
```

## Utilisation

### En Python

```python
from sentiment_app import predict

resultat = predict("Livraison rapide, produit conforme, très satisfait")
print(resultat)
# {'label': 'positive', 'score': 0.777}
```

### En ligne de commande

Une commande `sentiment` est mise à disposition lors de l'installation du paquet :

```bash
# Sortie au format texte
sentiment "Livraison rapide, produit conforme, très satisfait"
# positive (confiance 78%)

# Sortie au format JSON
sentiment --json "Produit cassé et service injoignable, à fuir"
# {"label": "negative", "score": 0.7465}

# Affichage de la version
sentiment --version
# sentiment 1.0.0
```

## Limites connues

- Le modèle est entraîné pour la langue française sur des avis clients.
- En cas de texte vide ou composé uniquement d'espaces, `predict` lève une exception `ValueError("texte vide")` et la commande en ligne de commande renvoie le code d'erreur 2.

