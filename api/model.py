from typing import Literal, TypedDict

from sentiment_app import predict
from sentiment_app.predict import predict as _predict

LabelType = Literal["POSITIVE", "NEGATIVE", "INDETERMINE"]

CORRESPONDANCE_LABELS: dict[str, LabelType] = {
    "positive": "POSITIVE",
    "negative": "NEGATIVE",
}

SEUIL_ABSTENTION = 0.60


class PredictionResult(TypedDict):
    label: LabelType
    score: float
    text: str


class SentimentClassifier:
    """Enveloppe du modèle avec l'interface attendue par l'API."""

    model_name = "tfidf-logreg (TP2, models/sentiment.joblib)"

    def __init__(self) -> None:
        pipeline = predict.charger_modele()  # échoue immédiatement si le joblib manque : on ne démarre pas sans modèle
        classes = getattr(pipeline, "classes_", None)
        if classes is not None:
            for c in classes:
                if str(c).lower() not in CORRESPONDANCE_LABELS:
                    raise ValueError(f"Classe inattendue dans le modèle: {c}")

    def predict(self, text: str) -> PredictionResult:
        """Renvoie {'label': 'POSITIVE'|'NEGATIVE'|'INDETERMINE', 'score': float, 'text': aperçu}."""
        r = _predict(text)  # lève ValueError si le texte est vide

        score = float(r["score"])
        if score < SEUIL_ABSTENTION:
            label: LabelType = "INDETERMINE"
        else:
            raw_label = str(r["label"]).lower()
            if raw_label not in CORRESPONDANCE_LABELS:
                raise ValueError(f"Label inattendu du modèle: {raw_label}")
            label = CORRESPONDANCE_LABELS[raw_label]
        return {"label": label, "score": score, "text": text[:100]}


def load_model() -> SentimentClassifier:
    """Construit le classifier (le cache de chargement est dans sentiment_app/predict.py)."""
    return SentimentClassifier()
