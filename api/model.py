import sys
from pathlib import Path
from typing import Literal, TypedDict

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
import predict  # noqa: E402  (fonctions du TP2)

MODELE = predict.MODELE
charger_modele = predict.charger_modele

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
    """Enveloppe du modèle TP2 avec l'interface attendue par l'API."""

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
        r = predict.predict(text)  # lève ValueError si le texte est vide

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
    """Construit le classifier (le cache de chargement est dans src/predict.py)."""
    return SentimentClassifier()

