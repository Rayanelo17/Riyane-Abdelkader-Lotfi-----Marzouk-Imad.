from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Requête de prédiction."""
    text: str = Field(..., min_length=1, max_length=2000, description="Texte à classifier",
                      examples=["Livraison rapide, produit conforme, très satisfait"])

    @field_validator("text")
    @classmethod
    def valider_texte_non_vide(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("texte vide")
        return v


class PredictResponse(BaseModel):
    """Réponse de prédiction."""
    label: Literal["POSITIVE", "NEGATIVE", "INDETERMINE"] = Field(..., description="POSITIVE, NEGATIVE ou INDETERMINE")
    score: float = Field(..., ge=0, le=1, description="Confiance de la classe prédite")
    text: str = Field(..., description="Aperçu du texte analysé (100 caractères)")


class HealthResponse(BaseModel):
    """Réponse de l'endpoint de santé."""
    status: str = Field(..., description="OK si le modèle est chargé")
    model: str = Field(..., description="Nom du modèle servi")


class ErrorResponse(BaseModel):
    """Format standard des erreurs FastAPI."""
    detail: str
