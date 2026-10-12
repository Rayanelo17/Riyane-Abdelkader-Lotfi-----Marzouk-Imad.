"""Fonctions pures de l'interface : chargement du modèle, prédiction, résumé, métriques.

Séparées de app.py pour être testables sans lancer Streamlit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RACINE = Path(__file__).resolve().parents[1]
MODELE = RACINE / "models" / "sentiment.joblib"
METRIQUES = RACINE / "models" / "metrics.json"
SEUIL_INCERTITUDE = 0.60   # en dessous, on affiche « à vérifier »


def charger_predicteur():
    """Renvoie la fonction predict() du TP2, ou lève FileNotFoundError."""
    if not MODELE.exists():
        raise FileNotFoundError(
            "models/sentiment.joblib absent : lancez d'abord python src/train.py"
        )
    sys.path.insert(0, str(RACINE / "src"))
    from predict import predict  # noqa: E402
    return predict


def charger_pipeline():
    """Charge l'objet Pipeline scikit-learn complet."""
    if not MODELE.exists():
        raise FileNotFoundError("models/sentiment.joblib absent")
    import joblib
    return joblib.load(MODELE)


def charger_metriques() -> dict:
    """Charge le fichier models/metrics.json s'il existe."""
    if not METRIQUES.exists():
        return {}
    return json.loads(METRIQUES.read_text(encoding="utf-8"))


def classer(texte: str, predicteur, seuil: float = SEUIL_INCERTITUDE) -> dict:
    """Prédiction enrichie d'un verdict lisible par un non-spécialiste."""
    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("texte vide")
    resultat = predicteur(texte)
    incertain = resultat["score"] < seuil
    resultat["verdict"] = "à vérifier" if incertain else resultat["label"]
    resultat["incertain"] = incertain
    return resultat


def predire_lot(df: pd.DataFrame, predicteur, colonne: str = "review", seuil: float = SEUIL_INCERTITUDE) -> pd.DataFrame:
    """Ajoute les colonnes prediction, confiance et verdict à un tableau d'avis."""
    if colonne not in df.columns:
        raise KeyError(f"colonne '{colonne}' absente du fichier")
    lignes = [classer(str(t), predicteur, seuil=seuil) for t in df[colonne].fillna("") if str(t).strip()]
    if not lignes:
        raise ValueError("aucun avis exploitable dans le fichier")
    sortie = df.loc[df[colonne].fillna("").astype(str).str.strip() != ""].copy()
    sortie["prediction"] = [l["label"] for l in lignes]
    sortie["confiance"] = [round(l["score"], 3) for l in lignes]
    sortie["verdict"] = [l["verdict"] for l in lignes]
    return sortie


def resumer(df: pd.DataFrame) -> dict:
    """Les chiffres clés que l'équipe relation client regarde en premier."""
    negatifs = int((df["prediction"] == "negative").sum())
    a_verifier = int((df["verdict"] == "à vérifier").sum())
    return {
        "total": len(df),
        "negatifs": negatifs,
        "part_negatifs": round(100 * negatifs / len(df), 1) if len(df) else 0.0,
        "a_verifier": a_verifier,
        "confiance_moyenne": round(float(df["confiance"].mean()), 3) if len(df) else 0.0,
    }


def expliquer_texte(texte: str, pipeline) -> list[dict]:
    """Extrait les termes présents dans le texte et leur contribution au score."""
    if not isinstance(texte, str) or not texte.strip():
        return []
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    
    vec = tfidf.transform([texte]).tocoo()
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    
    contributions = []
    for col, val in zip(vec.col, vec.data):
        mot = feature_names[col]
        poids = float(val * coefs[col])
        contributions.append({
            "mot": mot,
            "contribution": round(poids, 3),
            "polarite": "positive" if poids > 0 else "negative",
        })
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


def obtenir_top_mots(pipeline, n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie les mots les plus positifs et négatifs du modèle."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    
    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]
    
    df_pos = pd.DataFrame({
        "Mot / N-gram": [feature_names[i] for i in top_pos_idx],
        "Coefficient": [round(float(coefs[i]), 3) for i in top_pos_idx],
    })
    df_neg = pd.DataFrame({
        "Mot / N-gram": [feature_names[i] for i in top_neg_idx],
        "Coefficient": [round(float(coefs[i]), 3) for i in top_neg_idx],
    })
    return df_pos, df_neg


def calculer_matrice_confusion(df: pd.DataFrame, col_verite: str = "label", col_pred: str = "prediction") -> dict:
    """Calcule la matrice de confusion et les métriques associées."""
    if col_verite not in df.columns or col_pred not in df.columns:
        return {}
    
    labels = ["negative", "positive"]
    cm = confusion_matrix(df[col_verite], df[col_pred], labels=labels)
    vn, fp, fn, vp = cm.ravel()
    
    total = len(df)
    accuracy = (vp + vn) / total if total else 0.0
    f1 = f1_score(df[col_verite], df[col_pred], labels=labels, average="macro")
    rappel_neg = vn / (vn + fp) if (vn + fp) else 0.0
    precision_neg = vn / (vn + fn) if (vn + fn) else 0.0
    
    matrice_df = pd.DataFrame(
        cm,
        index=["Vérité : Négatif", "Vérité : Positif"],
        columns=["Prédit : Négatif", "Prédit : Positif"],
    )
    
    erreurs_df = df[df[col_verite] != df[col_pred]].copy()
    
    return {
        "matrice_df": matrice_df,
        "vn": int(vn),
        "fp": int(fp),
        "fn": int(fn),
        "vp": int(vp),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1), 4),
        "rappel_negatif": round(float(rappel_neg), 4),
        "precision_negatif": round(float(precision_neg), 4),
        "nb_erreurs": len(erreurs_df),
        "erreurs_df": erreurs_df,
    }
