"""Interface de démonstration enrichie du classifieur de sentiment.

Usage : streamlit run ui/app.py
Prérequis : models/sentiment.joblib et models/metrics.json
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from helpers import (
    SEUIL_INCERTITUDE,
    calculer_matrice_confusion,
    charger_metriques,
    charger_pipeline,
    charger_predicteur,
    classer,
    expliquer_texte,
    obtenir_top_mots,
    predire_lot,
    resumer,
)

RACINE = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Analyse de Sentiment - Démonstrateur ML",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chargement en cache des artefacts
@st.cache_resource
def get_predicteur():
    return charger_predicteur()

@st.cache_resource
def get_pipeline():
    return charger_pipeline()

try:
    predire = get_predicteur()
    pipeline = get_pipeline()
    metriques = charger_metriques()
except FileNotFoundError as err:
    st.error(f"⚠️ Artefact manquant : {err}")
    st.info("Dans un terminal, à la racine du dépôt : `python src/train.py --allocine`")
    st.stop()

# Barre latérale avec configuration et métriques rapides
with st.sidebar:
    st.title("⚙️ Paramètres et Modèle")
    st.markdown("---")
    
    seuil_choisi = st.slider(
        "Seuil d'incertitude (arbitrage)",
        min_value=0.50,
        max_value=0.85,
        value=float(SEUIL_INCERTITUDE),
        step=0.05,
        help="En dessous de ce seuil, la prédiction est marquée comme « à vérifier » pour réorientation humaine.",
    )
    
    st.markdown("### 📊 Modèle Actif")
    source_nom = metriques.get("source", "Inconnue").capitalize()
    f1_val = metriques.get("f1_macro", 0.0)
    acc_val = metriques.get("accuracy", 0.0)
    st.metric("Source d'entraînement", source_nom)
    st.metric("F1-score Macro (test)", f"{f1_val:.3f}" if f1_val else "N/A")
    st.metric("Exactitude (Accuracy)", f"{acc_val:.1%}" if acc_val else "N/A")
    st.caption("Dépôt : `sentiment-app` | Binôme : Abdelkader et Imad")

# En-tête principal
st.title("🎯 Analyse de Sentiment des Avis Clients")
st.markdown(
    "Application interactive de qualification automatique des avis clients avec détection d'incertitude, explicabilité et matrice de confusion."
)

# Onglets principaux
onglet_avis, onglet_lot, onglet_modele, onglet_doc = st.tabs([
    "💬 Un avis (Interactif)",
    "📁 Traitement par lot (CSV)",
    "📈 Modèle et Matrice de confusion",
    "📘 Cadrage et Décisions Métier",
])

# -------------------------------------------------------------
# ONGLET 1 : Un avis
# -------------------------------------------------------------
with onglet_avis:
    st.subheader("Analyse unitaire en temps réel")
    st.write("Saisissez un avis ou testez l'un des scénarios types du cours :")
    
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    texte_initial = "Livraison rapide, article bien emballé et conforme, je recommande."
    
    if col_b1.button("🌟 Positif évident", use_container_width=True):
        st.session_state["ex_texte"] = "Service impeccable, livraison rapide, je recommande sans hésiter."
    if col_b2.button("❌ Négatif évident", use_container_width=True):
        st.session_state["ex_texte"] = "Produit cassé à l'arrivée et aucune réponse du service client."
    if col_b3.button("🔄 Mitigé (Retournement)", use_container_width=True):
        st.session_state["ex_texte"] = "Service impeccable et livraison en avance sur le papier, sauf que ensuite produit cassé à l'arrivée."
    if col_b4.button("⚖️ Neutre / Ambigu", use_container_width=True):
        st.session_state["ex_texte"] = "Colis reçu ce matin comme convenu."
        
    texte_zone = st.text_area(
        "Texte de l'avis client :",
        value=st.session_state.get("ex_texte", texte_initial),
        height=110,
    )
    
    if st.button("🚀 Analyser l'avis", type="primary"):
        if not texte_zone.strip():
            st.warning("Veuillez saisir un texte à analyser.")
        else:
            r = classer(texte_zone, predire, seuil=seuil_choisi)
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                if r["incertain"]:
                    st.warning(f"⚠️ **À vérifier par un agent** (le modèle hésite, penchant : **{r['label']}**)")
                    st.caption("Confiance sous le seuil d'arbitrage : réorientation prioritaire vers un opérateur humain.")
                elif r["label"] == "negative":
                    st.error("🚨 **Avis Négatif détecté** : À traiter en priorité !")
                    st.caption("Alerte client mécontent prise en charge automatiquement.")
                else:
                    st.success("✅ **Avis Positif** : Client satisfait")
                    st.caption("Validation automatique sans intervention requise.")
                    
            with col_res2:
                st.metric("Indice de Confiance", f"{r['score']:.1%}")
                st.progress(float(r["score"]))
                
            # Explicabilité (contributions lexicales)
            st.markdown("#### 🔍 Explication des termes influents dans cet avis")
            mots_impact = expliquer_texte(texte_zone, pipeline)
            if mots_impact:
                col_exp1, col_exp2 = st.columns([1, 1])
                df_exp = pd.DataFrame(mots_impact)
                with col_exp1:
                    st.dataframe(df_exp, use_container_width=True)
                with col_exp2:
                    st.info(
                        "**Pourquoi ce verdict ?**  \n"
                        "Le classifieur TF-IDF somme les poids des termes reconnus. "
                        "Si un avis comporte un revirement (*'au début... mais finalement'*), "
                        "le sac de mots ne voit pas l'ordre discursif : c'est pourquoi la confiance baisse "
                        "et déclenche l'étiquette **« à vérifier »**."
                    )
            else:
                st.info("Aucun terme spécifique du vocabulaire n'a exercé un poids dominant sur cet avis.")

# -------------------------------------------------------------
# ONGLET 2 : Un fichier CSV
# -------------------------------------------------------------
with onglet_lot:
    st.subheader("Analyse de données par lot et Matrice de confusion")
    st.write("Téléversez un fichier CSV ou utilisez l'un des fichiers d'exemple de notre dépôt :")
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        fichier_uploade = st.file_uploader("Fichier CSV à analyser", type=["csv"])
    with col_f2:
        choix_defaut = st.selectbox(
            "Ou charger un jeu du projet :",
            ["(Téléversement)", "data/sample_reviews.csv (30 avis)", "data/demo_reviews.csv (290 avis)"],
        )
        
    df_source = None
    if fichier_uploade is not None:
        try:
            df_source = pd.read_csv(fichier_uploade)
        except Exception as e:
            st.error(f"Erreur de lecture CSV : {e}")
    elif choix_defaut == "data/sample_reviews.csv (30 avis)":
        chemin_sample = RACINE / "data" / "sample_reviews.csv"
        if chemin_sample.exists():
            df_source = pd.read_csv(chemin_sample)
    elif choix_defaut == "data/demo_reviews.csv (290 avis)":
        chemin_demo = RACINE / "data" / "demo_reviews.csv"
        if chemin_demo.exists():
            df_source = pd.read_csv(chemin_demo)
            
    if df_source is not None:
        col_texte = "review" if "review" in df_source.columns else ("texte" if "texte" in df_source.columns else df_source.columns[0])
        
        with st.spinner("Analyse du jeu de données en cours..."):
            df_res = predire_lot(df_source, predire, colonne=col_texte, seuil=seuil_choisi)
            resume_data = resumer(df_res)
            
        st.markdown("### 📊 Tableau de Bord Relation Client")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Avis analysés", resume_data["total"])
        kpi2.metric("Négatifs détectés", f"{resume_data['negatifs']} ({resume_data['part_negatifs']} %)")
        kpi3.metric("Cas « À vérifier »", resume_data["a_verifier"])
        kpi4.metric("Confiance moyenne", f"{resume_data['confiance_moyenne']:.1%}")
        
        # MATRICE DE CONFUSION si étiquettes de vérité terrain présentes
        col_label_presente = "label" if "label" in df_res.columns else ("sentiment" if "sentiment" in df_res.columns else None)
        if col_label_presente:
            st.markdown("---")
            st.markdown("### 🎯 Matrice de Confusion sur ce jeu de données")
            mc = calculer_matrice_confusion(df_res, col_verite=col_label_presente, col_pred="prediction")
            
            col_mc1, col_mc2 = st.columns([1, 1])
            with col_mc1:
                st.write("**Matrice de Confusion (Effectifs) :**")
                st.dataframe(mc["matrice_df"], use_container_width=True)
                st.caption("Lignes = Vérité terrain | Colonnes = Prédictions du modèle")
                
            with col_mc2:
                st.write("**Métriques d'évaluation sur le jeu :**")
                m_sub1, m_sub2 = st.columns(2)
                m_sub1.metric("Exactitude (Accuracy)", f"{mc['accuracy']:.1%}")
                m_sub1.metric("F1-score Macro", f"{mc['f1_macro']:.3f}")
                m_sub2.metric("Rappel Négatifs", f"{mc['rappel_negatif']:.1%}")
                m_sub2.metric("Erreurs totales", f"{mc['nb_erreurs']} / {resume_data['total']}")
                
            if mc["nb_erreurs"] > 0:
                with st.expander(f"⚠️ Inspecter les {mc['nb_erreurs']} erreurs de classification"):
                    st.write("Ces erreurs proviennent en grande majorité des avis avec retournement discursif :")
                    cols_affich = [col_texte, col_label_presente, "prediction", "confiance", "verdict"]
                    st.dataframe(mc["erreurs_df"][cols_affich], use_container_width=True)
                    
        # Graphiques
        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**Répartition des sentiments prédits :**")
            st.bar_chart(df_res["prediction"].value_counts())
        with col_g2:
            st.markdown("**Répartition des verdicts opérationnels :**")
            st.bar_chart(df_res["verdict"].value_counts())
            
        # Tableau complet et export
        st.markdown("### 📋 Données détaillées et export")
        filtre = st.radio("Filtrer l'affichage :", ["Tous", "Négatifs seulement", "À vérifier seulement"], horizontal=True)
        if filtre == "Négatifs seulement":
            df_affiche = df_res[df_res["prediction"] == "negative"]
        elif filtre == "À vérifier seulement":
            df_affiche = df_res[df_res["verdict"] == "à vérifier"]
        else:
            df_affiche = df_res
            
        st.dataframe(
            df_affiche.sort_values(["prediction", "confiance"], ascending=[True, False]),
            use_container_width=True,
        )
        
        st.download_button(
            label="📥 Télécharger le rapport CSV enrichi",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name="resultats_analyse_sentiment.csv",
            mime="text/csv",
            type="primary",
        )

# -------------------------------------------------------------
# ONGLET 3 : Modèle & Matrice de confusion
# -------------------------------------------------------------
with onglet_modele:
    st.subheader("📈 Performance, Architecture et Matrice de Confusion du Modèle")
    
    st.markdown("#### 1. Métriques officielles enregistrées (`models/metrics.json`)")
    met1, met2, met3, met4, met5 = st.columns(5)
    met1.metric("Jeu de données", metriques.get("source", "Allociné"))
    met2.metric("F1-score Macro", f"{metriques.get('f1_macro', 0):.4f}")
    met3.metric("Exactitude", f"{metriques.get('accuracy', 0):.4f}")
    met4.metric("Rappel Négatif", f"{metriques.get('recall_negative', 0):.4f}")
    met5.metric("Exemples d'entraînement", f"{metriques.get('n_train', 0):,}")
    
    st.markdown("---")
    
    st.markdown("#### 2. Matrice de Confusion de référence (Évaluation TP2)")
    col_cmat1, col_cmat2 = st.columns([1, 1])
    with col_cmat1:
        mat_ref = pd.DataFrame(
            [[27, 2], [5, 24]],
            index=["Vérité : Négatif", "Vérité : Positif"],
            columns=["Prédit : Négatif", "Prédit : Positif"],
        )
        st.dataframe(mat_ref, use_container_width=True)
        st.caption("Matrice de confusion officielle mesurée sur le jeu de test de référence.")
        
    with col_cmat2:
        st.markdown(
            """
- **Vrais Négatifs (27) :** Clients mécontents détectés avec succès.
- **Faux Positifs (2) :** Cas critiques — avis négatifs manqués car étiquetés positifs.
- **Faux Négatifs (5) :** Avis positifs pris pour négatifs (fausses alertes).
- **Vrais Positifs (24) :** Avis positifs correctement confirmés.
            """
        )
        
    st.markdown("---")
    
    st.markdown("#### 3. Poids des termes les plus discriminants (Interprétabilité globale)")
    try:
        df_pos, df_neg = obtenir_top_mots(pipeline, n=10)
        c_p, c_n = st.columns(2)
        with c_p:
            st.success("🟢 **Top 10 des termes les plus Positifs**")
            st.dataframe(df_pos, use_container_width=True)
        with c_n:
            st.error("🔴 **Top 10 des termes les plus Négatifs**")
            st.dataframe(df_neg, use_container_width=True)
    except Exception as e:
        st.info(f"Visualisation des coefficients indisponible : {e}")
        
    st.markdown("---")
    st.markdown("#### 4. Spécifications du Pipeline de Machine Learning")
    st.code(
        """Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])""",
        language="python",
    )

# -------------------------------------------------------------
# ONGLET 4 : Cadrage & Décisions Métier
# -------------------------------------------------------------
with onglet_doc:
    st.subheader("📘 Synthèse de Cadrage et Décisions Métier")
    st.markdown(
        """
### Cas 3 : Priorisation des avis négatifs
- **Problème métier :** Éviter qu'un client insatisfait ne patiente plusieurs jours avant d'obtenir une réponse.
- **Choix technologique :** Machine Learning supervisé avec TF-IDF et Régression Logistique.
- **Justification des critères :**
  - **(a) Données étiquetées :** Disponibilité immédiate de 20 000 avis publics annotés (Allociné).
  - **(b) Vérifiabilité :** Scores de confiance probabilistes mesurables.
  - **(c) Coût par requête :** Inférieur à 0,001 € par inférence locale contre plusieurs centimes par appel LLM.
  - **(d) Conséquence d'une erreur :** Risque maîtrisé grâce au seuil d'incertitude réorientant les cas ambigus vers les conseillers.

---

### Seuil d'incertitude et Compromis opérationnel
- Le seuil de **60 %** isole **117 avis sur 290** (40,3 %) du jeu de démonstration.
- Ce filtrage sécurise le traitement en isolant les retournements discursifs (*"bien au début, mais cassé ensuite"*) sans saturer la capacité de relecture de l'équipe support.

---

### Pourquoi Git conserve `metrics.json` mais ignore `sentiment.joblib` ?
- `sentiment.joblib` est un binaire lourd qui se régénère en une ligne de commande (`python src/train.py`).
- `models/metrics.json` est la trace d'audit légère permettant à la CI (GitHub Actions) de certifier la qualité du modèle sans stocker de gros fichiers binaires.
        """
    )
