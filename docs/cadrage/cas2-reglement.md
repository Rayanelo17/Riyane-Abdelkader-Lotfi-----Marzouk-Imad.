# Fiche de cadrage : Questions sur le règlement intérieur

> Remplir chaque rubrique en une à trois phrases. Les rubriques marquées (obligatoire) sont vérifiées par `pytest`.

## Besoin en une phrase (obligatoire)
Classifier automatiquement le sentiment (positif, négatif ou neutre) exprimé dans les avis des utilisateurs afin d'accélérer le traitement des retours clients
## Utilisateur final (obligatoire)
Les équipes du service client et les responsables produit souhaitant suivre la satisfaction globale des usagers. Ils exploiteront ces analyses à travers un tableau de bord pour prioriser les interventions
## Approche retenue (obligatoire)

Cocher une seule case :

- [ ] Règles métier
- [x] Machine learning
- [ ] RAG
- [ ] Modèle génératif seul

## Justification (obligatoire, citer au moins deux critères de la grille : (a) données étiquetées, (b) vérifiabilité, (c) coût par requête, (d) conséquence d'une erreur)
L'existence de (a) données étiquetées en quantité suffisante permet d'entraîner un modèle supervisé adapté au domaine. Cette solution garantit un (c) coût par requête négligeable et une (b) vérifiabilité élevée des prédictions grâce à l'analyse des scores de confiance.
## Données nécessaires et leur origine (obligatoire)
Le projet nécessite un corpus de textes courts annotés selon leur polarité émotionnelle. Les données proviennent de jeux de données publics open-source (type avis Allociné/Amazon) complétés par des retours clients internes anonymisés.
## Métrique de succès et seuil d'acceptation (obligatoire)
La performance sera évaluée à l'aide du F1-score macro pour tenir compte du déséquilibre entre les classes. Le seuil d'acceptation minimal pour le passage en production est fixé à un F1-score de 0,82 sur le jeu de test.
## Conséquence d'une erreur et validation humaine prévue (obligatoire)
La (d) conséquence d'une erreur se limite à une mauvaise catégorification ponctuelle, sans impact direct grave sur l'utilisateur. Une validation humaine est prévue pour les cas où le score de confiance du modèle est inférieur à 70 % ou pour les messages identifiés comme très critiques.
## Risques éthiques ou de confidentialité (obligatoire)
Le risque principal concerne la présence accidentelle de données personnelles (PII) dans les commentaires soumis par les utilisateurs. Un module d'anonymisation et de suppression des PII sera appliqué systématiquement avant tout traitement ou stockage.
## Approche écartée et pourquoi (facultatif)
L'utilisation d'un modèle génératif seul (LLM via API) a été écartée en raison d'un (c) coût par requête beaucoup plus élevé à grande échelle. De plus, la latence de traitement et le risque d'hallucinations rendent cette option moins adaptée qu'un classifieur léger.
