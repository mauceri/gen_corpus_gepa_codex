Notes pour futures sessions (pipeline GEPA + génération de corpus)

- Valider strictement les sorties : ajouter un validateur JSON et des règles métier (7 clés, types, longueur par catégorie, `expressions_clefs` présentes dans `contenu`, gestion de null) avant d’accepter une entrée.
- Métrique GEPA plus riche : inclure des vérifications structurelles (présence des champs, respect du thème/catégorie/type, expressions clés), et pondérer ces règles pour éviter un prompt qui optimise seulement la similarité sémantique.
- Reproductibilité et traçabilité : fixer un seed optionnel, logguer dans le corpus les métadonnées (version du prompt, modèles utilisés, date, paramètres) pour comparer les runs.
- Contrôle de coût/quota : ajouter un compteur d’appels et/ou une limite de tokens ou de coût pour éviter les dérives de facture si un bug survient.
- Distribution et qualité des données : prévoir déduplication, détection d’outliers et contrôle de la distribution des thèmes/types pour limiter les biais.
- Sécurité/contenus : passer un filtrage (PII, toxicité, hallucinations) sur le corpus synthétique avant usage aval.
