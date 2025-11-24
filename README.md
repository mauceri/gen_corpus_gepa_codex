**Aperçu**
- `promptGenGEPA.py` optimise un prompt avec DSPy.GEPA, sauvegarde le meilleur prompt trouvé (`GEPAPrompt.txt`) et archive le prompt initial. Il peut ensuite générer un corpus de notes avec ce prompt optimisé.
- `generate_corpus_dspy.py` génère un corpus JSONL à partir d’un prompt (par défaut `prompt.txt`, ou `GEPAPrompt.txt` si vous utilisez celui produit par GEPA) en combinant thèmes, catégories et types de document.

**Fichiers**
- `promptGenGEPA.py` — optimise le prompt via GEPA, exporte le meilleur prompt et peut générer un corpus.
- `generate_corpus_dspy.py` — programme de génération basé sur un prompt existant.
- `prompt.txt` — consignes strictes pour produire exactement 7 clés JSON (peut être remplacé par `GEPAPrompt.txt` issu de GEPA).
- `GEPAPrompt.txt` — meilleur prompt trouvé par GEPA (créé par `promptGenGEPA.py`).
- `prompt-init.txt` — prompt initial utilisé pour l’optimisation GEPA (copié dans `prompt-init.original.txt` lors de l’exécution).
- `thèmes.json` — liste de thèmes (format « Python assignment », pas du JSON strict).
- `catégories.jsonl` — liste tolérante des catégories (extrait, commentaire, synthese, texte_libre).
- `types_de_document_par_catégorie.json` — mapping catégorie → types (format « Python assignment »).
- `corpus.jsonl` — sortie (une ligne JSON par entrée, créé à l’exécution).

**Prérequis**
- Python 3.9+
- Compte OpenAI et variable d’environnement `OPENAI_API_KEY` définie.

**Installation**
- Créer un environnement et installer les dépendances:
  - `python3 -m venv .venv`
  - `.venv/bin/pip install dspy-ai openai`

**Exécution rapide**
- Optimiser le prompt + générer un corpus (GEPA, avec sauvegarde du meilleur prompt):
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python promptGenGEPA.py --count 50 --gepa-prompt GEPAPrompt.txt --output corpus_gepa.jsonl`
- Générer 50 entrées avec un prompt existant (OpenAI, modèle par défaut):
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python generate_corpus_dspy.py --count 50 --output corpus.jsonl`
- Test rapide sur 10 entrées:
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python generate_corpus_dspy.py --count 10 --output corpus_test.jsonl`

**Paramètres utiles (CLI)**
Pour `promptGenGEPA.py`:
- `--prompt` prompt initial (par défaut `prompt-init.txt`)
- `--gepa-prompt` fichier où écrire le meilleur prompt GEPA (par défaut `GEPAPrompt.txt`)
- `--prompt-backup` copie du prompt initial (par défaut `prompt-init.original.txt`)
- `--output` corpus généré après optimisation (par défaut `corpus-essai.txt`)
- `--count` nombre de notes générées (par défaut 100)
- `--generator-model` modèle principal pour GEPA/génération (par défaut `gpt-5-mini-2025-08-07`)
- `--comparator-model` modèle pour la métrique sémantique (par défaut `gpt-4o-mini`)
- `--max-metric-calls` borne d’appels métriques GEPA (par défaut 500)

Pour `generate_corpus_dspy.py`:
- `--count` nombre d’entrées (par défaut 50)
- `--output` chemin du fichier JSONL de sortie (par défaut `corpus.jsonl`)
- `--model` nom de modèle OpenAI (par défaut `gpt-4o-mini`)
- `--temperature` température d’échantillonnage (par défaut 0.7)
- `--prompt` chemin vers le prompt (par défaut `prompt.txt`; utilisez `GEPAPrompt.txt` si vous voulez le prompt optimisé)
- `--themes` chemin vers les thèmes (par défaut `thèmes.json`)
- `--categories` chemin vers les catégories (par défaut `catégories.jsonl`)
- `--types-map` chemin du mapping types (par défaut `types_de_document_par_catégorie.json`)
- `--max-attempts` tentatives par entrée en cas d’échec de validation (par défaut 3)

**Fonctionnement**
- LLM via DSPy: `dspy.LM("openai/<model>")`. La clé est lue dans `OPENAI_API_KEY`.
- `promptGenGEPA.py` entraîne un générateur avec GEPA, extrait le meilleur prompt (`GEPAPrompt.txt`) puis peut générer un corpus avec ce prompt optimisé. Le prompt initial est sauvegardé pour référence.
- `generate_corpus_dspy.py` injecte le prompt fourni (par défaut `prompt.txt` ou `GEPAPrompt.txt`) dans la Signature DSPy. Le programme échantillonne aléatoirement: `(theme, categorie, type_de_document)` conformément aux fichiers.
- Pour chaque triplet, il génère une unique sortie JSON (texte) et la valide selon les règles:
  - 7 clés exactes: `contenu`, `url`, `date`, `expressions_clefs`, `type_de_document`, `theme`, `categorie`.
  - Types corrects; `type_de_document` gère `null` et la catégorie `texte_libre`.
  - `contenu` commence par l’étiquette de catégorie (ex. “Synthèse :”, “Commentaire :”, “Extrait :”, “Note personnelle :”).
  - `expressions_clefs` (1–8) apparaissent littéralement dans `contenu`.
  - URL plausible (http/https) et date plausible (YYYY-MM-DD, 1900–2100).
  - `theme` et `categorie` renvoyés identiques aux entrées.
- En cas d’échec (parsing/validation), le programme ré-essaie jusqu’à `--max-attempts`, puis passe à un autre triplet.

**Remarques importantes**
- Le fichier des catégories emploie `synthese` (sans accent) comme clé d’entrée, mais le contenu exigera “Synthèse :” (avec accent). Le validateur s’en charge.
- Le cache DSPy est redirigé vers un dossier local `.dspy_cache` pour éviter les problèmes d’écriture. Vous pouvez aussi prefixer la commande avec `DSPY_CACHEDIR=.dspy_cache`.
- La génération est non déterministe (échantillonnage aléatoire). Pour des runs reproductibles, vous pouvez modifier le code pour fixer `random.seed(<valeur>)`.

**Dépannage**
- Erreur d’authentification OpenAI: assurez-vous que `OPENAI_API_KEY` est bien exportée dans le shell courant.
- Erreur de cache/permissions: gardez `DSPY_CACHEDIR` pointant vers un dossier local en écriture (ex. `.dspy_cache`).
- Warnings “structured output format”: DSPy bascule automatiquement en mode JSON classique, c’est attendu.
- Validation incomplète: le script signale le nombre d’entrées écrites; si moindre que `--count`, relancez ou augmentez `--max-attempts`.

**Exemples**
- Modèle différent:
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python generate_corpus_dspy.py --count 20 --model gpt-4o`
- Fichiers personnalisés:
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python generate_corpus_dspy.py --themes data/themes.py --categories data/categories.txt --types-map data/types.py`
