**Aperçu**
- Génère un corpus JSONL à partir de `prompt.txt` en utilisant DSPy et un LLM OpenAI.
- Thèmes, catégories et types de document sont lus depuis les fichiers fournis, puis combinés pour produire des entrées valides conforme au prompt.

**Fichiers**
- `generate_corpus_dspy.py` — programme principal de génération.
- `prompt.txt` — consignes strictes pour produire exactement 7 clés JSON.
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
- Générer 50 entrées (OpenAI, modèle par défaut):
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python generate_corpus_dspy.py --count 50 --output corpus.jsonl`
- Test rapide sur 10 entrées:
  - `DSPY_CACHEDIR=.dspy_cache .venv/bin/python generate_corpus_dspy.py --count 10 --output corpus_test.jsonl`

**Paramètres utiles (CLI)**
- `--count` nombre d’entrées (par défaut 50)
- `--output` chemin du fichier JSONL de sortie (par défaut `corpus.jsonl`)
- `--model` nom de modèle OpenAI (par défaut `gpt-4o-mini`)
- `--temperature` température d’échantillonnage (par défaut 0.7)
- `--prompt` chemin vers le prompt (par défaut `prompt.txt`)
- `--themes` chemin vers les thèmes (par défaut `thèmes.json`)
- `--categories` chemin vers les catégories (par défaut `catégories.jsonl`)
- `--types-map` chemin du mapping types (par défaut `types_de_document_par_catégorie.json`)
- `--max-attempts` tentatives par entrée en cas d’échec de validation (par défaut 3)

**Fonctionnement**
- LLM via DSPy: `dspy.LM("openai/<model>")`. La clé est lue dans `OPENAI_API_KEY`.
- Le prompt de `prompt.txt` est injecté dans la Signature DSPy et appliqué à chaque appel.
- Le programme échantillonne aléatoirement: `(theme, categorie, type_de_document)` conformément aux fichiers.
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

