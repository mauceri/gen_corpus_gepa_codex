#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Générateur de corpus JSONL à partir d'un prompt (DSPy)

Fonctionnalités
- Lit le prompt depuis `prompt.txt` (instructions strictes pour produire un JSON à 7 clés).
- Charge les thèmes depuis `thèmes.json` (format: affectation Python `themes = [...]`).
- Charge les catégories depuis `catégories.jsonl` (liste non strictement JSON, tolérante).
- Charge les types par catégorie depuis `types_de_document_par_catégorie.json` (format: affectation Python `source_types_par_categorie = {...}`).
- Utilise DSPy pour générer des sorties strictement JSON (une par ligne) et construit un corpus de N entrées.

Prérequis d'exécution
- Python 3.9+
- dspy (pip install dspy-ai)
- Un LLM configuré via DSPy. Par défaut: OpenAI avec variable d'environnement `OPENAI_API_KEY`.

Exemples d'usage
    python generate_corpus_dspy.py \
        --count 50 \
        --output corpus.jsonl \
        --provider openai \
        --model gpt-4o-mini \
        --temperature 0.7

Notes
- Le script tente de valider que la sortie du modèle est un JSON valide
  avec exactement les 7 clés demandées. En cas d'échec, il réessaie jusqu'à
  un nombre d'essais maximum par entrée.
- Les chemins en entrée sont paramétrables en CLI.
"""

from __future__ import annotations

import warnings
import argparse
import ast
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Assure un répertoire de cache DSPy local et accessible avant d'importer dspy
if "DSPY_CACHEDIR" not in os.environ:
    _cache_dir = os.path.join(os.getcwd(), ".dspy_cache")
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except Exception:
        pass
    os.environ["DSPY_CACHEDIR"] = _cache_dir

import dspy
from dspy.clients import LM, configure_cache as dspy_configure_cache
dspy.settings.cache = None


# ----------------------------
# Chargement des fichiers
# ----------------------------

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _extract_python_rhs_container(text: str, open_ch: str, close_ch: str) -> str:
    """Extrait un littéral conteneur Python (liste/dict) à partir du texte.
    Tente de trouver la première occurrence équilibrée.
    """
    start = text.find(open_ch)
    if start == -1:
        raise ValueError("Début de conteneur non trouvé")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Fin de conteneur non trouvée (déséquilibré)")


def load_themes(path: str) -> List[str]:
    """Charge une liste de thèmes depuis un fichier de type `themes = [ ... ]`."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        list_src = _extract_python_rhs_container(data, "[", "]")
        themes: List[str] = ast.literal_eval(list_src)
        if not isinstance(themes, list) or not all(isinstance(x, str) for x in themes):
            raise ValueError("Format des thèmes invalide")
        return themes
    except Exception as e:
        raise RuntimeError(f"Impossible de charger les thèmes depuis {path}: {e}")


def load_categories(path: str) -> List[str]:
    """Charge une liste de catégories depuis un pseudo-JSONL du type:
    [
    extrait,
    commentaire,
    synthese,
    texte_libre,
    ]
    Retour: ex. ["extrait", "commentaire", "synthese", "texte_libre"]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Tente JSON strict d'abord
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # Parse tolérant
    body = raw.strip()
    # Retire [ ] si présents
    body = body.lstrip().lstrip("[").rstrip().rstrip("]")
    # Sépare par virgule ou fin de ligne
    items: List[str] = []
    for line in body.splitlines():
        token = line.strip().rstrip(",")
        if not token:
            continue
        # Enlève guillemets si présents
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            token = token[1:-1]
        items.append(token)

    # Nettoyage final
    items = [x.strip() for x in items if x.strip()]
    # filtre éventuels éléments non alphanumériques
    items = [re.sub(r"[^a-zA-Z0-9_\-àâäéèêëîïôöùûüç]+$", "", x) for x in items]
    return items


def load_types_by_category(path: str) -> Dict[str, List[str]]:
    """Charge un dict depuis un fichier de type `source_types_par_categorie = { ... }`."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        dict_src = _extract_python_rhs_container(data, "{", "}")
        mapping: Dict[str, List[str]] = ast.literal_eval(dict_src)
        if not isinstance(mapping, dict) or not all(isinstance(v, list) for v in mapping.values()):
            raise ValueError("Format du mapping invalide")
        return mapping
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le mapping types/catégories depuis {path}: {e}")


# ----------------------------
# DSPy: Signature et prédicteur
# ----------------------------


class CorpusSignature(dspy.Signature):
    """Utiliser le prompt ci-dessous pour produire strictement un objet JSON (7 clés).

    Le modèle doit suivre ces instructions telles quelles, sans ajouter de texte hors JSON.
    """

    theme:str = dspy.InputField(desc="Thème principal de la note")
    categorie:str = dspy.InputField(desc="Type de note: 'extrait', 'commentaire', 'synthese', ou 'texte_libre' ")
    type_du_document: str = dspy.InputField(desc="Type du document source: article Wikipédia, roman original, tweet, billet de blog, etc.")
    
    contenu:str = dspy.OutputField(desc="Contenu de la note, ce contenu doit être en accord avec le thème 'theme' mais bien sûr aussi avec la catégorie 'categorie' et le type de document source_type")
    url:str = dspy.OutputField(desc="l'url ou la référence du document au sujet duquel la note est écrite")
    date:str = dspy.OutputField(desc="Date d'écriture de la note est une date au format ISO totalement imaginaire")
    expressions_clefs:list = dspy.OutputField(desc="Liste d'expressions catactéristiques apparaissant dans le champ contenu")


def build_predictor(instruction: str) -> dspy.Predict:
    # Injecte l'instruction complète (prompt.txt) en préambule pour guider le modèle
    # On concatène à la docstring de la signature pour un contrôle maximal.
    # NB: DSPy utilise le champ docstring comme instructions globales.
    CorpusSignature.__doc__ = (CorpusSignature.__doc__ or "") + "\n\n" + instruction
    return dspy.Predict(CorpusSignature)


# ----------------------------
# Validation simple de la sortie
# ----------------------------

REQUIRED_KEYS = {
    "contenu",
    "url",
    "date",
    "expressions_clefs",
    "type_du_document",
    "theme",
    "categorie",
}




# ----------------------------
# Génération du corpus
# ----------------------------


@dataclass
class Config:
    count: int = 50
    output: str = "corpus.jsonl"
    prompt_path: str = "prompt.txt"
    themes_path: str = "thèmes.json"
    categories_path: str = "catégories.jsonl"
    types_map_path: str = "types_de_document_par_catégorie.json"
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_attempts_per_item: int = 3


def configure_lm(provider: str, model: str, temperature: float) -> None:
    provider = provider.lower()
    # Compose le nom Litellm: ex. "openai/gpt-4o-mini"
    composed = f"{provider}/{model}" if "/" not in model else model
    # Désactive le cache disque pour éviter les erreurs en environnement en lecture seule
    try:
        dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    except Exception:
        pass
    lm = LM(composed, temperature=temperature,cache=False)
    dspy.settings.configure(lm=lm)


def choose_triplet(themes: List[str], categories: List[str], types_map: Dict[str, List[str]]) -> tuple[str, str, str]:
    theme = random.choice(themes)
    categorie = random.choice(categories)
    types = types_map.get(categorie, ["null"]) or ["null"]
    type_du_document = random.choice(types)
    return theme, categorie, type_du_document


def generate_one(predictor: dspy.Predict, theme: str, categorie: str, type_du_document: str) -> str:
    pred = predictor(theme=theme, categorie=categorie, type_du_document=type_du_document)
    #print(f"DEBUG: raw output: {pred}" )
    # Le champ de sortie s'appelle 'output_json'
    return {
            "contenu": pred.contenu,
            "expressions_clefs": pred.expressions_clefs,
            "url": pred.url,
            "date": pred.date,
            "categorie": categorie,
            "theme": theme,
            "type_du_document": type_du_document
        }





def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Génère un corpus JSONL à partir d'un prompt DSPy")
    parser.add_argument("--count", type=int, default=50, help="Nombre d'entrées à générer")
    parser.add_argument("--output", type=str, default="corpus.jsonl", help="Fichier de sortie JSONL")
    parser.add_argument("--prompt", type=str, default="prompt.txt", help="Chemin vers le prompt")
    parser.add_argument("--themes", type=str, default="thèmes.json", help="Chemin vers le fichier des thèmes")
    parser.add_argument(
        "--categories", type=str, default="catégories.jsonl", help="Chemin vers la liste des catégories"
    )
    parser.add_argument(
        "--types-map",
        type=str,
        default="types_de_document_par_catégorie.json",
        help="Chemin vers le mapping types par catégorie",
    )
    parser.add_argument("--provider", type=str, default="openai", help="Fournisseur LLM (ex: openai)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Nom du modèle")
    parser.add_argument("--temperature", type=float, default=0.7, help="Température du modèle")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Nombre maximum d'essais par entrée en cas d'échec de validation",
    )

    args = parser.parse_args(argv)
    cfg = Config(
        count=args.count,
        output=args.output,
        prompt_path=args.prompt,
        themes_path=args.themes,
        categories_path=args.categories,
        types_map_path=args.types_map,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_attempts_per_item=args.max_attempts,
    )

    # Chargement des données
    prompt_text = load_prompt(cfg.prompt_path)
    themes = load_themes(cfg.themes_path)
    categories = load_categories(cfg.categories_path)
    types_map = load_types_by_category(cfg.types_map_path)

    # Configuration du LM
    configure_lm(cfg.provider, cfg.model, cfg.temperature)
    """philocal = dspy.LM(
        model="Phi-4-mini-instruct-Q6_K.gguf",
        base_url="http://sanroque:8080/v1",
        custom_llm_provider="openai",
        temperature=0.2
    )
    dspy.settings.configure(lm=philocal)
    """
    predictor = build_predictor(prompt_text)

    # Génération
    random.seed()
    generated = 0
    attempts_global = 0
    max_global_attempts = cfg.count * cfg.max_attempts_per_item * 3

    warnings.filterwarnings("ignore")
    stime = time.time()
    with open(cfg.output, "w", encoding="utf-8") as out:
        while generated < cfg.count and attempts_global < max_global_attempts:
            theme, categorie, type_du_document = choose_triplet(themes, categories, types_map)
            success = False
            for attempt in range(cfg.max_attempts_per_item):
                attempts_global += 1
                try:
                    raw = generate_one(predictor, theme, categorie, type_du_document=type_du_document)
                except Exception as e:
                    if attempt == cfg.max_attempts_per_item - 1:
                        print(f"Erreur de génération (dernier essai) pour {categorie}/{type_du_document}: {e}", file=sys.stderr)
                    continue

                obj = raw
                if generated % 10 == 0:
                    print(f"{generated} obj in {time.time() - stime:.2f}s: {obj}")
                if obj is None:
                    if attempt == cfg.max_attempts_per_item - 1:
                        print("Échec de parsing JSON (dernier essai).", file=sys.stderr)
                    continue

                
                # OK -> écrit la ligne
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                generated += 1
                    
                success = True
                break

            if not success:
                # abandonne cet échantillon et passe au suivant
                continue

    print(f"Génération terminée: {generated} entrées valides écrites dans {cfg.output}")
    if generated < cfg.count:
        print(
            f"Attention: seule une partie a pu être générée (demandé={cfg.count}, obtenu={generated}).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
