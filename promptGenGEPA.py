#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimisation de prompt via GEPA puis génération de corpus d'exemples.

Fonctionnalités principales
- Charge le prompt initial, les thèmes, les catégories, les types de documents et les exemples d'entraînement.
- Optimise le prompt avec DSPy.GEPA en utilisant une métrique sémantique simple.
- Sauvegarde le meilleur prompt trouvé (GEPAPrompt) et conserve une copie du prompt initial.
- Génère un corpus de notes en s'appuyant sur le générateur compilé.

Prérequis
- OPENAI_API_KEY dans l'environnement.
- dspy installé.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from dspy.clients import LM, configure_cache as dspy_configure_cache
from functools import wraps

logging.basicConfig(
    filename="gepa_llm_calls.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Désactive le cache disque (souvent interdit en sandbox)
try:
    dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
except Exception:
    pass
dspy.settings.cache = None


# =============================================================================
# Chargement fichiers utilitaires
# =============================================================================

def load_json_or_text(path: str) -> str | Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def is_placeholder_prompt(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if "given the fields" in lower and "produce the fields" in lower:
        return True
    return len(lower.strip()) < 40


def _extract_python_rhs_container(text: str, open_ch: str, close_ch: str) -> str:
    """Extrait un littéral conteneur Python (liste/dict) à partir du texte."""
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
    """Charge une liste de catégories depuis un pseudo-JSONL tolérant."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    body = raw.strip()
    body = body.lstrip().lstrip("[").rstrip().rstrip("]")
    items: List[str] = []
    for line in body.splitlines():
        token = line.strip().rstrip(",")
        if not token:
            continue
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            token = token[1:-1]
        items.append(token)

    items = [x.strip() for x in items if x.strip()]
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


def load_exemples(path: str) -> List[dict]:
    """Charge une liste d'exemples depuis un fichier de type `exemples = [ ... ]`."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        list_src = _extract_python_rhs_container(data, "[", "]")
        exemples: List[dict] = ast.literal_eval(list_src)
        if not isinstance(exemples, list) or not all(isinstance(x, dict) for x in exemples):
            raise ValueError("Format des exemples invalide")
        return exemples
    except Exception as e:
        raise RuntimeError(f"Impossible de charger les exemples depuis {path}: {e}")


# =============================================================================
# Config et parsing CLI
# =============================================================================


@dataclass
class Config:
    count: int = 100
    batch_size: int = 20
    report_every: int = 10
    prompt_path: str = "prompt-init.txt"
    prompt_backup_path: str = "prompt-init.original.txt"
    gepa_prompt_path: str = "GEPAPrompt.txt"
    compiled_prompt_path: str = "compiled_generator_prompt.txt"
    themes_path: str = "thèmes.json"
    categories_path: str = "catégories.jsonl"
    types_map_path: str = "types_de_document_par_catégorie.json"
    exemples_path: str = "exemples.json"
    output_corpus: str = "corpus-essai.txt"
    log_file: str = "openai_prompts.jsonl"
    generator_model: str = "gpt-5-mini-2025-08-07"
    comparator_model: str = "gpt-4o-mini"
    reflection_temperature: float = 1.0
    max_metric_calls: int = 500
    cost_in: float = 0.25 / 1_000_000
    cost_out: float = 2.0 / 1_000_000


DEFAULT_CFG = Config()


def parse_args(argv: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Optimise un prompt via GEPA et génère un corpus.")
    parser.add_argument("--count", type=int, default=DEFAULT_CFG.count, help="Nombre de notes à générer.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG.batch_size, help="Taille de flush batch.")
    parser.add_argument("--report-every", type=int, default=DEFAULT_CFG.report_every, help="Fréquence d'affichage des stats.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_CFG.prompt_path, help="Prompt initial à optimiser.")
    parser.add_argument("--prompt-backup", type=str, default=DEFAULT_CFG.prompt_backup_path, help="Copie du prompt initial.")
    parser.add_argument("--gepa-prompt", type=str, default=DEFAULT_CFG.gepa_prompt_path, help="Fichier où écrire le meilleur prompt GEPA.")
    parser.add_argument("--compiled-prompt", type=str, default=DEFAULT_CFG.compiled_prompt_path, help="Copie du meilleur prompt (compatibilité).")
    parser.add_argument("--themes", type=str, default=DEFAULT_CFG.themes_path, help="Chemin vers les thèmes.")
    parser.add_argument("--categories", type=str, default=DEFAULT_CFG.categories_path, help="Chemin vers les catégories.")
    parser.add_argument("--types-map", type=str, default=DEFAULT_CFG.types_map_path, help="Chemin vers le mapping types/catégories.")
    parser.add_argument("--exemples", type=str, default=DEFAULT_CFG.exemples_path, help="Chemin vers les exemples d'entraînement.")
    parser.add_argument("--output", type=str, default=DEFAULT_CFG.output_corpus, help="Corpus généré.")
    parser.add_argument("--log-file", type=str, default=DEFAULT_CFG.log_file, help="Log des entrées/sorties générées.")
    parser.add_argument("--generator-model", type=str, default=DEFAULT_CFG.generator_model, help="Modèle principal pour GEPA et génération.")
    parser.add_argument("--comparator-model", type=str, default=DEFAULT_CFG.comparator_model, help="Modèle pour la métrique sémantique.")
    parser.add_argument("--max-metric-calls", type=int, default=DEFAULT_CFG.max_metric_calls, help="Borne sur les appels métriques GEPA.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CFG.reflection_temperature, help="Température du LM pour la réflexion GEPA.")
    args = parser.parse_args(argv)

    return Config(
        count=args.count,
        batch_size=args.batch_size,
        report_every=args.report_every,
        prompt_path=args.prompt,
        prompt_backup_path=args.prompt_backup,
        gepa_prompt_path=args.gepa_prompt,
        compiled_prompt_path=args.compiled_prompt,
        themes_path=args.themes,
        categories_path=args.categories,
        types_map_path=args.types_map,
        exemples_path=args.exemples,
        output_corpus=args.output,
        log_file=args.log_file,
        generator_model=args.generator_model,
        comparator_model=args.comparator_model,
        max_metric_calls=args.max_metric_calls,
        reflection_temperature=args.temperature,
    )


# =============================================================================
# Config LMs et logging
# =============================================================================


def ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Définissez la variable d'environnement OPENAI_API_KEY")
    return api_key


def configure_generator_lm(model: str, temperature: float) -> dspy.LM:
    api_key = ensure_api_key()
    lm = dspy.LM(model=model, api_key=api_key, temperature=temperature, max_tokens=16000, cache=False)
    dspy.settings.configure(lm=lm)
    return lm


def configure_comparator_lm(model: str) -> dspy.LM:
    api_key = ensure_api_key()
    lm = dspy.LM(model=model, api_key=api_key, temperature=0.0, max_tokens=128, cache=False)

    _old_lm_call = lm.__call__

    @wraps(_old_lm_call)
    def _lm_logging_call(*args, **kwargs):
        logging.info(f"[LM INPUT] args={args} kwargs={kwargs}")
        res = _old_lm_call(*args, **kwargs)
        logging.info(f"[LM OUTPUT] {res}")
        return res

    lm.__call__ = _lm_logging_call
    return lm


def patch_predict_logging() -> None:
    """Wrap dspy.Predict.__call__ pour tracer les prompts dans gepa_llm_calls.log."""
    _old_call = dspy.Predict.__call__

    @wraps(_old_call)
    def _logging_call(self, *args, **kwargs):
        logging.info(f"[DSPy.Predict INPUT] sig={getattr(self,'signature',None)} args={args} kwargs={kwargs}")
        out = _old_call(self, *args, **kwargs)
        try:
            logging.info(f"[DSPy.Predict OUTPUT] {out}")
        except Exception as e:
            logging.warning(f"[DSPy.Predict OUTPUT] impossible à sérialiser: {e}")
        return out

    dspy.Predict.__call__ = _logging_call


# =============================================================================
# Signatures DSPy
# =============================================================================


def build_note_generator(prompt_text: str) -> dspy.Module:
    class GenerateNote(dspy.Signature):
        f"""{prompt_text}"""

        theme: str = dspy.InputField(desc="Thème principal de la note")
        categorie: str = dspy.InputField(desc="Type de note: 'extrait', 'commentaire', 'synthese', ou 'texte_libre'")
        type_du_document: str = dspy.InputField(desc="Type du document source: article Wikipédia, roman original, tweet, billet de blog, etc.")

        contenu: str = dspy.OutputField(desc="Contenu de la note, aligné avec theme/categorie/type_du_document")
        url: str = dspy.OutputField(desc="URL ou référence du document (inventée)")
        date: str = dspy.OutputField(desc="Date imaginaire au format ISO")
        expressions_clefs: list = dspy.OutputField(desc="Expressions caractéristiques extraites du contenu")

    class NoteGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(GenerateNote)

        def forward(self, categorie: str, theme: str, type_du_document: str) -> dict:
            result = self.generate(categorie=categorie, theme=theme, type_du_document=type_du_document)
            return {
                "contenu": result.contenu,
                "url": result.url,
                "date": result.date,
                "expressions_clefs": result.expressions_clefs,
                "categorie": categorie,
                "theme": theme,
                "type_du_document": type_du_document,
            }

    return NoteGenerator()


class CompareTextsGlobal(dspy.Signature):
    """
    Compare deux textes (text_a, text_b) et produit UNE note globale 1..5
    (thème + ton + style). 5 = très ressemblant, 1 = très différent.
    Sortie : un entier 1..5, sans commentaire.
    """

    text_a: str = dspy.InputField(desc="Premier texte")
    text_b: str = dspy.InputField(desc="Second texte")
    score: int = dspy.OutputField(desc="Entier 1..5, score global")


class TextComparatorGlobal(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.pred = dspy.Predict(CompareTextsGlobal)
        self.lm = lm

    def forward(self, text_a: str, text_b: str) -> int:
        if self.lm is not None:
            with dspy.settings.context(lm=self.lm):
                out = self.pred(text_a=text_a, text_b=text_b)
        else:
            out = self.pred(text_a=text_a, text_b=text_b)

        try:
            s = int(getattr(out, "score", 3))
        except Exception:
            s = 3
        return max(1, min(5, s))


# =============================================================================
# Métrique
# =============================================================================


def mentions_expr_as_input(text: str) -> bool:
    return bool(re.search(r"(fournir|donnez|veuillez|merci de).*(expression[_ -]?cl[ée]s?)", text, flags=re.I))


def make_semantic_metric(comparator: TextComparatorGlobal):
    compteur_global = {"n": 0}

    def semantic_metric(
        gold: dict,
        pred: dict,
        trace: dict | None = None,
        pred_name: str | None = None,
        pred_trace: dict | None = None,
        *,
        w_semantic: float = 1.0,
        w_expr: float = 0.0,
    ) -> float:
        gold_text = str((gold or {}).get("contenu") or "")
        pred_text = str((pred or {}).get("contenu") or "")

        if not gold_text or not pred_text:
            return 0.1
        if mentions_expr_as_input(pred_text):
            return 0.0

        try:
            score_15 = comparator(text_a=gold_text, text_b=pred_text)
            s_semantic = score_15 / 5.0
        except Exception:
            s_semantic = 0.6

        score = float(w_semantic * s_semantic + w_expr * 0.0)

        if isinstance(trace, dict):
            trace.update({
                "semantic": s_semantic,
                "final": score,
                "pred_name": pred_name,
            })
        if isinstance(pred_trace, dict):
            pred_trace.update({"final": score})

        compteur_global["n"] += 1
        if compteur_global["n"] % 10 == 0:
            print(f"******************************************** au tour {compteur_global['n']} score = {score}")
        return score

    return semantic_metric


# =============================================================================
# Entraînement GEPA et sauvegarde du prompt
# =============================================================================


def build_trainset(exemples: List[dict]) -> List[dspy.Example]:
    return [
        dspy.Example(
            categorie=ex["categorie"],
            theme=ex["theme"],
            type_du_document=ex["type_du_document"],
            contenu=ex["contenu"],
            url=ex["url"],
            date=ex["date"],
        ).with_inputs("categorie", "theme", "type_du_document")
        for ex in exemples
    ]


def extract_best_prompt(compiled_generator, teleprompter, initial_prompt: str) -> str:
    candidates: List[str] = []

    # GEPA trackers
    for attr in ("best_prompt", "best_prompt_str", "best_prompt_text"):
        val = getattr(teleprompter, attr, None)
        if val:
            candidates.append(ensure_text(val))

    best_prompts_dict = getattr(teleprompter, "best_prompts", None)
    if isinstance(best_prompts_dict, dict):
        for val in best_prompts_dict.values():
            if val:
                candidates.append(ensure_text(val))

    # Compiled generator signatures
    sigs = [
        getattr(getattr(compiled_generator, "generate", None), "signature", None),
        getattr(compiled_generator, "signature", None),
    ]
    for sig in sigs:
        if sig is None:
            continue
        instr = getattr(sig, "instructions", None) or getattr(sig, "__doc__", None)
        if instr:
            candidates.append(ensure_text(instr))

    # Fallback to the initial prompt
    candidates.append(initial_prompt)

    for cand in candidates:
        if not is_placeholder_prompt(cand):
            return cand
    return ensure_text(initial_prompt)


def train_generator(prompt_text: str, exemples: List[dict], comparator: TextComparatorGlobal, cfg: Config):
    generator = build_note_generator(prompt_text)
    trainset = build_trainset(exemples)

    teleprompter = dspy.GEPA(
        metric=make_semantic_metric(comparator),
        reflection_lm=dspy.settings.lm,
        max_metric_calls=cfg.max_metric_calls,
        track_stats=True,
        track_best_outputs=True,
    )

    try:
        compiled_generator = teleprompter.compile(generator, trainset=trainset)
    except RuntimeError as e:
        raise RuntimeError(f"Échec GEPA (quota ou autre): {e}") from e

    best_prompt = extract_best_prompt(compiled_generator, teleprompter, ensure_text(prompt_text))
    return compiled_generator, best_prompt


def save_prompts(initial_prompt: str, best_prompt: str, cfg: Config) -> None:
    best_prompt = ensure_text(best_prompt)
    prompt_backup = Path(cfg.prompt_backup_path) if cfg.prompt_backup_path else None
    prompt_path = Path(cfg.prompt_path)
    if prompt_backup and prompt_backup.resolve() != prompt_path.resolve():
        prompt_backup.write_text(initial_prompt, encoding="utf-8")
    Path(cfg.gepa_prompt_path).write_text(best_prompt, encoding="utf-8")
    if cfg.compiled_prompt_path and cfg.compiled_prompt_path != cfg.gepa_prompt_path:
        Path(cfg.compiled_prompt_path).write_text(best_prompt, encoding="utf-8")
    print(f"Prompt optimisé sauvegardé dans {cfg.gepa_prompt_path}")
    if prompt_backup:
        print(f"Prompt initial sauvegardé dans {prompt_backup}")


# =============================================================================
# Génération du corpus
# =============================================================================


def pick_type_for_category(categorie: str, types_map: Dict[str, List[str]]) -> str:
    candidates = types_map.get(categorie) or []
    return random.choice(candidates) if candidates else "texte"


def generate_corpus(compiled_generator, cfg: Config, categories: List[str], types_map: Dict[str, List[str]], themes: List[str]) -> None:
    total_in_tokens = 0
    total_out_tokens = 0
    generated_notes: List[dict] = []
    stime = time.time()

    with open(cfg.output_corpus, "w", encoding="utf-8") as fout, open(cfg.log_file, "w", encoding="utf-8") as flog:
        for i in range(cfg.count):
            cat = random.choice(categories)
            st = pick_type_for_category(cat, types_map)
            th = random.choice(themes)

            note = compiled_generator(categorie=cat, theme=th, type_du_document=st)
            flog.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "input": {"categorie": cat, "theme": th, "type_du_document": st},
                "output": note,
            }, ensure_ascii=False) + "\n")

            tin = sum(len(str(v).split()) for v in [cat, st, th])
            tout = len(str(note.get("contenu", "")).split())
            total_in_tokens += tin
            total_out_tokens += tout

            generated_notes.append(note)

            if (i + 1) % cfg.batch_size == 0:
                for gnote in generated_notes:
                    fout.write(json.dumps(gnote, ensure_ascii=False) + "\n")
                generated_notes = []

            if (i + 1) % cfg.report_every == 0:
                elapsed = time.time() - stime
                cost_est = total_in_tokens * cfg.cost_in + total_out_tokens * cfg.cost_out
                print(f"[{i+1}/{cfg.count}] {elapsed:.1f}s, tokens_in={total_in_tokens}, tokens_out={total_out_tokens}, coût≈{cost_est:.2f}$")

        for gnote in generated_notes:
            fout.write(json.dumps(gnote, ensure_ascii=False) + "\n")

    print(f"Corpus sauvegardé en {time.time()-stime:.1f}s dans {cfg.output_corpus}")


# =============================================================================
# Main
# =============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    patch_predict_logging()
    cfg = parse_args(argv)

    prompt_obj = load_json_or_text(cfg.prompt_path)
    initial_prompt = ensure_text(prompt_obj)
    themes = load_themes(cfg.themes_path)
    categories = load_categories(cfg.categories_path)
    types_map = load_types_by_category(cfg.types_map_path)
    exemples = load_exemples(cfg.exemples_path)

    configure_generator_lm(cfg.generator_model, cfg.reflection_temperature)
    lm_comparator = configure_comparator_lm(cfg.comparator_model)
    comparator = TextComparatorGlobal(lm=lm_comparator)

    compiled_generator, best_prompt = train_generator(initial_prompt, exemples, comparator, cfg)
    save_prompts(initial_prompt, best_prompt, cfg)
    generate_corpus(compiled_generator, cfg, categories, types_map, themes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
