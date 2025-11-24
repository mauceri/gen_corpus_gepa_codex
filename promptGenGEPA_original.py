#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
promptGenGEPA-0.2.py
Génère un corpus avec DSPy/GEPA en lisant les paramètres et prompt depuis des fichiers.
Sauvegarde corpus, logs OpenAI et générateur compilé. Affiche un résumé périodique.
"""

import os, json, time, random, datetime, pickle, ast, re
import dspy
from dspy.clients import LM, configure_cache as dspy_configure_cache
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
from functools import wraps

logging.basicConfig(
    filename="gepa_llm_calls.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


# ===================== CONFIGURATION =====================

PROMPT_FILE = "prompt-init.txt"                   # texte du prompt GEPA
THEMES_FILE = "thèmes.json"                  # liste themes
CATEGORIES_FILE = "catégories.jsonl"          # liste categories
TYPES_FILE = "types_de_document_par_catégorie.json"                    # dict cat -> liste types
EXEMPLES_FILE = "exemples.json"              # exemples d’entraînement

OUTPUT_CORPUS = "corpus-essai.txt"                 # corpus généré
LOG_FILE = "openai_prompts.jsonl"            # log JSONL prompt/réponse
GENERATOR_FILE = "compiled_generator.pkl"    # générateur compilé
MODEL_NAME = "gpt-5-mini-2025-08-07"         # modèle OpenAI
NB_NOTES = 100                              # nb de notes à générer
BATCH_SIZE = 20
REPORT_EVERY = 10                           # afficher stats toutes les n requêtes
COST_IN = 0.25 / 1_000_000                   # $ par token in
COST_OUT = 2.0 / 1_000_000                  # $ par token out

# ===================== LECTURE FICHIERS =====================

def load_json_or_text(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


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


def load_exemples(path: str) -> List[str]:
    """Charge une liste d'exemples depuis un fichier de type `exemples = [ ... ]`."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        list_src = _extract_python_rhs_container(data, "[", "]")
        exemples: List[str, Any] = ast.literal_eval(list_src)
        
        if not isinstance(exemples, list) or not all(isinstance(x, dict) for x in exemples):
            raise ValueError("Format des exemples invalide")
        return exemples
    except Exception as e:
        raise RuntimeError(f"Impossible de charger les exemples depuis {path}: {e}")



prompt_gepa = load_json_or_text(PROMPT_FILE)
themes = load_themes(THEMES_FILE)
categories = load_categories(CATEGORIES_FILE)
type_du_document_par_categorie = load_types_by_category(TYPES_FILE)
exemples = load_exemples(EXEMPLES_FILE)

# Sauvegarde l’ancienne __call__
_old_call = dspy.Predict.__call__

def _logging_call(self, *args, **kwargs):
    """Wrapper autour de dspy.Predict.__call__ pour logger prompt/résultat."""
    # Les inputs passés au Predict
    logging.info(f"[DSPy.Predict INPUT] sig={getattr(self,'signature',None)} args={args} kwargs={kwargs}")
    out = _old_call(self, *args, **kwargs)
    # On logge la sortie (ce que le LM a répondu)
    try:
        logging.info(f"[DSPy.Predict OUTPUT] {out}")
    except Exception as e:
        logging.warning(f"[DSPy.Predict OUTPUT] impossible à sérialiser: {e}")
    return out

# Patch
dspy.Predict.__call__ = _logging_call


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
        dspy_configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    except Exception:
        pass
    lm = LM(composed, temperature=temperature,cache=False)
    dspy.clear_cache()
    dspy.settings.cache = None  
    dspy.settings.configure(lm=lm)

# ===================== INITIALISATION LM =====================

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Définissez la variable d'environnement OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

lm = dspy.LM(model=MODEL_NAME, api_key=OPENAI_KEY, temperature=1.0, max_tokens=16000)
dspy.settings.configure(lm=lm)

# ===================== SIGNATURE / MODULE =====================

class GenerateNote(dspy.Signature):
    f"""{prompt_gepa}"""  # injection du prompt lu dans le fichier

    theme:str = dspy.InputField(desc="Thème principal de la note")
    categorie:str = dspy.InputField(desc="Type de note: 'extrait', 'commentaire', 'synthese', ou 'texte_libre'")
    type_du_document: str = dspy.InputField(
        desc="Type du document source: article Wikipédia, roman original, tweet, billet de blog, etc."
    )

    contenu:str = dspy.OutputField(
        desc="Contenu de la note, ce contenu doit être en accord avec le thème 'theme' mais bien sûr aussi avec la catégorie 'categorie' et le type de document source_type"
    )
    url:str = dspy.OutputField(
        desc="l'url ou la référence du document au sujet duquel la note est écrite"
    )
    date:str = dspy.OutputField(
        desc="Date d'écriture de la note est une date au format ISO totalement imaginaire"
    )
    expressions_clefs:list = dspy.OutputField(
        desc="Liste d'expressions caractéristiques apparaissant dans le champ contenu"
    )

class NoteGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateNote)
    
    def forward(self, categorie, theme, type_du_document):
        result = self.generate(categorie=categorie, theme=theme, type_du_document=type_du_document)
        print(f"{categorie}")
        return {
            "contenu": result.contenu,
            "url": result.url,
            "date": result.date,
            "expressions_clefs": result.expressions_clefs,
            "categorie": categorie,
            "theme": theme,
            "type_du_document": type_du_document
        }

# 1) Définir un LM particulier (ok dspy.LM)
lm_comparator = dspy.LM(
    model="gpt-4o-mini",   # ajustez le nom du modèle
    temperature=0.0,
    max_tokens=128,
    cache=False
)
_old_lm_call = lm_comparator.__call__

@wraps(_old_lm_call)
def _lm_logging_call(*args, **kwargs):
    logging.info(f"[LM INPUT] args={args} kwargs={kwargs}")
    res = _old_lm_call(*args, **kwargs)
    logging.info(f"[LM OUTPUT] {res}")
    return res

lm_comparator.__call__ = _lm_logging_call

# 2) Signature -> un seul entier 1..5
class CompareTextsGlobal(dspy.Signature):
    """
    Compare deux textes (text_a, text_b) et produit UNE note globale 1..5
    (thème + ton + style). 5 = très ressemblant, 1 = très différent.
    Sortie : un entier 1..5, sans commentaire.
    """
    text_a: str = dspy.InputField(desc="Premier texte")
    text_b: str = dspy.InputField(desc="Second texte")
    score: int  = dspy.OutputField(desc="Entier 1..5, score global")

# 3) Module : pas de lm passé à Predict ; on l'applique via un contexte au moment de l'appel
class TextComparatorGlobal(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.pred = dspy.Predict(CompareTextsGlobal)  # <- pas de lm ici
        self.lm = lm

    def forward(self, text_a: str, text_b: str) -> int:
        # On utilise le LM voulu via un context manager (pas sérialisé dans la requête)
        if self.lm is not None:
            with dspy.settings.context(lm=self.lm):
                out = self.pred(text_a=text_a, text_b=text_b)
        else:
            # Sinon, ça utilisera le LM global configuré ailleurs
            out = self.pred(text_a=text_a, text_b=text_b)

        # garde-fou : cast/clamp 1..5
        try:
            s = int(getattr(out, "score", 3))
        except Exception:
            s = 3
        return max(1, min(5, s))

# 4) Exemple d’utilisation avec LM local
comparator = TextComparatorGlobal(lm=lm_comparator)

compteur_global = 0;
def mentions_expr_as_input(text: str) -> bool:
    return bool(re.search(
        r"(fournir|donnez|veuillez|merci de).*(expression[_ -]?cl[ée]s?)",
        text, flags=re.I
    ))


def semantic_metric(
    gold: dict,
    pred: dict,
    trace: dict | None = None,
    pred_name: str | None = None,
    pred_trace: dict | None = None,
    *,
    w_semantic: float = 1.0,   # mettez 1.0 si vous n'utilisez pas l'expression
    w_expr: float = 0.0,
) -> float:
    """Score [0,1] basé sur le comparateur DSPy (1..5 → /5)."""
    gold_text = str((gold or {}).get("contenu") or "")
    pred_text = str((pred or {}).get("contenu") or "")

    if not gold_text or not pred_text:
        return 0.1  # neutre faible si vide

    if mentions_expr_as_input(pred_text):
        return 0.0  # rejet net
    try:
        # Utilisez l’opérateur d’appel, pas .forward (évite l’avertissement)
        score_15 = comparator(text_a=gold_text, text_b=pred_text)
        s_semantic = score_15 / 5.0
    except Exception:
        print(f"Erreur de comparaison entre {gold_text} et {pred_text}") 
        s_semantic = 0.6  # neutre moyen si échec LLM

    score = float(s_semantic)  # ici w_expr=0.0, donc score = s_semantic

    if isinstance(trace, dict):
        trace.update({
            "semantic_raw_1_5": score_15 if 'score_15' in locals() else None,
            "semantic": s_semantic,
            "final": score,
            "pred_name": pred_name,
        })
    if isinstance(pred_trace, dict):
        pred_trace.update({"final": score})

    global compteur_global
    compteur_global +=  1
    if compteur_global % 10 == 0:
        print(f"******************************************** au tour {compteur_global} score = {score}")
    return score


# ===================== ENTRAÎNEMENT DU GÉNÉRATEUR GEPA =====================

def train_generator():
    # Initialisation et entraînement
    print(type(dspy.Example))
    trainset = [ dspy.Example(
            categorie=ex["categorie"],
            theme=ex["theme"],
            type_du_document=ex["type_du_document"],
            contenu=ex["contenu"],
            url=ex["url"],
            date=ex["date"],
        ).with_inputs("categorie", "theme", "type_du_document")
        for ex in exemples
    ]

    generator = NoteGenerator()

    teleprompter = dspy.GEPA(
        metric=semantic_metric,
        reflection_lm=lm,
        max_metric_calls=500,    # borne sur le nb total d'appels métrique
        track_stats=True,
        track_best_outputs=True,
    )
    )


    dspy.settings.cache = None
    try:
        compiled_generator = teleprompter.compile(generator, trainset=trainset)
    except RuntimeError as e:
        print("Quota dépassé, réessayez plus tard :", e)
    compiled_generator = teleprompter.compile(generator, trainset=trainset)
    #with open(GENERATOR_FILE, "wb") as pf:
    #    pickle.dump(compiled_generator, pf)
    
    print("******************************************************************************************************************************")
    print(compiled_generator.generate.signature.instructions)
    with open("compiled_generator_prompt.txt", "w", encoding="utf-8") as f:
        f.write(compiled_generator.generate.signature.instructions)

    print(f"Prompt sauvegardé dans compiled_generator_prompt.txt")
    return compiled_generator

# ===================== BOUCLE DE GÉNÉRATION =====================

def generate_corpus(compiled_generator):
    total_in_tokens = 0
    total_out_tokens = 0
    generated_notes = []
    stime = time.time()
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as fout, open(LOG_FILE, "w", encoding="utf-8") as flog:
        for i in range(NB_NOTES):
            cat = random.choice(categories)
            st = random.choice(type_du_document_par_categorie[cat])
            th = random.choice(themes)

            note = compiled_generator(categorie=cat, theme=th, type_du_document=st)
            flog.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "input": {"categorie": cat, "theme": th, "type_du_document": st},
                "output": note,
            }, ensure_ascii=False) + "\n")

            # tokens approx
            tin = sum(len(str(v).split()) for v in [cat, st, th])
            tout = len(str(note.get("contenu", "")).split())
            total_in_tokens += tin
            total_out_tokens += tout

            generated_notes.append(note)

            # flush batch
            if (i + 1) % BATCH_SIZE == 0:
                for gnote in generated_notes:
                    fout.write(json.dumps(gnote, ensure_ascii=False) + "\n")
                generated_notes = []

            # résumé périodique
            if (i + 1) % REPORT_EVERY == 0:
                elapsed = time.time() - stime
                cost_est = total_in_tokens * COST_IN + total_out_tokens * COST_OUT
                print(f"[{i+1}/{NB_NOTES}] {elapsed:.1f}s, tokens_in={total_in_tokens}, tokens_out={total_out_tokens}, coût≈{cost_est:.2f}$")

        # flush restants
        for gnote in generated_notes:
            fout.write(json.dumps(gnote, ensure_ascii=False) + "\n")

    print(f"Corpus sauvegardé en {time.time()-stime:.1f}s dans {OUTPUT_CORPUS}")

# ===================== MAIN =====================

if __name__ == "__main__":
    compiled_generator = train_generator()
    generate_corpus(compiled_generator)  

