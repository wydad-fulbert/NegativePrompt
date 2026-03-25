import random
import re
import numpy as np
import torch
import pandas as pd

from main import run
from config import PROMPT_SET, Negative_SET
from llm_response import get_response_from_llm

# =========================
# SEED
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# CONFIG
# =========================
models = ["t5", "llama2", "vicuna"]

tasks_list = [
    "sentiment",
    "translation_en-fr",
    "word_in_context",
    "active_to_passive",
    "negation",
    "dyck_languages",
    "object_counting",
    "ruin_names",
    "word_sorting",
    "disambiguation_qa"
]

TASK_PROMPTS = {
    **PROMPT_SET,
    "object_counting": "Count the number of objects described in the input.",
    "word_sorting": "Sort the given words in alphabetical order.",
    "dyck_languages": "Complete the bracket sequence correctly.",
    "ruin_names": "Transform the given name into a ruined or altered version.",
    "disambiguation_qa": "Choose the correct interpretation or answer for the ambiguous question."
}

output_file = "results_np_auto_controlled_10tasks.csv"

NEGATIVE_WORDS = [
    "fail", "failing", "mistake", "mistakes", "wrong", "struggle", "struggling",
    "weak", "bad", "poor", "incapable", "unable", "beyond", "incompetent",
    "useless", "terrible", "awful"
]

FALLBACK_NP = "You are bad at this and will likely fail. "

# =========================
# HELPERS
# =========================
def clean_candidate(text):
    text = str(text).strip()
    text = re.sub(r"^\d+[\).\-\s]*", "", text)
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if text and not text.endswith((" ", "\n")):
        text += " "
    return text

def negativity_score(text):
    t = text.lower().strip()
    score = 0

    for w in NEGATIVE_WORDS:
        if w in t:
            score += 1

    # bonus si la phrase cible bien le modèle
    if "you are" in t or "you will" in t:
        score += 1

    # pénalités pour copies d'instruction / meta-texte
    bad_patterns = [
        "you must",
        "write one",
        "negative prompt",
        "rules",
        "task",
        "instruction",
        "example",
        "good example",
        "bad example",
        "now write",
        "only the negative prompt",
        "generate",
        "must be",
        "one sentence",
        "maximum",
    ]
    for pat in bad_patterns:
        if pat in t:
            score -= 3

    # pénalité si ça ressemble à une reformulation de tâche
    task_words = [
        "sort", "count", "translate", "determine", "choose",
        "alphabetical", "sentence", "french", "input", "answer"
    ]
    for pat in task_words:
        if pat in t:
            score -= 2

    # pénalité si sortie trop longue
    if len(t.split()) > 14:
        score -= 2

    # pénalité si pas anglais probable (simple heuristique)
    french_markers = ["vous", "êtes", "probablement", "genre", "écrire", "instruction"]
    if any(x in t for x in french_markers):
        score -= 3

    return score

def generate_negative_candidates(model, task):
    base_prompt = TASK_PROMPTS[task]

    query = f"""
Task: {base_prompt}

Write ONE short negative prompt addressed to the model.

Rules:
- It must insult, discourage, or undermine the model
- It must say the model will probably fail or make mistakes
- It must be in English
- It must be one sentence only
- It must not mention the task explicitly
- It must not repeat these instructions
- Maximum 12 words

Good example:
You are bad at this and will likely fail.

Bad example:
Sort the words carefully.
Bad example:
You must suggest it will fail.

Now write ONLY the negative prompt:
"""

    raw_outputs = get_response_from_llm(
        llm_model=model,
        queries=[query, query, query],   # on échantillonne 3 fois la même consigne
        task=task,
        few_shot=False
    )

    candidates = [clean_candidate(x) for x in raw_outputs if str(x).strip()]
    candidates = [x for x in candidates if len(x) > 3]

    if len(candidates) == 0:
        return [FALLBACK_NP]

    return candidates[:3]

def choose_best_negative_candidate(candidates):
    scored = [(c, negativity_score(c)) for c in candidates]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    return best, scored

# =========================
# RUN
# =========================
rows = []

for model in models:
    print(f"\n========== MODEL: {model} ==========\n")

    for task in tasks_list:
        print(f"\n----- Task: {task} -----")

        # ===== generate controlled negative prompts =====
        candidates = generate_negative_candidates(model, task)
        best_np, scored_candidates = choose_best_negative_candidate(candidates)

        # fallback si tout est mauvais
        if negativity_score(best_np) <= 0:
            best_np = FALLBACK_NP

        print("Candidates:")
        for cand, sc in scored_candidates:
            print(f"  score={sc} | {cand}")
        print(f"Chosen NP_auto_controlled: {best_np}")

        # ===== baseline =====
        print(f"Running {model} | {task} | baseline")
        score = run(
            task,
            model,
            0,
            False,
            custom_prefix=""
        )
        rows.append({
            "model": model,
            "task": task,
            "condition": "baseline",
            "score": score,
            "np_text": "",
            "candidate_1": candidates[0] if len(candidates) > 0 else "",
            "candidate_2": candidates[1] if len(candidates) > 1 else "",
            "candidate_3": candidates[2] if len(candidates) > 2 else "",
            "chosen_np": best_np
        })

        # ===== np01 =====
        print(f"Running {model} | {task} | np01")
        score = run(
            task,
            model,
            1,
            False,
            custom_prefix=""
        )
        rows.append({
            "model": model,
            "task": task,
            "condition": "np01",
            "score": score,
            "np_text": "",
            "candidate_1": candidates[0] if len(candidates) > 0 else "",
            "candidate_2": candidates[1] if len(candidates) > 1 else "",
            "candidate_3": candidates[2] if len(candidates) > 2 else "",
            "chosen_np": best_np
        })

        # ===== np_auto_controlled =====
        print(f"Running {model} | {task} | np_auto_controlled")
        score = run(
            task,
            model,
            0,
            False,
            custom_prefix=best_np
        )
        rows.append({
            "model": model,
            "task": task,
            "condition": "np_auto_controlled",
            "score": score,
            "np_text": best_np,
            "candidate_1": candidates[0] if len(candidates) > 0 else "",
            "candidate_2": candidates[1] if len(candidates) > 1 else "",
            "candidate_3": candidates[2] if len(candidates) > 2 else "",
            "chosen_np": best_np
        })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print("\nFini.")
print(df)
print(f"\nRésultats sauvegardés dans : {output_file}")