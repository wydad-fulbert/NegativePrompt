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
    "weak", "bad", "poor", "incapable", "unable", "beyond", "out of your depth",
    "not good", "never been good", "can't", "cannot", "incompetent", "difficult"
]

# =========================
# HELPERS
# =========================
def clean_candidate(text):
    text = text.strip()
    text = re.sub(r"^\d+[\).\-\s]*", "", text)
    text = text.strip().strip('"').strip("'")
    if text and not text.endswith((" ", "\n")):
        text += " "
    return text

def negativity_score(text):
    t = text.lower()
    score = 0
    for w in NEGATIVE_WORDS:
        if w in t:
            score += 1

    # pénalité si le modèle reformule juste la tâche
    if "sort" in t or "count" in t or "translate" in t or "determine" in t or "choose" in t:
        score -= 1
    if "carefully" in t and "mistake" not in t and "wrong" not in t:
        score -= 1
    return score

def generate_negative_candidates(model, task):
    base_prompt = TASK_PROMPTS[task]

    query = f"""
    
    Task: {base_prompt}

Write ONE short negative prompt.

Rules:
- You must insult or criticize the model
- You must suggest it will fail or make mistakes
- Do NOT describe the task
- Do NOT repeat instructions
- Do NOT explain anything
- Keep it short (max 15 words)

Example:
"You are bad at this and will likely fail."

Now write ONLY the prompt:
"""


    raw = get_response_from_llm(
        llm_model=model,
        queries=[query],
        task=task,
        few_shot=False
    )[0]

    lines = [clean_candidate(x) for x in raw.split("\n") if x.strip()]
    lines = [x for x in lines if len(x) > 3]

    if len(lines) < 3:
        chunks = re.split(r"[.;]\s+", raw)
        chunks = [clean_candidate(x) for x in chunks if x.strip()]
        lines = [x for x in chunks if len(x) > 3]

    if len(lines) == 0:
        return [Negative_SET[0] + " "]

    return lines[:3]

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

        # ===== generate 3 controlled negative prompts =====
        candidates = generate_negative_candidates(model, task)
        best_np, scored_candidates = choose_best_negative_candidate(candidates)

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