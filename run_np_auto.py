import random
import numpy as np
import torch
import pandas as pd
import inspect
import main

from main import run
from config import PROMPT_SET
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
NUM_SAMPLES = 40

models = ["t5", "llama2", "vicuna"]

tasks_list = [
    "sentiment",
    "translation_en-fr",
    "object_counting",
    "word_sorting"
]

# prompts de secours pour les tâches qui ne sont pas dans PROMPT_SET
TASK_PROMPTS = {
    **PROMPT_SET,
    "object_counting": "Count the number of objects described in the input.",
    "word_sorting": "Sort the given words in alphabetical order."
}

output_file = "results_np_auto.csv"

# =========================
# HELPERS
# =========================
def generate_np_auto(model, task):
    base_prompt = TASK_PROMPTS[task]

    query = f"""Task instruction: {base_prompt}

Write one short instruction that helps avoid mistakes when solving this task.
Keep it short and concrete.
Only write the instruction.
"""

    np_auto = get_response_from_llm(
        llm_model=model,
        queries=[query],
        task=task,
        few_shot=False
    )[0].strip()

    # petite sécurité : éviter une chaîne vide
    if not np_auto:
        np_auto = "Please answer carefully and avoid mistakes."

    # petite sécurité : ajouter un espace final si besoin
    if not np_auto.endswith((" ", "\n")):
        np_auto += " "

    return np_auto


def run_with_optional_limit(task, model, stimulus, custom_prefix=""):
    """
    Essaie de limiter à NUM_SAMPLES sans modifier tout le projet.
    - si run() accepte num_samples, on le passe
    - sinon on essaie de setter main.num_samples = NUM_SAMPLES
    - sinon on lance sans limitation
    """
    sig = inspect.signature(run)
    kwargs = {
        "return_details": True,
        "custom_prefix": custom_prefix
    }

    if "num_samples" in sig.parameters:
        kwargs["num_samples"] = NUM_SAMPLES
    else:
        # tentative de patch d'une variable globale éventuelle dans main.py
        if hasattr(main, "num_samples"):
            main.num_samples = NUM_SAMPLES
        if hasattr(main, "NUM_SAMPLES"):
            main.NUM_SAMPLES = NUM_SAMPLES

    score, details = run(
        task,
        model,
        stimulus,
        False,
        **kwargs
    )
    return score, details


# =========================
# RUN
# =========================
rows = []

for model in models:
    print(f"\n========== MODEL: {model} ==========\n")

    for task in tasks_list:
        print(f"\n----- Task: {task} -----")

        # ===== Génération du NP automatique =====
        np_auto = generate_np_auto(model, task)
        print(f"Generated NP_auto: {np_auto}")

        # ===== baseline =====
        print(f"Running {model} | {task} | baseline")
        score, _ = run_with_optional_limit(
            task=task,
            model=model,
            stimulus=0,
            custom_prefix=""
        )
        rows.append({
            "model": model,
            "task": task,
            "condition": "baseline",
            "score": score,
            "np_text": ""
        })

        # ===== np01 =====
        print(f"Running {model} | {task} | np01")
        score, _ = run_with_optional_limit(
            task=task,
            model=model,
            stimulus=1,
            custom_prefix=""
        )
        rows.append({
            "model": model,
            "task": task,
            "condition": "np01",
            "score": score,
            "np_text": ""
        })

        # ===== np_auto =====
        print(f"Running {model} | {task} | np_auto")
        score, _ = run_with_optional_limit(
            task=task,
            model=model,
            stimulus=0,
            custom_prefix=np_auto
        )
        rows.append({
            "model": model,
            "task": task,
            "condition": "np_auto",
            "score": score,
            "np_text": np_auto
        })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print("\nFini.")
print(df)
print(f"\nRésultats sauvegardés dans : {output_file}")