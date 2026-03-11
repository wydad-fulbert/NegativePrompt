import random
import numpy as np
import pandas as pd
import torch
import re
import string

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data.instruction_induction.load_data import load_data
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
    "translation_en-fr",
    "active_to_passive",
    "ruin_names",
]

stimuli = [0, 1, 5, 10]

num_samples = 100
output_file = "results_bleu_3models_detailed.csv"


# =========================
# HELPERS
# =========================
def get_prompt_with_stimulus(base_prompt, stimulus):
    if stimulus == 0:
        return base_prompt
    return base_prompt + Negative_SET[stimulus - 1]


def normalize_text(x):
    x = str(x).lower().strip()
    x = x.translate(str.maketrans("", "", string.punctuation))
    x = re.sub(r"\s+", " ", x)
    return x


def normalized_em(reference, prediction):
    return int(normalize_text(reference) == normalize_text(prediction))


def bleu_score(reference, prediction):
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()

    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)


# =========================
# RUN
# =========================
rows = []

for model in models:
    print(f"\n========== MODEL: {model} ==========\n")

    for task in tasks_list:
        print(f"\n===== TASK: {task} =====")

        test_data = load_data("eval", task)
        inputs = test_data[0][:num_samples]
        outputs = test_data[1][:num_samples]

        references = []
        for out in outputs:
            if isinstance(out, list):
                references.append(out[0])
            else:
                references.append(out)

        base_prompt = PROMPT_SET.get(task, "Solve the task carefully.")

        for stimulus in stimuli:
            print(f"Model={model} | Task={task} | Stimulus={stimulus}")

            prompt = get_prompt_with_stimulus(base_prompt, stimulus)

            queries = []
            for inp in inputs:
                q = f"Instruction: {prompt}\n\nInput: {inp}\nAnswer:"
                queries.append(q)

            predictions = get_response_from_llm(
                llm_model=model,
                queries=queries,
                task=task,
                few_shot=False
            )

            for inp, ref, pred in zip(inputs, references, predictions):
                rows.append({
                    "model": model,
                    "task": task,
                    "stimulus": stimulus,
                    "input": inp,
                    "reference": ref,
                    "prediction": pred,
                    "em_normalized": normalized_em(ref, pred),
                    "bleu": bleu_score(ref, pred),
                    "prompt": prompt
                })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print("\nFini.")
print(f"Fichier sauvegardé : {output_file}")
print(df.head())
