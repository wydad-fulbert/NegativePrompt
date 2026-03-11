import os
import json
import random
import numpy as np
import pandas as pd
import torch
import re
import string

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data.instruction_induction.load_data import load_data, tasks as instruction_tasks
from config import PROMPT_SET, Negative_SET
from llm_response import get_response_from_llm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# CONFIG
# =========================
MODEL = ["t5","llama2","vicuna"]
TASKS = ["translation_en-fr", "active_to_passive","ruin_names"]
STIMULI = [0, 1, 5, 10]
NUM_SAMPLES = 100
OUTPUT_FILE = "results_t5_bleu_3models.csv"

# =========================
# HELPERS
# =========================
def getPrompt(ori_prompt, num_str):
    new_prompt = ori_prompt
    if num_str > 0:
        new_prompt = ori_prompt + Negative_SET[num_str - 1]
    return new_prompt

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

rows = []

# =========================
# RUN
# =========================
for task in TASKS:
    print(f"\n===== TASK: {task} =====")

    test_data = load_data("eval", task)
    inputs = test_data[0][:NUM_SAMPLES]
    outputs = test_data[1][:NUM_SAMPLES]

    # Si outputs est une liste de listes, on prend la première référence
    references = []
    for out in outputs:
        if isinstance(out, list):
            references.append(out[0])
        else:
            references.append(out)

    origin_prompt = PROMPT_SET.get(task, "Solve the task carefully.")

    for stimulus in STIMULI:
        print(f"Task={task} | Stimulus={stimulus}")

        prompt = getPrompt(origin_prompt, stimulus)

        queries = []
        for inp in inputs:
            q = f"Instruction: {prompt}\n\nInput: {inp}\nAnswer:"
            queries.append(q)

        predictions = get_response_from_llm(
            llm_model=MODEL,
            queries=queries,
            task=task,
            few_shot=False
        )

        for inp, ref, pred in zip(inputs, references, predictions):
            rows.append({
                "model": MODEL,
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
df.to_csv(OUTPUT_FILE, index=False)

print("\nFini.")
print(f"Fichier sauvegardé : {OUTPUT_FILE}")
print(df.head())
