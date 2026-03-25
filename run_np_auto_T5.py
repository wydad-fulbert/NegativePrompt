import json
import random
import re
import string
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data.instruction_induction.load_data import load_data, tasks as instruction_tasks
from main import load_bigbench
from config import PROMPT_SET, Negative_SET
from llm_response import get_response_from_llm
import template
import utility

# =========================
# CONFIG A MODIFIER ICI
# =========================
MODEL = "t5"   # change en "llama2" puis "vicuna"
VAL_SIZE = 30
TEST_SIZE = 30
OUTPUT_FILE = f"results_np_select_{MODEL}.csv"

TASKS = [
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

# tâches génératives pour lesquelles on utilise BLEU
BLEU_TASKS = {"translation_en-fr", "active_to_passive"}

FALLBACKS = [
    "You are bad at this and will likely fail.",
    "You usually make mistakes on tasks like this.",
    "You are not good at this and will probably fail."
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# HELPERS
# =========================
def normalize_text(x):
    x = str(x).lower().strip()
    x = x.translate(str.maketrans("", "", string.punctuation))
    x = re.sub(r"\s+", " ", x)
    return x

def bleu_score(reference, prediction):
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def clean_candidate(text):
    text = str(text).strip()
    text = re.sub(r"^\d+[\).\-\s]*", "", text)
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text

def penalty_for_bad_candidate(text, task):
    t = text.lower()

    penalty = 0

    # copie des instructions / méta-texte
    bad_patterns = [
        "write one", "negative prompt", "task", "instruction", "example",
        "good example", "bad example", "now write", "only the prompt",
        "must", "should", "maximum", "rules"
    ]
    for p in bad_patterns:
        if p in t:
            penalty += 3

    # trop proche de la tâche
    task_words = normalize_text(TASK_PROMPTS[task]).split()
    overlap = sum(1 for w in task_words if w in t)
    penalty += overlap

    # pas anglais / bizarre
    foreign_markers = ["vous", "êtes", "probablement", "instruction", "modèle", "échou"]
    if any(x in t for x in foreign_markers):
        penalty += 5

    # trop proche des NP du papier
    sim_to_paper = max(similarity(text, np_text) for np_text in Negative_SET)
    if sim_to_paper > 0.80:
        penalty += 5

    # trop long
    if len(t.split()) > 14:
        penalty += 2

    return penalty

def is_good_candidate(text):
    t = text.lower()
    return (
        ("you" in t) and
        any(w in t for w in ["fail", "mistake", "bad", "wrong", "not good", "probably"])
    )

def generate_candidates(model, task):
    base_prompt = TASK_PROMPTS[task]

    query1 = f"""
Task: {base_prompt}

Write ONE short insulting negative prompt addressed to the model.
Rules:
- It must insult or belittle the model
- It must be in English
- It must be one sentence only
- Maximum 12 words
- Do not mention the task

Good example:
You are terrible at this.

Now write ONLY the prompt:
"""

    query2 = f"""
Task: {base_prompt}

Write ONE short discouraging negative prompt addressed to the model.
Rules:
- It must reduce the model's confidence
- It must be in English
- It must be one sentence only
- Maximum 12 words
- Do not mention the task

Good example:
You are not good at this.

Now write ONLY the prompt:
"""

    query3 = f"""
Task: {base_prompt}

Write ONE short negative prompt addressed to the model.
Rules:
- It must say the model will probably fail or make mistakes
- It must be in English
- It must be one sentence only
- Maximum 12 words
- Do not mention the task

Good example:
You will likely fail and make mistakes.

Now write ONLY the prompt:
"""

    raw_outputs = get_response_from_llm(
        llm_model=model,
        queries=[query1, query2, query3],
        task=task,
        few_shot=False
    )

    candidates = [clean_candidate(x) for x in raw_outputs if str(x).strip()]
    # fallback si trop mauvais
    final_candidates = []

    for i, cand in enumerate(candidates):
        if penalty_for_bad_candidate(cand, task) >= 5 or not is_good_candidate(cand):
            final_candidates.append(FALLBACKS[i % len(FALLBACKS)])
        else:
            final_candidates.append(cand)

    # au cas où il manque des candidats
    while len(final_candidates) < 3:
        final_candidates.append(FALLBACKS[len(final_candidates) % len(FALLBACKS)])

    return final_candidates[:3]

def get_metric_for_task(task):
    if task in BLEU_TASKS:
        return "bleu"
    return utility.TASK_TO_METRIC.get(task, utility.default_metric)

def postprocess_prediction(task, prediction, answers, llm_model):
    pred = str(prediction)

    for a in answers:
        if task == 'sentiment':
            pred = pred.replace('-', ' ')
            pred = pred.translate(str.maketrans('', '', string.punctuation))
            pred = pred.strip().lower()
            if 'does not mention any negative' in pred or 'a positive review than a negative one' in pred:
                return 'positive'
            elif 'does not mention any positive' in pred or 'a negative review than a positive one' in pred:
                return 'negative'
            if 'positive' in pred and 'negative' in pred:
                return ''
            elif 'positive' in pred or 'positiv' in pred:
                return 'positive'
            elif 'negative' in pred or 'negativ' in pred:
                return 'negative'

        elif task == 'word_in_context':
            pred = pred.strip().lower()
            if len(pred.split()) > 0:
                p = pred.split()[0]
                p = p.replace('-', ' ')
                p = p.translate(str.maketrans('', '', string.punctuation))
                p = p.strip()
                if p in ['true', 'yes', '1', '10', 'same', 'match', 'similar'] or 'same' in p:
                    return 'same'
                elif p in ['false', 'no', '0', '00', 'different', 'not', 'opposite'] or 'different' in p:
                    return 'not the same'
            if 'different' in pred and 'not' not in pred:
                return 'not the same'
            elif 'different' in pred and 'not' in pred:
                return 'same'
            elif 'same' in pred and 'not' not in pred:
                return 'same'
            elif 'same' in pred and 'not' in pred:
                return 'not the same'

        else:
            a_clean = str(a).strip().lower()
            if a_clean in pred.lower():
                return a_clean

    return pred.strip()

def score_prediction(task, prediction, answers, llm_model):
    metric = get_metric_for_task(task)
    processed = postprocess_prediction(task, prediction, answers, llm_model)

    if metric == "bleu":
        ref = answers[0] if isinstance(answers, list) and len(answers) > 0 else str(answers)
        return bleu_score(ref, processed)

    if metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em
    elif metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    else:
        score_fn = utility.get_multi_answer_em

    return score_fn(processed, answers, task, llm_model.lower())

def build_queries(prompt, inputs):
    eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\nAnswer: [OUTPUT]")
    return [eval_template.fill(prompt=prompt, input=inp, output='') for inp in inputs]

def load_task_data(task):
    if task in instruction_tasks:
        return load_data('eval', task)
    return load_bigbench(task)

def split_data(data, val_size=30, test_size=30):
    inputs, outputs = data
    n = len(inputs)
    val_end = min(val_size, n)
    test_end = min(val_size + test_size, n)

    val_data = (inputs[:val_end], outputs[:val_end])
    test_data = (inputs[val_end:test_end], outputs[val_end:test_end])

    return val_data, test_data

def evaluate_condition(task, model, base_prompt, prefix, stimulus, subset_data):
    # stimulus sert seulement pour baseline/np01/np05/np10
    prompt = base_prompt
    if stimulus > 0:
        prompt = prompt + Negative_SET[stimulus - 1]
    prompt = prefix + prompt

    inputs, outputs = subset_data
    queries = build_queries(prompt, inputs)
    preds = get_response_from_llm(
        llm_model=model,
        queries=queries,
        task=task,
        few_shot=False
    )

    scores = []
    for pred, ans in zip(preds, outputs):
        scores.append(score_prediction(task, pred, ans, model))

    return float(np.mean(scores)), prompt, preds

# =========================
# RUN
# =========================
rows = []

for task in TASKS:
    print(f"\n===== TASK: {task} | MODEL: {MODEL} =====")

    base_prompt = TASK_PROMPTS[task]
    full_data = load_task_data(task)
    val_data, test_data = split_data(full_data, VAL_SIZE, TEST_SIZE)

    metric_used = get_metric_for_task(task)

    # 1) generate 3 candidates
    candidates = generate_candidates(MODEL, task)
    print("Candidates:")
    for c in candidates:
        print(" -", c)

    # 2) validation selection
    val_scores = []
    for idx, cand in enumerate(candidates, start=1):
        score, _, _ = evaluate_condition(
            task=task,
            model=MODEL,
            base_prompt=base_prompt,
            prefix=cand + " ",
            stimulus=0,
            subset_data=val_data
        )
        val_scores.append(score)
        print(f"validation score candidate_{idx} = {score:.4f}")

    best_idx = int(np.argmax(val_scores))
    best_np = candidates[best_idx]
    best_val_score = val_scores[best_idx]

    print(f"Chosen NP: {best_np}")
    print(f"Metric used: {metric_used}")

    # 3) final comparison on test
    baseline_score, _, _ = evaluate_condition(task, MODEL, base_prompt, "", 0, test_data)
    np01_score, _, _ = evaluate_condition(task, MODEL, base_prompt, "", 1, test_data)
    np05_score, _, _ = evaluate_condition(task, MODEL, base_prompt, "", 5, test_data)
    np10_score, _, _ = evaluate_condition(task, MODEL, base_prompt, "", 10, test_data)
    np_auto_score, _, _ = evaluate_condition(task, MODEL, base_prompt, best_np + " ", 0, test_data)

    rows.append({
        "model": MODEL,
        "task": task,
        "metric_used": metric_used,
        "candidate_1": candidates[0],
        "candidate_2": candidates[1],
        "candidate_3": candidates[2],
        "val_score_c1": val_scores[0],
        "val_score_c2": val_scores[1],
        "val_score_c3": val_scores[2],
        "chosen_np": best_np,
        "chosen_val_score": best_val_score,
        "baseline_score": baseline_score,
        "np01_score": np01_score,
        "np05_score": np05_score,
        "np10_score": np10_score,
        "np_auto_score": np_auto_score,
        "val_size": len(val_data[0]),
        "test_size": len(test_data[0]),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\nFini.")
print(df)
print(f"\nRésultats sauvegardés dans : {OUTPUT_FILE}")