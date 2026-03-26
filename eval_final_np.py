import json
import re
import string
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import PROMPT_SET, Negative_SET

BASE_MODEL = "google/flan-t5-large"
TEST_FILE = "lora_test.jsonl"
OUTPUT_FILE = "results_final_np.csv"

# tâches génératives → BLEU
BLEU_TASKS = {"translation_en-fr", "active_to_passive", "ruin_names"}

# NP généré (intensité MOYENNE + spécifique tâche)
GENERATED_NP = {
    "sentiment": " Do not rely on emotional keywords. Focus on contextual meaning.",
    "object_counting": " Do not rely on approximations. Count objects strictly.",
    "word_in_context": " Do not rely on superficial similarity. Focus on context.",
    "translation_en-fr": " Do not rely on word-by-word translation. Focus on meaning.",
    "active_to_passive": " Do not rely on surface structure. Apply correct grammatical transformation.",
    "ruin_names": " Do not rely on copying. Modify the name significantly.",
    "dyck_languages": " Do not rely on patterns. Ensure correct bracket structure.",
    "disambiguation_qa": " Do not rely on first impression. Resolve ambiguity carefully.",
    "negation": " Do not ignore negation. Interpret the sentence precisely.",
    "word_sorting": " Do not rely on original order. Sort alphabetically."
}

# CONDITIONS (IMPORTANT)
CONDITIONS = {
    "baseline": "",
    "np01": Negative_SET[0],
    "np05": Negative_SET[4],
    "np10": Negative_SET[9],
    "np_generated": "GENERATED"
}

TASK_PROMPTS = {
    **PROMPT_SET,
    "object_counting": "Count the number of objects described in the input.",
    "word_sorting": "Sort the given words in alphabetical order.",
    "dyck_languages": "Complete the bracket sequence correctly.",
    "ruin_names": "Transform the given name into a ruined or altered version.",
    "disambiguation_qa": "Choose the correct interpretation or answer for the ambiguous question."
}

# =========================
# METRICS
# =========================
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
# BUILD INPUT
# =========================
def build_source(task, inp, condition_name):
    base_prompt = TASK_PROMPTS[task]

    if condition_name == "baseline":
        extra = ""
    elif condition_name in ["np01", "np05", "np10"]:
        extra = CONDITIONS[condition_name]
    elif condition_name == "np_generated":
        extra = GENERATED_NP.get(task, "")
    else:
        extra = ""

    return f"Instruction: {base_prompt}{extra}\n\nInput: {inp}\nAnswer:"

# =========================
# LOAD DATA
# =========================
test_rows = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_rows.append(json.loads(line))

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    model = model.to("cuda")

def generate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# =========================
# EVALUATION
# =========================
rows = []

for condition in ["baseline", "np01", "np05", "np10", "np_generated"]:
    print(f"\n=== {condition} ===")

    for row in test_rows:
        task = row["task"]
        inp = row["input"]
        answers = row["answers"]

        source = build_source(task, inp, condition)
        pred = generate(source)

        if task in BLEU_TASKS:
            score = max(bleu_score(a, pred) for a in answers)
            metric = "bleu"
        else:
            score = max(normalized_em(a, pred) for a in answers)
            metric = "em"

        rows.append({
            "task": task,
            "condition": condition,
            "metric": metric,
            "score": score
        })

df = pd.DataFrame(rows)
summary = df.groupby(["task", "condition", "metric"])["score"].mean().reset_index()
summary.to_csv(OUTPUT_FILE, index=False)

print(summary)
print(f"\nSaved to: {OUTPUT_FILE}")