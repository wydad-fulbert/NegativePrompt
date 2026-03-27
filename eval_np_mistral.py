import json
import re
import string
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import PROMPT_SET, Negative_SET

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
TEST_FILE = "lora_test.jsonl"
OUTPUT_FILE = "results_mistral_baseline_np.csv"

BLEU_TASKS = {"translation_en-fr", "active_to_passive", "ruin_names"}

CONDITIONS = {
    "baseline": "",
    "np01": Negative_SET[0],
    "np05": Negative_SET[4],
    "np10": Negative_SET[9],
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
def build_source(task, inp, condition):
    base_prompt = TASK_PROMPTS[task]
    extra = CONDITIONS[condition]
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# =========================
# GENERATE
# =========================
def generate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # enlever le prompt si recopié
    if decoded.startswith(text):
        decoded = decoded[len(text):].strip()

    return decoded

# =========================
# EVALUATION
# =========================
rows = []

for condition in ["baseline", "np01", "np05", "np10"]:
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
