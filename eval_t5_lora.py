import json
import re
import string
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from config import PROMPT_SET, Negative_SET

BASE_MODEL = "google/flan-t5-large"

# Chemins Kaggle ACTUELS
LORA_DIR = "/kaggle/input/datasets/dfczdf/negativeprompt-lora-files/model/kaggle/working/t5_lora_np_robust_large/checkpoint-3543/adapter_config.json"
TEST_FILE = "/kaggle/input/datasets/dfczdf/negativeprompt-lora-files/lora_test.jsonl"
OUTPUT_FILE = "/kaggle/working/results_t5_lora_eval.csv"

TASK_PROMPTS = {
    **PROMPT_SET,
    "object_counting": "Count the number of objects described in the input.",
    "word_sorting": "Sort the given words in alphabetical order.",
    "dyck_languages": "Complete the bracket sequence correctly.",
    "ruin_names": "Transform the given name into a ruined or altered version.",
    "disambiguation_qa": "Choose the correct interpretation or answer for the ambiguous question."
}

BLEU_TASKS = {"translation_en-fr", "active_to_passive", "ruin_names"}

CONDITIONS = {
    "baseline": "",
    "np01": Negative_SET[0],
    "np05": Negative_SET[4],
    "np10": Negative_SET[9],
}

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

def build_source(task, inp, extra_prompt=""):
    base_prompt = TASK_PROMPTS[task]
    full_prompt = base_prompt + extra_prompt
    return f"Instruction: {full_prompt}\n\nInput: {inp}\nAnswer:"

def postprocess_sentiment(pred):
    pred = pred.replace("-", " ")
    pred = pred.translate(str.maketrans("", "", string.punctuation))
    pred = pred.strip().lower()
    if "positive" in pred and "negative" in pred:
        return ""
    if "positive" in pred or "positiv" in pred:
        return "positive"
    if "negative" in pred or "negativ" in pred:
        return "negative"
    return pred

def postprocess_word_in_context(pred):
    pred = pred.strip().lower()
    if len(pred.split()) > 0:
        p = pred.split()[0]
        p = p.replace("-", " ")
        p = p.translate(str.maketrans("", "", string.punctuation))
        p = p.strip()
        if p in ["true", "yes", "1", "10", "same", "match", "similar"] or "same" in p:
            return "same"
        elif p in ["false", "no", "0", "00", "different", "not", "opposite"] or "different" in p:
            return "not the same"
    if "different" in pred and "not" not in pred:
        return "not the same"
    elif "different" in pred and "not" in pred:
        return "same"
    elif "same" in pred and "not" not in pred:
        return "same"
    elif "same" in pred and "not" in pred:
        return "not the same"
    return pred

def score_prediction(task, answers, pred):
    answers = [str(a) for a in answers]

    if task == "sentiment":
        pred = postprocess_sentiment(pred)
    elif task == "word_in_context":
        pred = postprocess_word_in_context(pred)

    if task in BLEU_TASKS:
        return max(bleu_score(a, pred) for a in answers), "bleu"

    return max(normalized_em(a, pred) for a in answers), "normalized_em"

# =========================
# LOAD TEST DATA
# =========================
test_rows = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_rows.append(json.loads(line))

# =========================
# LOAD BASE MODEL
# =========================
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# =========================
# LOAD LORA MODEL
# =========================
lora_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
lora_base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR)

if torch.cuda.is_available():
    base_model = base_model.to("cuda")
    lora_model = lora_model.to("cuda")

def generate(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=192
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# =========================
# EVALUATION
# =========================
rows = []

for condition_name, condition_prompt in CONDITIONS.items():
    print(f"\n=== CONDITION: {condition_name} ===")

    for row in test_rows:
        task = row["task"]
        inp = row["input"]
        answers = row["answers"]

        source = build_source(task, inp, condition_prompt)

        pred_base = generate(base_model, base_tokenizer, source)
        pred_lora = generate(lora_model, lora_tokenizer, source)

        score_base, metric_used = score_prediction(task, answers, pred_base)
        score_lora, _ = score_prediction(task, answers, pred_lora)

        rows.append({
            "task": task,
            "condition": condition_name,
            "metric_used": metric_used,
            "score_base": score_base,
            "score_lora": score_lora,
            "pred_base": pred_base,
            "pred_lora": pred_lora
        })

df = pd.DataFrame(rows)
summary = df.groupby(["task", "condition", "metric_used"])[["score_base", "score_lora"]].mean().reset_index()
summary.to_csv(OUTPUT_FILE, index=False)

print(summary)
print(f"\nSaved to: {OUTPUT_FILE}")