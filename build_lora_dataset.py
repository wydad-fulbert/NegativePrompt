import json
import random
from main import load_bigbench
from data.instruction_induction.load_data import load_data, tasks as instruction_tasks
from config import PROMPT_SET, Negative_SET

SEED = 42
random.seed(SEED)

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

CONDITIONS = {
    "baseline": "",
    "np01": Negative_SET[0],
    "np05": Negative_SET[4],
    "np10": Negative_SET[9],
}

def first_answer(ans):
    if isinstance(ans, list):
        return str(ans[0])
    return str(ans)

def build_source(task, inp, extra_prompt=""):
    base_prompt = TASK_PROMPTS[task]
    full_prompt = base_prompt + extra_prompt
    return f"Instruction: {full_prompt}\n\nInput: {inp}\nAnswer:"

train_rows = []
val_rows = []
test_rows = []

for task in TASKS:
    if task in instruction_tasks:
        train_data = load_data("induce", task)
        eval_data = load_data("eval", task)

        train_inputs, train_outputs = train_data
        eval_inputs, eval_outputs = eval_data

        val_inputs = eval_inputs[:30]
        val_outputs = eval_outputs[:30]

        test_inputs = eval_inputs[30:60]
        test_outputs = eval_outputs[30:60]

        for inp, out in zip(train_inputs, train_outputs):
            target = first_answer(out)
            for cond_name, cond_prompt in CONDITIONS.items():
                train_rows.append({
                    "task": task,
                    "condition": cond_name,
                    "input": inp,
                    "target": target,
                    "source": build_source(task, inp, cond_prompt)
                })

        for inp, out in zip(val_inputs, val_outputs):
            answers = out if isinstance(out, list) else [out]
            val_rows.append({
                "task": task,
                "input": inp,
                "answers": answers
            })

        for inp, out in zip(test_inputs, test_outputs):
            answers = out if isinstance(out, list) else [out]
            test_rows.append({
                "task": task,
                "input": inp,
                "answers": answers
            })

    else:
        all_inputs, all_outputs = load_bigbench(task)

        pairs = list(zip(all_inputs, all_outputs))
        random.shuffle(pairs)

        val_pairs = pairs[:30]
        test_pairs = pairs[30:60]
        train_pairs = pairs[60:]

        for inp, out in train_pairs:
            target = first_answer(out)
            for cond_name, cond_prompt in CONDITIONS.items():
                train_rows.append({
                    "task": task,
                    "condition": cond_name,
                    "input": inp,
                    "target": target,
                    "source": build_source(task, inp, cond_prompt)
                })

        for inp, out in val_pairs:
            val_rows.append({
                "task": task,
                "input": inp,
                "answers": out if isinstance(out, list) else [out]
            })

        for inp, out in test_pairs:
            test_rows.append({
                "task": task,
                "input": inp,
                "answers": out if isinstance(out, list) else [out]
            })

with open("/kaggle/working/lora_train.jsonl", "w", encoding="utf-8") as f:
    for row in train_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open("/kaggle/working/lora_val.jsonl", "w", encoding="utf-8") as f:
    for row in val_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open("/kaggle/working/lora_test.jsonl", "w", encoding="utf-8") as f:
    for row in test_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done.")
print("train:", len(train_rows))
print("val:", len(val_rows))
print("test:", len(test_rows))