import json
import random
from data.instruction_induction.load_data import load_data
from config import PROMPT_SET, Negative_SET

SEED = 42
random.seed(SEED)

TASKS = [
    "sentiment",
    "word_in_context",
    "object_counting"
]

TASK_PROMPTS = {
    **PROMPT_SET,
    "object_counting": "Count the number of objects described in the input."
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
test_rows = []

for task in TASKS:
    induce_inputs, induce_outputs = load_data("induce", task)
    eval_inputs, eval_outputs = load_data("eval", task)

    # train = induce
    for inp, out in zip(induce_inputs, induce_outputs):
        target = first_answer(out)
        for cond_name, cond_prompt in CONDITIONS.items():
            train_rows.append({
                "task": task,
                "condition": cond_name,
                "input": inp,
                "target": target,
                "source": build_source(task, inp, cond_prompt)
            })

    # test = 40 exemples
    for inp, out in zip(eval_inputs[:40], eval_outputs[:40]):
        answers = out if isinstance(out, list) else [out]
        test_rows.append({
            "task": task,
            "input": inp,
            "answers": answers
        })

with open("/kaggle/working/lora_train_targeted.jsonl", "w", encoding="utf-8") as f:
    for row in train_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open("/kaggle/working/lora_test_targeted.jsonl", "w", encoding="utf-8") as f:
    for row in test_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done.")
print("train:", len(train_rows))
print("test:", len(test_rows))