from main import run
import random
import numpy as np
import torch
import csv
import time
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# MODELS
# =========================

models = ["t5", "llama2", "vicuna"]

# =========================
# TASKS (5 comme demandé)
# =========================

tasks_list = [
    "sentiment",
    "translation_en-fr",
    "ruin_names",
    "word_sorting",
    "object_counting"
]

# =========================
# CONDITIONS
# =========================

conditions = [
    ("baseline", 0, ""),
    ("np01", 1, ""),
    ("neutral_salience", 0, "IMPORTANT: "),
    ("neutral_length", 0, "Please answer the following task carefully and accurately. ")
]

output_file = "results_controls_all_models.csv"

file_exists = os.path.exists(output_file)

with open(output_file, mode="a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "model",
            "task",
            "condition",
            "score"
        ])

    for model in models:

        print(f"\n========== MODEL: {model} ==========\n")

        for task in tasks_list:

            print(f"\n----- Task: {task} -----\n")

            for condition_name, stimulus, prefix in conditions:

                print(f"Running {model} | {task} | {condition_name}")

                start_time = time.time()

                score, details = run(
                    task,
                    model,
                    stimulus,
                    False,
                    return_details=True,
                    custom_prefix=prefix
                )

                elapsed = round(time.time() - start_time, 2)

                print(f"Score: {score} | Time: {elapsed}s\n")

                writer.writerow([
                    model,
                    task,
                    condition_name,
                    score
                ])

                file.flush()

print("Fini.")