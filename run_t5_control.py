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

model = "t5"

tasks_list = [
    "sentiment",
    "translation_en-fr",
    "active_to_passive",
    "ruin_names",
    "word_sorting"
]

conditions = [
    "baseline",
    "np01",
    "neutral_salience",
    "neutral_length",
]

output_file = "results_t5_controls.csv"

total_runs = len(tasks_list) * len(conditions)
current_run = 0

file_exists = os.path.exists(output_file)

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "model",
        "task",
        "condition",
        "score",
        "time_sec",
        "prompt_type",
        "seed",
    ])

    for task in tasks_list:
        print(f"\n===== TASK: {task} =====\n")

        for condition in conditions:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Model={model} | Task={task} | Condition={condition}")

            start_time = time.time()

            if condition == "baseline":
                stimulus = 0
                score, details = run(task, model, stimulus, False, return_details=True)

            elif condition == "np01":
                stimulus = 1
                score, details = run(task, model, stimulus, False, return_details=True)

            elif condition == "neutral_salience":
                score, details = run(
                    task,
                    model,
                    0,
                    False,
                    return_details=True,
                    custom_prefix="IMPORTANT: Please answer carefully.\n"
                )

            elif condition == "neutral_length":
                score, details = run(
                    task,
                    model,
                    0,
                    False,
                    return_details=True,
                    custom_prefix="Please read the instruction carefully and provide a clear answer without adding unnecessary information.\n"
                )

            elapsed = round(time.time() - start_time, 2)

            print(f"Score: {score} | Time: {elapsed}s\n")

            writer.writerow([
                model,
                task,
                condition,
                score,
                elapsed,
                condition,
                SEED
            ])

            file.flush()

print(f"Fini. Résultats sauvegardés dans : {output_file}")

