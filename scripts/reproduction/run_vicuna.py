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

models = ["vicuna"]

tasks_list = [
    "sentiment",
    "translation_en-fr",
    "word_in_context",
    "active_to_passive",
    "negation",
    "dyck_languages",
    "object_counting",
    "ruin_names",
    "word_sorting",
    "disambiguation_qa",
]

stimuli = [0,1,5,10]

output_file = "results_vicuna.csv"

total_runs = len(models)*len(tasks_list)*len(stimuli)
current_run = 0

file_exists = os.path.exists(output_file)

with open(output_file, mode="a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "model",
            "task",
            "stimulus",
            "score",
            "time_sec",
            "prompt",
            "metric",
            "evaluation_method",
            "temperature",
            "top_p",
            "max_new_tokens",
            "do_sample",
            "few_shot",
            "num_samples",
            "seed",
        ])

    for model in models:

        print(f"\n========== MODEL: {model} ==========\n")

        for task in tasks_list:

            print(f"\n----- Task: {task} -----\n")

            for stimulus in stimuli:

                stimulus_label = f"{stimulus} (Baseline)" if stimulus == 0 else stimulus

                current_run += 1

                print(f"[{current_run}/{total_runs}] Model={model} | Task={task} | Stimulus={stimulus_label}")

                start_time = time.time()

                score, details = run(task, model, stimulus, False, return_details=True)

                elapsed = round(time.time() - start_time, 2)

                print(f"Score: {score} | Time: {elapsed}s\n")

                writer.writerow([
                    model,
                    task,
                    stimulus_label,
                    score,
                    elapsed,
                    details["prompt"],
                    details["metric"],
                    "exec_accuracy_evaluator",
                    0.0,
                    1.0,
                    50,
                    False,
                    False,
                    details["num_samples"],
                    SEED
                ])

                file.flush()




print(f"Fini. Resultats sauvegardés dans : {output_file}")