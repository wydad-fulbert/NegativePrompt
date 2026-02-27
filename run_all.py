from main import run
from data.instruction_induction.load_data import tasks
import random
import numpy as np
import torch
import csv
import time


# ==============================
# SEED FIXE
# ==============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================
# CONFIGURATION
# ==============================

models = ["llama2"]

# 
tasks_list = ["sentiment"]
stimuli = [0]

output_file = "results_phase2.csv"

# ==============================
# EXECUTION
# ==============================

total_runs = len(models) * len(tasks_list) * len(stimuli)
current_run = 0

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
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

                current_run += 1
                print(f"[{current_run}/{total_runs}] Model={model} | Task={task} | Stimulus={stimulus}")

                start_time = time.time()

                score, details = run(task, model, stimulus, False,return_details= True)

                elapsed = round(time.time() - start_time, 2)

                writer.writerow([
                           model,
                           task,
                           stimulus,
                           score,
                           elapsed,
                           details["prompt"],
                           details["metric"],
                           "exec_accuracy_evaluator",
                           0.0,  #temperature
                           1.0,  #top_p
                           50,   #max_new_tokens
                           False,  #do_sample
                           False, #few_shot
                           details["num_samples"],
                           SEED
])
                print(f"Score: {score} | Time: {elapsed}s\n")



writer.writerow([
    model,
    task,
    stimulus,
    score,
    elapsed,
    details["prompt"],
    details["metric"],
    0.0,      # temperature
    1.0,      # top_p
    50,       # max_new_tokens
    False,    # do_sample
    False,    # few_shot
    details["num_samples"],
    SEED,
])