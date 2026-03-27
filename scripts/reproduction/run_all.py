from main import run
from data.instruction_induction.load_data import tasks
import random
import numpy as np
import torch
import csv
import time
import os 

file_exists = os.path.exists(output_file)and os.path.getsize(output_file) > 0

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

models = ["t5","llama2","vicuna"]

# 
tasks_list = [
    # Instruction Induction (5)
    "sentiment",
    "translation_en-fr",
    "word_in_context",
    "active_to_passive",
    "negation",

    # BigBench (5)
    "dyck_languages",
    "object_counting",
    "ruin_names",
    "word_sorting",
    "disambiguation_qa",
]
stimuli = [0,1,5,10]

output_file = "results_phase2.csv"

# ==============================
# EXECUTION
# ==============================

total_runs = len(models) * len(tasks_list) * len(stimuli)
current_run = 0

with open(output_file, mode="a", newline="") as file:
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

                print(f"Score: {score} | Time: {elapsed}s\n")
