from main import run
import csv
import os
import time
from data.instruction_induction.load_data import tasks

# ==============================
# CONFIGURATION
# ==============================

models = ["t5"]  # ajouter "llama" plus tard
stimuli = list(range(0, 11))
output_file = "results.csv"

# ==============================
# LOAD EXISTING RESULTS (if any)
# ==============================

existing = set()

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            existing.add((row[0], row[1], int(row[2])))

# ==============================
# EXECUTION
# ==============================

with open(output_file, mode="a", newline="") as file:
    writer = csv.writer(file)

    if os.stat(output_file).st_size == 0:
        writer.writerow(["model", "task", "stimulus", "score", "time_sec"])

    for model in models:
        for task in tasks:
            for stimulus in stimuli:

                if (model, task, stimulus) in existing:
                    print(f"Skipping {model}-{task}-{stimulus}")
                    continue

                print(f"\nRunning: Model={model} | Task={task} | Stimulus={stimulus}")

                start = time.time()
                score = run(task, model, stimulus, False)
                end = time.time()

                duration = round(end - start, 2)

                writer.writerow([model, task, stimulus, score, duration])
                file.flush()

                print(f"Saved score: {score} | Time: {duration}s")