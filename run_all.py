from main import run
from data.instruction_induction.load_data import tasks
import csv
import time

# ==============================
# CONFIGURATION
# ==============================

models = ["t5"]

#  limite à 10 tâches pour test
tasks_list = [
    "sentiment",
    "translation_en-fr",
    "word_in_context"
]

stimuli = [0, 1, 5, 10]

output_file = "results_phase2.csv"

# ==============================
# EXECUTION
# ==============================

total_runs = len(models) * len(tasks_list) * len(stimuli)
current_run = 0

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "task", "stimulus", "score", "time_sec"])

    for model in models:
        print(f"\n========== MODEL: {model} ==========\n")

        for task in tasks_list:
            print(f"\n----- Task: {task} -----\n")

            for stimulus in stimuli:

                current_run += 1
                print(f"[{current_run}/{total_runs}] Model={model} | Task={task} | Stimulus={stimulus}")

                start_time = time.time()

                score = run(task, model, stimulus, False)

                elapsed = round(time.time() - start_time, 2)

                writer.writerow([model, task, stimulus, score, elapsed])
                print(f"Score: {score} | Time: {elapsed}s\n")