import subprocess
import csv

# ==============================
# CONFIGURATION
# ==============================

models = ["t5"]  # On pourra ajouter "llama", etc plus tard

tasks = [
    "sum",
    "sentiment",
    "word_in_context"
    # On ajoutera les autres tâches ensuite
]

stimuli = list(range(0, 11))  # 0 = baseline, 1-10 = NP01-NP10

output_file = "results.csv"

# ==============================
# EXECUTION
# ==============================

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "task", "stimulus", "score"])

    for model in models:
        for task in tasks:
            for stimulus in stimuli:

                print(f"\nRunning: Model={model} | Task={task} | Stimulus={stimulus}")

                command = [
                    "python",
                    "main.py",
                    task,
                    model,
                    str(stimulus),
                    "False"
                ]

                result = subprocess.run(command, capture_output=True, text=True)

                # Cherche la ligne contenant "Test score:"
                score = None
                for line in result.stdout.split("\n"):
                    if "Test score:" in line:
                        score = line.split(":")[-1].strip()

                if score is not None:
                    writer.writerow([model, task, stimulus, score])
                    print(f"Saved score: {score}")
                else:
                    print("Score not found!")