import json
import fire
from data.instruction_induction.load_data import load_data, tasks
from exec_accuracy import exec_accuracy_evaluator
from config import PROMPT_SET, APE_PROMPT_SET, APE_PROMPTs, Negative_SET
import template
import os
import random



def load_bigbench(task):
    path = f"data/bigbench/{task}/task.json"

    with open(path, "r") as f:
        data = json.load(f)

    inputs = []
    outputs = []

    for example in data["examples"]:
        inputs.append(example["input"])

        # Cas 1 : target simple
        if "target" in example:
            if isinstance(example["target"], list):
                outputs.append(example["target"])
            else:
                outputs.append([example["target"]])

        # Cas 2 : target_scores (multiple choice)
        elif "target_scores" in example:
            correct_answers = [
                k for k, v in example["target_scores"].items() if v == 1
            ]
            outputs.append(correct_answers)

        else:
            raise ValueError(f"Unknown format in task {task}")

    return inputs, outputs



def getPrompt(ori_prompt, num_str):
    new_prompt = ori_prompt
    if num_str > 0:
        new_prompt = ori_prompt + Negative_SET[num_str - 1]
    return new_prompt


def run(task, model, pnum, few_shot):

    from data.instruction_induction.load_data import tasks as instruction_tasks

    num_demos = 5
    # ========================
    # LOAD DATA (II or BIGBENCH)
    # ========================

    if task in instruction_tasks:
        test_data = load_data('eval', task)
        induce_data = load_data('induce', task)

        few_shot_data = induce_data[0], [
            random.sample(output, 1)[0]
            for output in induce_data[1]
        ]

    else:
        test_data = load_bigbench(task)
        few_shot_data = ([], [])
        num_demos = 0

    # ========================
    # PROMPT
    # ========================

    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nAnswer: [OUTPUT]"
    origin_prompt = PROMPT_SET.get(task, "Solve the task carefully.")

    demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")
    eval_template = template.EvalTemplate(eval_template)

    new_prompt = getPrompt(origin_prompt, pnum)

    test_num = min(100, len(test_data[0]))

    # ========================
    # EVALUATION
    # ========================

    test_res = exec_accuracy_evaluator(
        prompts=[new_prompt],
        eval_template=eval_template,
        eval_data=test_data,
        llm_model=model,
        pnum=pnum,
        task=task,
        num_samples=test_num,
        few_shot=few_shot,
        demos_template=demos_template,
        few_shot_data=few_shot_data,
        num_demos= num_demos
    )

    test_score = test_res.sorted()[1][0]

    # ========================
    # SAVE RESULTS
    # ========================

    dir_path = f'results/neg/{model}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(f'{dir_path}/{task}.txt', 'a+') as f:
        f.write(f'Test score: {test_score}\n')
        f.write(f'Prompt(few-shot: {few_shot}): {new_prompt}\n')

    return test_score


if __name__ == '__main__':
    fire.Fire(run)
